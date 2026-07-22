import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from stage1_shared_latent_modules.dataset import Stage1SharedLatentDataset, collate_stage1
from stage1_shared_latent_modules.model import Stage1Config, Stage1SharedLatentModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Stage2Config:
    text_dim: int
    image_dim: int
    latent_dim: int
    hidden_dim: int
    output_dim: int
    dropout: float
    transformer_layers: int
    transformer_heads: int


class Stage2RolloutModel(nn.Module):
    """
    Stage 2 stochastic latent rollout model.

    The trainable backbone is the Stage 1 shared-latent model:
      phi_text, phi_image: Gemini embedding -> refined latent
      G: refined latent -> generated Gemini-space embedding

    Rollout trajectory:
      z_k = z + eps_k
      x_hat_k = G(z_k)
      z_hat_k = phi(x_hat_k)
    """

    def __init__(self, cfg: Stage2Config) -> None:
        super().__init__()
        self.backbone = Stage1SharedLatentModel(
            Stage1Config(
                text_dim=cfg.text_dim,
                image_dim=cfg.image_dim,
                latent_dim=cfg.latent_dim,
                hidden_dim=cfg.hidden_dim,
                output_dim=cfg.output_dim,
                dropout=cfg.dropout,
                transformer_layers=cfg.transformer_layers,
                transformer_heads=cfg.transformer_heads,
            )
        )

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.backbone(text_emb, image_emb)

    def generate_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.backbone.generate_from_latent(z)

    def embed_output(self, output: torch.Tensor, modality: str = "image") -> torch.Tensor:
        return self.backbone.embed_output(output, modality=modality)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 2 stochastic latent rollouts + preference optimization")

    parser.add_argument(
        "--prompts-path",
        type=str,
        default="/path/to/dataset/t2i/prompts.json",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="/path/to/dataset/t2i",
    )
    parser.add_argument(
        "--embedding-root",
        type=str,
        default="/path/to/dataset_embedding/t2i",
    )
    parser.add_argument("--text-embedding-dir", type=str, default="text_embedding")
    parser.add_argument("--image-embedding-dir", type=str, default="image_embedding")
    parser.add_argument("--stage1-ckpt", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="results/stage2_stochastic_rollout")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--latent-dim", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--output-dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)

    parser.add_argument("--lambda-task", type=float, default=1.0)
    parser.add_argument("--lambda-pref", type=float, default=0.1)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--rollout-noise-std", type=float, default=0.05)
    parser.add_argument(
        "--task-latent-source",
        choices=("text", "fused"),
        default="text",
        help="Use prompt-only z_text or paired unified z_student for L_x-task and stochastic rollouts.",
    )
    parser.add_argument("--normalize-inputs", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--cache-embeddings", action="store_true")

    return parser.parse_args()


def _maybe_normalize(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return x
    return F.normalize(x, dim=-1)


def _load_stage1_weights(model: Stage2RolloutModel, ckpt_path: str) -> None:
    if not ckpt_path:
        return

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported Stage1 checkpoint format: {ckpt_path}")

    if any(key.startswith("backbone.") for key in state_dict):
        state_dict = {
            key[len("backbone."):]: value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }

    missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
    print(f"[Stage2] Loaded Stage1 checkpoint: {ckpt_path}")
    print(f"[Stage2] Stage1 load missing keys: {len(missing)}")
    print(f"[Stage2] Stage1 load unexpected keys: {len(unexpected)}")


def stochastic_preference_loss(
    model: Stage2RolloutModel,
    z: torch.Tensor,
    num_rollouts: int,
    noise_std: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    z_k = z + eps_k, eps_k ~ N(0, sigma^2 I)
    x_hat_k = G(z_k)
    z_hat_k = phi(x_hat_k)
    s_k = cosine(z, z_hat_k)
    L_pref = -log sigmoid(max_k s_k - min_k s_k)
    """
    if num_rollouts < 2:
        raise ValueError("--num-rollouts must be >= 2 for preference optimization")

    scores = []
    z_hats = []
    for _ in range(num_rollouts):
        z_k = z + noise_std * torch.randn_like(z)
        x_hat_k = model.generate_from_latent(z_k)
        z_hat_k = model.embed_output(x_hat_k, modality="image")
        z_hats.append(z_hat_k)
        scores.append(F.cosine_similarity(z, z_hat_k, dim=-1))

    score_tensor = torch.stack(scores, dim=0)
    z_hat_tensor = torch.stack(z_hats, dim=0)
    best_score = score_tensor.max(dim=0).values
    worst_score = score_tensor.min(dim=0).values
    pref_per_sample = -F.logsigmoid(best_score - worst_score)

    return pref_per_sample.mean(), {
        "rollout_scores": score_tensor,
        "rollout_z_hats": z_hat_tensor,
        "best_score": best_score,
        "worst_score": worst_score,
        "score_margin": best_score - worst_score,
    }


def compute_losses(
    model: Stage2RolloutModel,
    text_emb: torch.Tensor,
    image_emb: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    out = model(text_emb, image_emb)

    loss_modal = F.mse_loss(out["z_text"], out["z_image"])
    if args.task_latent_source == "text":
        z = out["z_text"]
        z_hat = out["z_hat_text"]
    else:
        z = out["z_student"]
        z_hat = out["z_hat_fused"]

    loss_task = F.mse_loss(z, z_hat)
    loss_pref, rollout_stats = stochastic_preference_loss(
        model=model,
        z=z,
        num_rollouts=args.num_rollouts,
        noise_std=args.rollout_noise_std,
    )
    loss_total = loss_modal + args.lambda_task * loss_task + args.lambda_pref * loss_pref

    metrics = {
        "loss_total": loss_total.detach(),
        "loss_modal": loss_modal.detach(),
        "loss_task": loss_task.detach(),
        "loss_pref": loss_pref.detach(),
        "latent_cosine": out["text_image_cosine"].mean().detach(),
        "score_best": rollout_stats["best_score"].mean().detach(),
        "score_worst": rollout_stats["worst_score"].mean().detach(),
        "score_margin": rollout_stats["score_margin"].mean().detach(),
    }
    return loss_total, metrics


def train_one_epoch(
    model: Stage2RolloutModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    global_step: int,
    epoch: int,
    log_fh,
) -> Tuple[dict, int]:
    model.train()
    running = {
        "loss_total": 0.0,
        "loss_modal": 0.0,
        "loss_task": 0.0,
        "loss_pref": 0.0,
        "latent_cosine": 0.0,
        "score_best": 0.0,
        "score_worst": 0.0,
        "score_margin": 0.0,
        "grad_norm": 0.0,
        "sample_count": 0,
    }

    for step, batch in enumerate(dataloader):
        if args.max_steps > 0 and step >= args.max_steps:
            break

        text_emb = _maybe_normalize(batch["text_embedding"].to(device), args.normalize_inputs)
        image_emb = _maybe_normalize(batch["image_embedding"].to(device), args.normalize_inputs)

        optimizer.zero_grad(set_to_none=True)
        loss_total, metrics = compute_losses(model, text_emb, image_emb, args)
        loss_total.backward()

        if args.max_grad_norm > 0:
            grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            grad_norm = float(grad_norm_t.item())
        else:
            grad_norm = 0.0
        optimizer.step()

        bsz = text_emb.size(0)
        for key in (
            "loss_total",
            "loss_modal",
            "loss_task",
            "loss_pref",
            "latent_cosine",
            "score_best",
            "score_worst",
            "score_margin",
        ):
            running[key] += float(metrics[key].item()) * bsz
        running["grad_norm"] += grad_norm * bsz
        running["sample_count"] += bsz

        if step % args.log_interval == 0:
            row = {
                "global_step": global_step,
                "epoch": epoch,
                "epoch_step": step,
                "grad_norm": grad_norm,
                **{key: float(value.item()) for key, value in metrics.items()},
            }
            print(
                f"[step {global_step}] "
                f"modal={row['loss_modal']:.6f} "
                f"task={row['loss_task']:.6f} "
                f"pref={row['loss_pref']:.6f} "
                f"total={row['loss_total']:.6f} "
                f"margin={row['score_margin']:.4f} "
                f"grad={grad_norm:.4f}"
            )
            log_fh.write(json.dumps(row) + "\n")
            log_fh.flush()

        global_step += 1

    denom = max(1, running["sample_count"])
    metrics = {key: value / denom for key, value in running.items() if key != "sample_count"}
    return metrics, global_step


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.num_rollouts < 2:
        raise ValueError("--num-rollouts must be >= 2")
    if args.rollout_noise_std < 0:
        raise ValueError("--rollout-noise-std must be non-negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage2] device={device}")

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    dataset = Stage1SharedLatentDataset(
        prompts_path=args.prompts_path,
        image_root=args.image_root,
        embedding_root=args.embedding_root,
        text_embedding_dir=args.text_embedding_dir,
        image_embedding_dir=args.image_embedding_dir,
        load_image=False,
        cache_embeddings=args.cache_embeddings,
    )
    if args.limit_samples > 0:
        n = min(args.limit_samples, len(dataset))
        dataset = Subset(dataset, list(range(n)))
        print(f"[Stage2] Using subset of {n} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_stage1,
    )

    sample = dataset[0]
    text_dim = int(sample["text_embedding"].numel())
    image_dim = int(sample["image_embedding"].numel())
    if text_dim != image_dim and args.output_dim <= 0:
        raise ValueError(
            f"Text/Image embedding dimensions differ: text_dim={text_dim}, image_dim={image_dim}. "
            "Set --output-dim explicitly if this is intentional."
        )

    latent_dim = text_dim if args.latent_dim <= 0 else args.latent_dim
    output_dim = text_dim if args.output_dim <= 0 else args.output_dim
    cfg = Stage2Config(
        text_dim=text_dim,
        image_dim=image_dim,
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        dropout=args.dropout,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
    )
    model = Stage2RolloutModel(cfg).to(device)
    _load_stage1_weights(model, args.stage1_ckpt)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{run_name}.jsonl"
    print(f"[Stage2] samples={len(dataset)}")
    print(f"[Stage2] config={asdict(cfg)}")
    print(
        "[Stage2] objective="
        f"L_x-modal + {args.lambda_task}*L_x-task + {args.lambda_pref}*L_pref, "
        f"K={args.num_rollouts}, sigma={args.rollout_noise_std}"
    )

    global_step = 0
    with log_path.open("w", encoding="utf-8") as log_fh:
        for epoch in range(args.epochs):
            metrics, global_step = train_one_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                args=args,
                global_step=global_step,
                epoch=epoch,
                log_fh=log_fh,
            )

            epoch_row = {"epoch": epoch, "global_step": global_step, **metrics}
            print(
                f"[epoch {epoch}] "
                f"modal={metrics['loss_modal']:.6f} "
                f"task={metrics['loss_task']:.6f} "
                f"pref={metrics['loss_pref']:.6f} "
                f"total={metrics['loss_total']:.6f} "
                f"margin={metrics['score_margin']:.4f}"
            )
            log_fh.write(json.dumps(epoch_row) + "\n")
            log_fh.flush()

            if (epoch + 1) % args.save_every == 0:
                ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": asdict(cfg),
                        "args": vars(args),
                        "metrics": metrics,
                    },
                    ckpt_path,
                )
                print(f"[Stage2] Saved checkpoint: {ckpt_path}")

    final_model_path = output_dir / "stage2_stochastic_rollout_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "args": vars(args),
        },
        final_model_path,
    )
    print(f"[Stage2] Saved final model: {final_model_path}")
    print(f"[Stage2] Logs: {log_path}")


if __name__ == "__main__":
    main()
