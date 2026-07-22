import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .dataset import Stage1SharedLatentDataset, collate_stage1
from .model import Stage1Config, Stage1SharedLatentModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 1 dual modal/capacity alignment")
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
    parser.add_argument("--output-dir", type=str, default="results/stage1_shared_latent")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--latent-dim", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--output-dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)

    parser.add_argument("--lambda-modal", type=float, default=1.0)
    parser.add_argument("--lambda-task", type=float, default=1.0)
    parser.add_argument(
        "--lambda-gemini-target",
        type=float,
        default=0.0,
        help="Optional supervised loss from generated embedding to paired image Gemini embedding.",
    )
    parser.add_argument(
        "--task-latent-source",
        choices=("text", "fused"),
        default="text",
        help="Use prompt-only latent z_text or paired unified latent z_student for L_x-task.",
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


def compute_losses(
    model: Stage1SharedLatentModel,
    text_emb: torch.Tensor,
    image_emb: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    out = model(text_emb, image_emb)

    loss_modal = F.mse_loss(out["z_text"], out["z_image"])
    if args.task_latent_source == "text":
        z = out["z_text"]
        z_hat = out["z_hat_text"]
        generated = out["generated_from_text"]
    else:
        z = out["z_student"]
        z_hat = out["z_hat_fused"]
        generated = out["generated_from_fused"]

    loss_task = F.mse_loss(z, z_hat)
    loss_gemini_target = F.mse_loss(generated, image_emb) if args.lambda_gemini_target > 0 else torch.zeros(
        (), device=text_emb.device
    )
    loss_total = (
        args.lambda_modal * loss_modal
        + args.lambda_task * loss_task
        + args.lambda_gemini_target * loss_gemini_target
    )

    metrics = {
        "loss_total": loss_total.detach(),
        "loss_modal": loss_modal.detach(),
        "loss_task": loss_task.detach(),
        "loss_gemini_target": loss_gemini_target.detach(),
        "latent_cosine": out["text_image_cosine"].mean().detach(),
        "latent_norm_text": out["z_text"].norm(dim=-1).mean().detach(),
        "latent_norm_image": out["z_image"].norm(dim=-1).mean().detach(),
    }
    return loss_total, metrics


def train_one_epoch(
    model: Stage1SharedLatentModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    log_fh,
) -> Tuple[Dict[str, float], int]:
    model.train()
    running = {
        "loss_total": 0.0,
        "loss_modal": 0.0,
        "loss_task": 0.0,
        "loss_gemini_target": 0.0,
        "latent_cosine": 0.0,
        "latent_norm_text": 0.0,
        "latent_norm_image": 0.0,
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
            "loss_gemini_target",
            "latent_cosine",
            "latent_norm_text",
            "latent_norm_image",
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
                f"total={row['loss_total']:.6f} "
                f"cos={row['latent_cosine']:.4f} "
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage1] device={device}")

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
        print(f"[Stage1] Using subset of {n} samples")
    print(f"[Stage1] samples={len(dataset)}")

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
            f"Text/Image embedding dims differ: text_dim={text_dim}, image_dim={image_dim}. "
            "Set --output-dim explicitly if this is intentional."
        )
    latent_dim = text_dim if args.latent_dim <= 0 else args.latent_dim
    output_dim = text_dim if args.output_dim <= 0 else args.output_dim

    cfg = Stage1Config(
        text_dim=text_dim,
        image_dim=image_dim,
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        dropout=args.dropout,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
    )
    model = Stage1SharedLatentModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{run_name}.jsonl"
    print(f"[Stage1] config={asdict(cfg)}")
    print(
        "[Stage1] losses="
        f"{args.lambda_modal}*L_x-modal + {args.lambda_task}*L_x-task"
        + (f" + {args.lambda_gemini_target}*L_gemini_target" if args.lambda_gemini_target > 0 else "")
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
                epoch=epoch,
                global_step=global_step,
                log_fh=log_fh,
            )
            row = {"epoch": epoch, "global_step": global_step, **metrics}
            print(
                f"[epoch {epoch}] "
                f"modal={metrics['loss_modal']:.6f} "
                f"task={metrics['loss_task']:.6f} "
                f"total={metrics['loss_total']:.6f} "
                f"cos={metrics['latent_cosine']:.4f}"
            )
            log_fh.write(json.dumps(row) + "\n")
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
                print(f"[Stage1] Saved checkpoint: {ckpt_path}")

    final_path = output_dir / "stage1_shared_latent_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "args": vars(args),
        },
        final_path,
    )
    print(f"[Stage1] Saved final model: {final_path}")
    print(f"[Stage1] Logs: {log_path}")


if __name__ == "__main__":
    main()
