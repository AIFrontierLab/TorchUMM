import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

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


class Stage2RolloutModel(nn.Module):
    """
    Stage 2 model built on Stage 1 shared-latent backbone with differentiable
    cycle-consistency heads for latent -> image_hat -> image_emb_hat -> text_hat_emb.
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
            )
        )

        # Differentiable rollout/cycle modules.
        self.latent_to_image = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.image_dim),
        )
        self.vision_encoder = nn.Sequential(
            nn.Linear(cfg.image_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.image_dim),
        )
        self.image_to_text = nn.Sequential(
            nn.Linear(cfg.image_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.text_dim),
        )

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.backbone(text_emb, image_emb)
        z_student = out["z_student"]

        # Student rollout path (differentiable)
        image_hat = self.latent_to_image(z_student)
        image_emb_hat = self.vision_encoder(image_hat)
        text_emb_hat = self.image_to_text(image_emb_hat)

        out.update(
            {
                "image_hat": image_hat,
                "image_emb_hat": image_emb_hat,
                "text_emb_hat": text_emb_hat,
            }
        )
        return out

    def generate_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.backbone.generate_from_latent(z)

    def embed_output(self, output: torch.Tensor) -> torch.Tensor:
        return self.backbone.embedding_model(output)

    def align(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        return self.backbone.aligner(text_emb, image_emb)


def compute_preference_loss(
    score_1: torch.Tensor,
    score_2: torch.Tensor,
    trigger_mask: torch.Tensor,
) -> torch.Tensor:
    """
    L_pref = -log(sigmoid(score_w - score_l)), only for triggered samples.
    """
    trigger_count = int(trigger_mask.sum().item())
    if trigger_count == 0:
        return torch.zeros((), device=score_1.device)

    score_w = torch.maximum(score_1, score_2)
    score_l = torch.minimum(score_1, score_2)
    pref_all = -F.logsigmoid(score_w - score_l)
    return pref_all[trigger_mask].mean()


def compute_cycle_loss(
    model: Stage2RolloutModel,
    z_input: torch.Tensor,
    prompt_emb: torch.Tensor,
    trigger_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cycle path:
      latent -> image_hat -> vision_encoder -> text_hat_emb
    L_cycle = 1 - cosine(text_hat_emb, prompt_emb)

    Returns:
      scalar loss over triggered samples (or zero)
      per-sample cycle losses
    """
    image_hat = model.latent_to_image(z_input)
    image_emb_hat = model.vision_encoder(image_hat)
    text_hat_emb = model.image_to_text(image_emb_hat)

    cycle_per_sample = 1.0 - F.cosine_similarity(text_hat_emb, prompt_emb, dim=-1)
    trigger_count = int(trigger_mask.sum().item())
    if trigger_count == 0:
        return torch.zeros((), device=z_input.device), cycle_per_sample

    cycle_loss = cycle_per_sample[trigger_mask].mean()
    return cycle_loss, cycle_per_sample


def rollout(
    model: Stage2RolloutModel,
    z_student: torch.Tensor,
    text_emb: torch.Tensor,
    image_emb: torch.Tensor,
    trigger_mask: torch.Tensor,
    noise_std: float,
    image_score_weight: float,
) -> Dict[str, torch.Tensor]:
    """
    Generate K=2 candidates from z_student and compute preference stats.
    """
    z_1 = z_student + noise_std * torch.randn_like(z_student)
    z_2 = z_student + noise_std * torch.randn_like(z_student)

    out_1 = model.generate_from_latent(z_1)
    out_2 = model.generate_from_latent(z_2)

    emb_1 = model.embed_output(out_1)
    emb_2 = model.embed_output(out_2)

    score_1 = F.cosine_similarity(emb_1, text_emb, dim=-1)
    score_2 = F.cosine_similarity(emb_2, text_emb, dim=-1)
    if image_score_weight > 0.0:
        score_1 = score_1 + image_score_weight * F.cosine_similarity(emb_1, image_emb, dim=-1)
        score_2 = score_2 + image_score_weight * F.cosine_similarity(emb_2, image_emb, dim=-1)

    pref_loss = compute_preference_loss(score_1, score_2, trigger_mask)

    choose_1 = score_1 >= score_2
    z_winner = torch.where(choose_1.unsqueeze(-1), z_1, z_2)

    return {
        "pref_loss": pref_loss,
        "z_winner": z_winner,
        "score_1": score_1,
        "score_2": score_2,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage2 rollout + preference + cycle training")

    parser.add_argument(
        "--prompts-path",
        type=str,
        default="<path-to-prompts-json>",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="<path-to-generated-image-root>",
    )
    parser.add_argument(
        "--embedding-root",
        type=str,
        default="<path-to-embedding-root>",
    )
    parser.add_argument(
        "--stage1-ckpt",
        type=str,
        default="<path-to-stage1-shared-latent-model-ckpt>",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="<path-to-stage2-output-dir>",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--latent-dim", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--output-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lambda-pref", type=float, default=0.1)
    parser.add_argument("--lambda-cycle", type=float, default=0.2)

    parser.add_argument("--rollout-every", type=int, default=5)
    parser.add_argument("--rollout-noise-std", type=float, default=0.05)
    parser.add_argument("--latent-trigger-threshold", type=float, default=0.90)
    parser.add_argument("--cycle-trigger-threshold", type=float, default=0.20)
    parser.add_argument("--image-score-weight", type=float, default=0.0)

    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)

    return parser.parse_args()


def _load_stage1_weights(model: Stage2RolloutModel, ckpt_path: str) -> None:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
    print(f"[Stage2] Loaded Stage1 checkpoint: {ckpt_path}")
    print(f"[Stage2] Stage1 load missing keys: {len(missing)}")
    print(f"[Stage2] Stage1 load unexpected keys: {len(unexpected)}")


def _build_trigger_reason(
    latent_trigger: torch.Tensor,
    cycle_trigger: torch.Tensor,
) -> Counter:
    reasons = Counter()
    both = latent_trigger & cycle_trigger
    latent_only = latent_trigger & (~cycle_trigger)
    cycle_only = cycle_trigger & (~latent_trigger)
    reasons["both"] = int(both.sum().item())
    reasons["latent_only"] = int(latent_only.sum().item())
    reasons["cycle_only"] = int(cycle_only.sum().item())
    reasons["none"] = int((~(latent_trigger | cycle_trigger)).sum().item())
    return reasons


def _set_backbone_trainable(model: Stage2RolloutModel, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = trainable


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
        "align": 0.0,
        "distill": 0.0,
        "pref": 0.0,
        "cycle": 0.0,
        "total": 0.0,
        "grad_norm": 0.0,
        "trigger_count": 0,
        "sample_count": 0,
        "optimizer_steps": 0,
        "skipped_backward": 0,
    }

    for step, batch in enumerate(dataloader):
        if args.max_steps > 0 and step >= args.max_steps:
            break

        text_emb = batch["text_embedding"].to(device)
        image_emb = batch["image_embedding"].to(device)

        out = model(text_emb, image_emb)
        z_student = out["z_student"]
        z_teacher = out["z_teacher"].detach()

        target_latent = 0.5 * (text_emb + image_emb)
        if target_latent.shape[-1] != z_student.shape[-1]:
            raise ValueError(
                f"L_align mismatch: z_student dim={z_student.shape[-1]}, target dim={target_latent.shape[-1]}"
            )

        loss_align = F.mse_loss(z_student, target_latent)
        loss_distill = F.mse_loss(z_student, z_teacher)

        # Trigger conditions (base)
        latent_cos = F.cosine_similarity(z_student, z_teacher, dim=-1)
        latent_trigger_base = latent_cos < args.latent_trigger_threshold

        cycle_proxy = 1.0 - F.cosine_similarity(out["text_emb_hat"], text_emb, dim=-1)
        cycle_trigger_base = cycle_proxy > args.cycle_trigger_threshold

        do_rollout_step = (global_step % max(1, args.rollout_every) == 0)
        if do_rollout_step:
            latent_trigger = latent_trigger_base
            cycle_trigger = cycle_trigger_base
        else:
            latent_trigger = torch.zeros_like(latent_trigger_base, dtype=torch.bool)
            cycle_trigger = torch.zeros_like(cycle_trigger_base, dtype=torch.bool)

        trigger_mask = latent_trigger | cycle_trigger
        trigger_count = int(trigger_mask.sum().item())
        trigger_reasons = _build_trigger_reason(latent_trigger, cycle_trigger)

        if trigger_count > 0:
            rollout_out = rollout(
                model=model,
                z_student=z_student,
                text_emb=text_emb,
                image_emb=image_emb,
                trigger_mask=trigger_mask,
                noise_std=args.rollout_noise_std,
                image_score_weight=args.image_score_weight,
            )
            loss_pref = rollout_out["pref_loss"]

            # Cycle consistency on winner candidate latent.
            loss_cycle, cycle_per_sample = compute_cycle_loss(
                model=model,
                z_input=rollout_out["z_winner"],
                prompt_emb=text_emb,
                trigger_mask=trigger_mask,
            )
        else:
            loss_pref = torch.zeros((), device=device)
            loss_cycle = torch.zeros((), device=device)
            cycle_per_sample = cycle_proxy

        loss_total = (
            loss_align
            + loss_distill
            + args.lambda_pref * loss_pref
            + args.lambda_cycle * loss_cycle
        )

        optimizer_step = False
        optimizer.zero_grad(set_to_none=True)
        if loss_total.requires_grad:
            loss_total.backward()

            if args.max_grad_norm > 0:
                grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                grad_norm = float(grad_norm_t.item())
            else:
                grad_norm = 0.0

            optimizer.step()
            optimizer_step = True
        else:
            # This can happen when backbone is frozen and rollout is inactive for this batch.
            grad_norm = 0.0

        bsz = text_emb.size(0)
        running["align"] += float(loss_align.item()) * bsz
        running["distill"] += float(loss_distill.item()) * bsz
        running["pref"] += float(loss_pref.item()) * bsz
        running["cycle"] += float(loss_cycle.item()) * bsz
        running["total"] += float(loss_total.item()) * bsz
        running["grad_norm"] += grad_norm * bsz
        running["trigger_count"] += trigger_count
        running["sample_count"] += bsz
        running["optimizer_steps"] += int(optimizer_step)
        running["skipped_backward"] += int(not optimizer_step)

        if step % args.log_interval == 0:
            log_row = {
                "global_step": global_step,
                "epoch": epoch,
                "epoch_step": step,
                "rollout_step": int(do_rollout_step),
                "backbone_trainable": int(any(p.requires_grad for p in model.backbone.parameters())),
                "loss_align": float(loss_align.item()),
                "loss_distill": float(loss_distill.item()),
                "loss_pref": float(loss_pref.item()),
                "loss_cycle": float(loss_cycle.item()),
                "loss_total": float(loss_total.item()),
                "grad_norm": grad_norm,
                "trigger_rate": trigger_count / max(1, bsz),
                "optimizer_step": int(optimizer_step),
                "latent_trigger_rate_base": float(latent_trigger_base.float().mean().item()),
                "cycle_trigger_rate_base": float(cycle_trigger_base.float().mean().item()),
                "mean_cycle_proxy": float(cycle_proxy.mean().item()),
                "mean_cycle_sample": float(cycle_per_sample.mean().item()),
                "trigger_reason": dict(trigger_reasons),
            }
            print(
                f"[step {global_step}] "
                f"align={log_row['loss_align']:.6f} "
                f"distill={log_row['loss_distill']:.6f} "
                f"pref={log_row['loss_pref']:.6f} "
                f"cycle={log_row['loss_cycle']:.6f} "
                f"total={log_row['loss_total']:.6f} "
                f"grad={log_row['grad_norm']:.4f} "
                f"trigger_rate={log_row['trigger_rate']:.4f} "
                f"opt_step={log_row['optimizer_step']} "
                f"reason={log_row['trigger_reason']}"
            )
            log_fh.write(json.dumps(log_row) + "\n")
            log_fh.flush()

        global_step += 1

    denom = max(1, running["sample_count"])
    metrics = {
        "loss_align": running["align"] / denom,
        "loss_distill": running["distill"] / denom,
        "loss_pref": running["pref"] / denom,
        "loss_cycle": running["cycle"] / denom,
        "loss_total": running["total"] / denom,
        "grad_norm": running["grad_norm"] / denom,
        "trigger_rate": running["trigger_count"] / denom,
        "optimizer_step_rate": running["optimizer_steps"] / max(1, (running["optimizer_steps"] + running["skipped_backward"])),
    }
    return metrics, global_step


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

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
        load_image=False,
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
    latent_dim = text_dim if args.latent_dim <= 0 else args.latent_dim

    if text_dim != image_dim:
        raise ValueError(
            f"Text/Image embedding dimensions differ: text_dim={text_dim}, image_dim={image_dim}."
        )

    cfg = Stage2Config(
        text_dim=text_dim,
        image_dim=image_dim,
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
    )
    model = Stage2RolloutModel(cfg).to(device)

    if args.stage1_ckpt:
        _load_stage1_weights(model, args.stage1_ckpt)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{run_name}.jsonl"

    global_step = 0
    with open(log_path, "w", encoding="utf-8") as log_fh:
        for epoch in range(args.epochs):
            backbone_trainable = epoch >= args.freeze_backbone_epochs
            _set_backbone_trainable(model, backbone_trainable)
            print(
                f"[Stage2] epoch={epoch} backbone_trainable={backbone_trainable} "
                f"freeze_backbone_epochs={args.freeze_backbone_epochs}"
            )

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

            epoch_row = {
                "epoch": epoch,
                "global_step": global_step,
                "backbone_trainable": int(backbone_trainable),
                **metrics,
            }
            print(
                f"[epoch {epoch}] "
                f"align={metrics['loss_align']:.6f} "
                f"distill={metrics['loss_distill']:.6f} "
                f"pref={metrics['loss_pref']:.6f} "
                f"cycle={metrics['loss_cycle']:.6f} "
                f"total={metrics['loss_total']:.6f} "
                f"grad={metrics['grad_norm']:.4f} "
                f"trigger_rate={metrics['trigger_rate']:.4f}"
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
                        "args": vars(args),
                        "metrics": metrics,
                    },
                    ckpt_path,
                )
                print(f"[Stage2] Saved checkpoint: {ckpt_path}")

    final_model_path = output_dir / "stage2_rollout_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"[Stage2] Saved final model: {final_model_path}")
    print(f"[Stage2] Logs: {log_path}")


if __name__ == "__main__":
    main()
