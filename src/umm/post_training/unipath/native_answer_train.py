from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from umm.post_training.unipath.bagel_paths import add_bagel_code_to_sys_path

BAGEL_ROOT = add_bagel_code_to_sys_path(REPO_ROOT)

from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae

from umm.post_training.unipath.native_answer_batch import build_native_answer_pieces, build_single_native_answer_example
from umm.post_training.unipath.image_aware_train import (
    get_language_model_core,
    install_lora_on_language_model,
    install_partial_sft_on_language_model,
)
from umm.post_training.unipath.native_bridge import load_full_bagel_dispatch, select_native_visual_trainables
from umm.post_training.unipath.native_targets import iter_native_image_answer_rows
from umm.post_training.unipath.train import build_context_text, compute_visual_summary_loss, extract_structural_tag_sequence
from umm.post_training.unipath.train import expected_tag_sequence_for_row, format_matches_expected
from umm.post_training.unipath.train import maybe_load_trainable_model_state, maybe_load_visual_head, save_checkpoint_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small-scale native image-answer training smoke for UniPath.")
    parser.add_argument("--train-jsonl", action="append", required=True)
    parser.add_argument("--val-jsonl", action="append", required=True)
    parser.add_argument("--latent-cache-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-max-samples", type=int, default=16)
    parser.add_argument("--val-max-samples", type=int, default=8)
    parser.add_argument("--train-sample-selection", choices=["random", "longest", "stratified"], default="random")
    parser.add_argument("--val-sample-selection", choices=["random", "longest", "stratified"], default="random")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--eval-every-steps", type=int, default=4)
    parser.add_argument("--initial-eval", action="store_true")
    parser.add_argument("--save-best-metric", choices=["val_loss", "val_mse", "val_format_accuracy", "val_format_accuracy_minus_mse"], default="val_loss")
    parser.add_argument("--best-require-format-no-degrade", action="store_true")
    parser.add_argument("--best-format-tolerance", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--min-steps-before-early-stop", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--stop-on-zero-format", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--visual-summary-weight", type=float, default=0.2)
    parser.add_argument("--latent-target-dim", type=int, default=16)
    parser.add_argument("--eval-generate-max-samples", type=int, default=4)
    parser.add_argument("--eval-generate-sampling", choices=["head", "stratified"], default="stratified")
    parser.add_argument("--eval-generate-max-new-tokens", type=int, default=320)
    parser.add_argument("--eval-generate-image-samples", type=int, default=0)
    parser.add_argument("--eval-generate-image-steps", type=int, default=8)
    parser.add_argument("--eval-generate-image-size", type=int, default=512)
    parser.add_argument("--save-eval-cases", action="store_true")
    parser.add_argument("--save-last", action="store_true")
    parser.add_argument("--max-mem-per-gpu", default="80GiB")
    parser.add_argument("--offload-folder", default="/tmp/unipath_native_answer_offload")
    parser.add_argument("--finetune-mode", choices=["lora", "partial_sft", "bridge"], default="lora")
    parser.add_argument("--train-last-n-layers", type=int, default=8)
    parser.add_argument("--train-lm-head", action="store_true")
    parser.add_argument("--train-final-norm", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume-adapter-path", default=None)
    parser.add_argument("--resume-visual-head-path", default=None)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return parser.parse_args()


def native_best_metric_is_higher(metric_name: str) -> bool:
    return metric_name in {"val_format_accuracy", "val_format_accuracy_minus_mse"}


def native_best_metric_value(metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name == "val_format_accuracy_minus_mse":
        return float(metrics.get("val_format_accuracy", 0.0) - metrics.get("val_mse", 0.0))
    return float(metrics[metric_name])


def native_metric_improved(metric_value: float, best_value: float, metric_name: str, min_delta: float = 0.0) -> bool:
    delta = max(min_delta, 0.0)
    return metric_value > best_value + delta if native_best_metric_is_higher(metric_name) else metric_value < best_value - delta


def load_vae_for_generation(model_path: Path, device: torch.device) -> torch.nn.Module:
    vae_model, _ = load_ae(local_path=str(model_path / "ae.safetensors"))
    vae_model = vae_model.to(device=device).eval()
    vae_dtype = next(vae_model.parameters()).dtype
    vae_device = next(vae_model.parameters()).device
    original_encode = vae_model.encode
    original_decode = vae_model.decode

    def encode_with_cast(x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return original_encode(x.to(device=vae_device, dtype=vae_dtype), *args, **kwargs)

    def decode_with_cast(latent: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return original_decode(latent.to(device=vae_device, dtype=vae_dtype), *args, **kwargs)

    vae_model.encode = encode_with_cast
    vae_model.decode = decode_with_cast
    return vae_model


def generate_reasoning_text(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    new_token_ids: dict[str, int],
    prompt_text: str,
    max_new_tokens: int,
) -> str:
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=None,
        tokenizer=tokenizer,
        vae_transform=ImageTransform(1024, 512, 16),
        vit_transform=ImageTransform(980, 224, 14),
        new_token_ids=new_token_ids,
    )
    was_training = model.training
    model.eval()
    language_model = getattr(model, "language_model", None)
    language_model_core = get_language_model_core(language_model) if language_model is not None else None
    if language_model_core is not None and language_model_core is not language_model:
        model.language_model = language_model_core
    try:
        with torch.no_grad():
            result = inferencer(
                image=None,
                text=prompt_text,
                understanding_output=True,
                think=False,
                do_sample=False,
                max_think_token_n=max_new_tokens,
            )
    finally:
        if language_model_core is not None and language_model_core is not language_model:
            model.language_model = language_model
        if was_training:
            model.train()
    if isinstance(result, dict):
        return str(result.get("text") or "")
    return str(result or "")


def generate_reasoning_image(
    *,
    model: torch.nn.Module,
    vae_model: torch.nn.Module,
    tokenizer: Any,
    new_token_ids: dict[str, int],
    prompt_text: str,
    image_size: int,
    num_timesteps: int,
    max_new_tokens: int,
) -> tuple[str, Any]:
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=ImageTransform(1024, 512, 16),
        vit_transform=ImageTransform(980, 224, 14),
        new_token_ids=new_token_ids,
    )
    was_training = model.training
    model.eval()
    language_model = getattr(model, "language_model", None)
    language_model_core = get_language_model_core(language_model) if language_model is not None else None
    if language_model_core is not None and language_model_core is not language_model:
        model.language_model = language_model_core
    try:
        with torch.no_grad():
            result = inferencer(
                image=None,
                text=prompt_text,
                understanding_output=False,
                think=True,
                do_sample=False,
                max_think_token_n=max_new_tokens,
                image_shapes=(image_size, image_size),
                num_timesteps=num_timesteps,
            )
    finally:
        if language_model_core is not None and language_model_core is not language_model:
            model.language_model = language_model
        if was_training:
            model.train()
    if not isinstance(result, dict):
        raise TypeError(f"Expected inferencer result dict, got {type(result)}")
    generated_text = str(result.get("text") or "")
    generated_image = result.get("image")
    if generated_image is None:
        raise RuntimeError("single_context_image_generation_failed: inferencer returned no image")
    return generated_text, generated_image


def move_batch_to_device(packed: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in packed.items():
        if isinstance(value, list) and value and all(isinstance(item, torch.Tensor) for item in value):
            moved[key] = [item.to(device) for item in value]
            continue
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total) if total > 0 else 0.0


def build_rows(
    paths: list[str],
    *,
    seed: int,
    max_samples: int,
    sample_selection: str,
) -> list[dict[str, Any]]:
    return iter_native_image_answer_rows(
        [Path(item).expanduser() for item in paths],
        seed=seed,
        max_samples=max_samples,
        sample_selection=sample_selection,
    )


def select_fixed_generate_row_ids(
    rows: list[dict[str, Any]],
    *,
    max_generate_samples: int,
    generate_sampling: str,
) -> set[str]:
    if max_generate_samples <= 0:
        return set()
    if generate_sampling == "stratified":
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        level_order: list[str] = []
        for row in rows:
            level = str(row.get("reasoning_level") or "unknown")
            if level not in groups:
                level_order.append(level)
            groups[level].append(row)
        cursors = {level: 0 for level in level_order}
        selected_rows: list[dict[str, Any]] = []
        while len(selected_rows) < max_generate_samples:
            progressed = False
            for level in level_order:
                cursor = cursors[level]
                bucket = groups[level]
                if cursor < len(bucket):
                    selected_rows.append(bucket[cursor])
                    cursors[level] = cursor + 1
                    progressed = True
                    if len(selected_rows) >= max_generate_samples:
                        break
            if not progressed:
                break
    else:
        selected_rows = rows[:max_generate_samples]
    return {str(row.get("id") or row.get("row_id") or "") for row in selected_rows}


def compute_losses(
    model: torch.nn.Module,
    visual_head: torch.nn.Module,
    example: Any,
    device: torch.device,
    ce_weight: float,
    mse_weight: float,
    visual_summary_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = move_batch_to_device(example.packed, device)
    visual_summary_spans = batch.pop("visual_summary_spans")
    visual_summary_targets = batch.pop("visual_summary_targets")
    model_batch = {key: value for key, value in batch.items() if key != "ce_loss_weights"}
    model_batch["return_hidden_state"] = True
    language_model = getattr(model, "language_model", None)
    language_model_core = get_language_model_core(language_model) if language_model is not None else None
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        if language_model_core is not None and language_model_core is not language_model:
            model.language_model = language_model_core
            try:
                outputs = model(**model_batch)
            finally:
                model.language_model = language_model
        else:
            outputs = model(**model_batch)
    ce_values = outputs["ce"].float()
    ce_weights = batch["ce_loss_weights"].to(device=ce_values.device, dtype=ce_values.dtype)
    ce = (ce_values * ce_weights).sum() / ce_weights.sum().clamp_min(1e-8)
    mse = outputs["mse"].float().mean()
    hidden_states = outputs["last_hidden_state"].unsqueeze(0)
    if visual_summary_spans.numel() > 0:
        visual_mask = torch.ones((1, visual_summary_spans.size(0)), device=device, dtype=torch.float32)
        visual_loss = compute_visual_summary_loss(
            hidden_states,
            visual_summary_spans.unsqueeze(0),
            visual_summary_targets.unsqueeze(0),
            visual_mask,
            visual_head,
        )
    else:
        visual_loss = next(visual_head.parameters()).sum() * 0.0 + hidden_states.sum() * 0.0
    loss = ce * ce_weight + mse * mse_weight + visual_loss * visual_summary_weight
    return loss, ce, mse, visual_loss.float()


def evaluate(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    new_token_ids: dict[str, int],
    rows: list[dict[str, Any]],
    model_path: Path,
    latent_cache_root: Path,
    device: torch.device,
    visual_head: torch.nn.Module,
    ce_weight: float,
    mse_weight: float,
    visual_summary_weight: float,
    latent_target_dim: int,
    max_generate_samples: int,
    generate_sampling: str,
    max_generate_new_tokens: int,
    max_generate_image_samples: int,
    generate_image_steps: int,
    generate_image_size: int,
    case_output_path: Path | None,
    image_output_dir: Path | None,
    fixed_generate_row_ids: set[str] | None = None,
) -> dict[str, Any]:
    was_training = model.training
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_mse = 0.0
    total_visual = 0.0
    format_hits = 0
    generated_format_hits = 0
    generated_count = 0
    generated_image_count = 0
    vae_model: torch.nn.Module | None = None
    count = 0
    cases: list[dict[str, Any]] = []
    by_level: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "loss": 0.0,
            "ce": 0.0,
            "mse": 0.0,
            "visual": 0.0,
            "target_format": 0.0,
            "generated_count": 0.0,
            "generated_format": 0.0,
        }
    )
    generate_indices: set[int] = set()
    if fixed_generate_row_ids is not None:
        for idx, row in enumerate(rows):
            row_id = str(row.get("id") or row.get("row_id") or "")
            if row_id in fixed_generate_row_ids:
                generate_indices.add(idx)
    elif max_generate_samples > 0:
        if generate_sampling == "stratified":
            groups: dict[str, list[int]] = defaultdict(list)
            level_order: list[str] = []
            for idx, row in enumerate(rows):
                level = str(row.get("reasoning_level") or "unknown")
                if level not in groups:
                    level_order.append(level)
                groups[level].append(idx)
            cursors = {level: 0 for level in level_order}
            selected: list[int] = []
            while len(selected) < max_generate_samples:
                progressed = False
                for level in level_order:
                    cursor = cursors[level]
                    bucket = groups[level]
                    if cursor < len(bucket):
                        selected.append(bucket[cursor])
                        cursors[level] = cursor + 1
                        progressed = True
                        if len(selected) >= max_generate_samples:
                            break
                if not progressed:
                    break
            generate_indices = set(selected)
        else:
            generate_indices = set(range(min(max_generate_samples, len(rows))))
    with torch.no_grad():
        for row_idx, row in enumerate(rows):
            example = build_single_native_answer_example(
                row=row,
                model_path=model_path,
                latent_cache_root=latent_cache_root,
                latent_target_dim=latent_target_dim,
            )
            loss, ce, mse, visual = compute_losses(model, visual_head, example, device, ce_weight, mse_weight, visual_summary_weight)
            level = str(row.get("reasoning_level") or "unknown")
            total_loss += float(loss.item())
            total_ce += float(ce.item())
            total_mse += float(mse.item())
            total_visual += float(visual.item())
            rendered_target = "".join(build_native_answer_pieces(row, None, latent_target_dim)[0][1:])
            if format_matches_expected(rendered_target, expected_tag_sequence_for_row(row)):
                format_hits += 1
                by_level[level]["target_format"] += 1
            count += 1
            by_level[level]["count"] += 1
            by_level[level]["loss"] += float(loss.item())
            by_level[level]["ce"] += float(ce.item())
            by_level[level]["mse"] += float(mse.item())
            by_level[level]["visual"] += float(visual.item())
            if max_generate_samples > 0 and generated_count < max_generate_samples and row_idx in generate_indices:
                prompt_text = build_context_text(row)
                expected_sequence = expected_tag_sequence_for_row(row)
                generated_text = generate_reasoning_text(
                    model=model,
                    tokenizer=tokenizer,
                    new_token_ids=new_token_ids,
                    prompt_text=prompt_text,
                    max_new_tokens=max_generate_new_tokens,
                )
                generated_sequence = extract_structural_tag_sequence(generated_text)
                generated_format_ok = generated_sequence == expected_sequence
                generated_count += 1
                by_level[level]["generated_count"] += 1
                if generated_format_ok:
                    generated_format_hits += 1
                    by_level[level]["generated_format"] += 1
                cases.append(
                    {
                        "row_id": str(row.get("id") or row.get("row_id") or ""),
                        "reasoning_level": level,
                        "expected_tag_sequence": expected_sequence,
                        "generated_tag_sequence": generated_sequence,
                        "format_ok": generated_format_ok,
                        "prompt_used": prompt_text,
                        "generated_text": generated_text,
                    }
                )
                if max_generate_image_samples > 0 and generated_image_count < max_generate_image_samples:
                    if image_output_dir is None:
                        raise ValueError("image_output_dir is required when max_generate_image_samples > 0")
                    if vae_model is None:
                        vae_model = load_vae_for_generation(model_path, device)
                    generated_think_text, generated_image = generate_reasoning_image(
                        model=model,
                        vae_model=vae_model,
                        tokenizer=tokenizer,
                        new_token_ids=new_token_ids,
                        prompt_text=prompt_text,
                        image_size=generate_image_size,
                        num_timesteps=generate_image_steps,
                        max_new_tokens=max_generate_new_tokens,
                    )
                    image_output_dir.mkdir(parents=True, exist_ok=True)
                    image_name = f"{str(row.get('id') or row.get('row_id') or generated_image_count)}.png"
                    image_path = image_output_dir / image_name
                    generated_image.save(image_path)
                    cases[-1]["single_context_generated_think_text"] = generated_think_text
                    cases[-1]["single_context_generated_image_path"] = str(image_path)
                    generated_image_count += 1
    if not was_training:
        model.eval()
    if case_output_path is not None:
        case_output_path.parent.mkdir(parents=True, exist_ok=True)
        with case_output_path.open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    denom = max(count, 1)
    generated_denom = max(generated_count, 1)
    metrics: dict[str, Any] = {
        "val_samples": count,
        "val_generate_samples": generated_count,
        "val_generate_image_samples": generated_image_count,
        "val_loss": total_loss / denom,
        "val_ce": total_ce / denom,
        "val_mse": total_mse / denom,
        "val_visual": total_visual / denom,
        "val_target_format_accuracy": format_hits / denom,
        "val_format_accuracy": generated_format_hits / generated_denom if generated_count else 0.0,
        "per_level": {},
    }
    for level, values in by_level.items():
        level_denom = max(int(values["count"]), 1)
        level_generated_denom = max(int(values["generated_count"]), 1)
        metrics["per_level"][level] = {
            "samples": int(values["count"]),
            "generated_samples": int(values["generated_count"]),
            "val_loss": values["loss"] / level_denom,
            "val_ce": values["ce"] / level_denom,
            "val_mse": values["mse"] / level_denom,
            "val_visual": values["visual"] / level_denom,
            "val_target_format_accuracy": values["target_format"] / level_denom,
            "val_format_accuracy": values["generated_format"] / level_generated_denom if values["generated_count"] else 0.0,
        }
    return metrics


def save_trainable_subset(model: torch.nn.Module, visual_head: torch.nn.Module, output_dir: Path, metrics: dict[str, Any], tag: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            subset[name] = param.detach().cpu()
    torch.save(subset, output_dir / f"{tag}_trainable_state.pt")
    torch.save(visual_head.state_dict(), output_dir / f"{tag}_visual_head.pt")
    (output_dir / f"{tag}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path).expanduser()
    latent_cache_root = Path(args.latent_cache_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    train_rows = build_rows(
        args.train_jsonl,
        seed=args.seed,
        max_samples=args.train_max_samples,
        sample_selection=args.train_sample_selection,
    )
    val_rows = build_rows(
        args.val_jsonl,
        seed=args.seed + 1,
        max_samples=args.val_max_samples,
        sample_selection=args.val_sample_selection,
    )
    if not train_rows:
        raise SystemExit("No train rows selected.")
    if not val_rows:
        raise SystemExit("No val rows selected.")

    model, tokenizer = load_full_bagel_dispatch(
        model_path,
        max_mem_per_gpu=args.max_mem_per_gpu,
        offload_folder=args.offload_folder,
        visual_und=False,
    )
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    model.train()
    if args.finetune_mode == "lora":
        model, trainable, total = install_lora_on_language_model(
            model,
            target_keywords=args.target_modules,
            rank=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            resume_adapter_path=args.resume_adapter_path,
        )
        stats = {"trainable": trainable, "total": total}
    elif args.finetune_mode == "partial_sft":
        model, trainable, total = install_partial_sft_on_language_model(
            model,
            train_last_n_layers=args.train_last_n_layers,
            train_lm_head=args.train_lm_head,
            train_final_norm=args.train_final_norm,
        )
        model = maybe_load_trainable_model_state(model, args.resume_adapter_path, allow_missing=False)
        stats = {"trainable": trainable, "total": total}
    else:
        stats = select_native_visual_trainables(model, include_moe_gen=False)
    device = next(model.parameters()).device
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    visual_head = torch.nn.Linear(model.hidden_size, args.latent_target_dim, bias=True).to(device=device, dtype=next(model.parameters()).dtype)
    visual_head = maybe_load_visual_head(visual_head, args.resume_visual_head_path, allow_missing=False)
    trainable_params.extend(list(visual_head.parameters()))
    stats["trainable"] += sum(param.numel() for param in visual_head.parameters())
    stats["total"] += sum(param.numel() for param in visual_head.parameters())
    if not trainable_params:
        raise SystemExit("No trainable parameters selected.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    print(
        f"[unipath_native] train_rows={len(train_rows)} val_rows={len(val_rows)} "
        f"trainable={stats['trainable']} trainable_pct={100.0 * stats['trainable'] / max(stats['total'], 1):.4f}",
        flush=True,
    )
    fixed_generate_row_ids = select_fixed_generate_row_ids(
        val_rows,
        max_generate_samples=args.eval_generate_max_samples,
        generate_sampling=args.eval_generate_sampling,
    )
    if args.eval_generate_max_samples > 0:
        print(
            f"[unipath_native] fixed_eval_generate_set size={len(fixed_generate_row_ids)} "
            f"sampling={args.eval_generate_sampling}",
            flush=True,
        )

    best_metric = -math.inf if native_best_metric_is_higher(args.save_best_metric) else math.inf
    format_floor: float | None = None
    no_improve_evals = 0
    stop_training = False

    if args.initial_eval:
        metrics = evaluate(
            model=model,
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
            rows=val_rows,
            model_path=model_path,
            latent_cache_root=latent_cache_root,
            device=device,
            visual_head=visual_head,
            ce_weight=args.ce_weight,
            mse_weight=args.mse_weight,
            visual_summary_weight=args.visual_summary_weight,
            latent_target_dim=args.latent_target_dim,
            max_generate_samples=args.eval_generate_max_samples,
            generate_sampling=args.eval_generate_sampling,
            max_generate_new_tokens=args.eval_generate_max_new_tokens,
            max_generate_image_samples=args.eval_generate_image_samples,
            generate_image_steps=args.eval_generate_image_steps,
            generate_image_size=args.eval_generate_image_size,
            case_output_path=output_dir / "eval_cases" / "step_0.jsonl" if args.save_eval_cases else None,
            image_output_dir=output_dir / "eval_images" / "step_0" if args.save_eval_cases else None,
            fixed_generate_row_ids=fixed_generate_row_ids,
        )
        metric_value = native_best_metric_value(metrics, args.save_best_metric)
        best_metric = metric_value
        if args.best_require_format_no_degrade:
            format_floor = float(metrics.get("val_format_accuracy", 0.0))
        print(
            f"[unipath_native] eval step=0/{args.max_steps} "
            f"val_loss={metrics['val_loss']:.6f} val_ce={metrics['val_ce']:.6f} val_mse={metrics['val_mse']:.6f} "
            f"val_visual={metrics['val_visual']:.6f} val_format={metrics['val_format_accuracy']:.6f}",
            flush=True,
        )
        save_checkpoint_bundle(model, visual_head, output_dir / "adapter_imitation_best", global_step=0, stage="imitation", kind="best", finetune_mode=args.finetune_mode, metrics=metrics)
        if args.stop_on_zero_format and metrics.get("val_format_accuracy", 0.0) == 0.0:
            print("[unipath_native] zero_format_stop_triggered step=0", flush=True)
            stop_training = True

    for step in range(1, args.max_steps + 1):
        if stop_training:
            break
        row = train_rows[(step - 1) % len(train_rows)]
        example = build_single_native_answer_example(
            row=row,
            model_path=model_path,
            latent_cache_root=latent_cache_root,
            latent_target_dim=args.latent_target_dim,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, ce, mse, visual = compute_losses(model, visual_head, example, device, args.ce_weight, args.mse_weight, args.visual_summary_weight)
        loss.backward()
        current_grad_norm = grad_norm(trainable_params)
        optimizer.step()
        print(
            f"[unipath_native] step={step}/{args.max_steps} "
            f"row={example.row_id} loss={loss.item():.6f} ce={ce.item():.6f} mse={mse.item():.6f} visual={visual.item():.6f} "
            f"grad_norm={current_grad_norm:.6f}",
            flush=True,
        )

        if step % args.eval_every_steps == 0 or step == args.max_steps:
            metrics = evaluate(
                model=model,
                tokenizer=tokenizer,
                new_token_ids=new_token_ids,
                rows=val_rows,
                model_path=model_path,
                latent_cache_root=latent_cache_root,
                device=device,
                visual_head=visual_head,
                ce_weight=args.ce_weight,
                mse_weight=args.mse_weight,
                visual_summary_weight=args.visual_summary_weight,
                latent_target_dim=args.latent_target_dim,
                max_generate_samples=args.eval_generate_max_samples,
                generate_sampling=args.eval_generate_sampling,
                max_generate_new_tokens=args.eval_generate_max_new_tokens,
                max_generate_image_samples=args.eval_generate_image_samples,
                generate_image_steps=args.eval_generate_image_steps,
                generate_image_size=args.eval_generate_image_size,
                case_output_path=output_dir / "eval_cases" / f"step_{step}.jsonl" if args.save_eval_cases else None,
                image_output_dir=output_dir / "eval_images" / f"step_{step}" if args.save_eval_cases else None,
                fixed_generate_row_ids=fixed_generate_row_ids,
            )
            print(
                f"[unipath_native] eval step={step}/{args.max_steps} "
                f"val_loss={metrics['val_loss']:.6f} val_ce={metrics['val_ce']:.6f} val_mse={metrics['val_mse']:.6f} "
                f"val_visual={metrics['val_visual']:.6f} val_format={metrics['val_format_accuracy']:.6f}",
                flush=True,
            )
            for level, values in sorted(metrics["per_level"].items()):
                print(
                    f"[unipath_native] eval level={level} samples={values['samples']} "
                    f"val_loss={values['val_loss']:.6f} val_ce={values['val_ce']:.6f} val_mse={values['val_mse']:.6f} "
                    f"val_visual={values['val_visual']:.6f} val_format={values['val_format_accuracy']:.6f}",
                    flush=True,
                )
            metric_value = native_best_metric_value(metrics, args.save_best_metric)
            format_gate_ok = True
            if args.best_require_format_no_degrade:
                if format_floor is None:
                    format_floor = float(metrics.get("val_format_accuracy", 0.0))
                format_gate_ok = (
                    float(metrics.get("val_format_accuracy", 0.0))
                    >= float(format_floor) - max(args.best_format_tolerance, 0.0)
                )
            improved = format_gate_ok and native_metric_improved(metric_value, best_metric, args.save_best_metric, args.min_delta)
            if improved:
                best_metric = metric_value
                no_improve_evals = 0
                save_checkpoint_bundle(model, visual_head, output_dir / "adapter_imitation_best", global_step=step, stage="imitation", kind="best", finetune_mode=args.finetune_mode, metrics=metrics)
                print(
                    f"[unipath_native] saved_best step={step} metric={best_metric:.6f} "
                    f"path={output_dir / 'adapter_imitation_best'}",
                    flush=True,
                )
            else:
                no_improve_evals += 1
                if args.best_require_format_no_degrade and not format_gate_ok:
                    print(
                        f"[unipath_native] best_gate_blocked step={step} "
                        f"val_format={float(metrics.get('val_format_accuracy', 0.0)):.6f} "
                        f"format_floor={float(format_floor or 0.0):.6f} tol={max(args.best_format_tolerance, 0.0):.6f}",
                        flush=True,
                    )
            if args.stop_on_zero_format and metrics.get("val_format_accuracy", 0.0) == 0.0:
                print(f"[unipath_native] zero_format_stop_triggered step={step}", flush=True)
                stop_training = True
            if (
                args.early_stop_patience > 0
                and step >= args.min_steps_before_early_stop
                and no_improve_evals >= args.early_stop_patience
            ):
                print(f"[unipath_native] early_stop_triggered step={step} no_improve_evals={no_improve_evals}", flush=True)
                stop_training = True

    final_metrics = evaluate(
        model=model,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        rows=val_rows,
        model_path=model_path,
        latent_cache_root=latent_cache_root,
        device=device,
        visual_head=visual_head,
        ce_weight=args.ce_weight,
        mse_weight=args.mse_weight,
        visual_summary_weight=args.visual_summary_weight,
        latent_target_dim=args.latent_target_dim,
        max_generate_samples=args.eval_generate_max_samples,
        generate_sampling=args.eval_generate_sampling,
        max_generate_new_tokens=args.eval_generate_max_new_tokens,
        max_generate_image_samples=args.eval_generate_image_samples,
        generate_image_steps=args.eval_generate_image_steps,
        generate_image_size=args.eval_generate_image_size,
        case_output_path=output_dir / "eval_cases" / "last.jsonl" if args.save_eval_cases else None,
        image_output_dir=output_dir / "eval_images" / "last" if args.save_eval_cases else None,
        fixed_generate_row_ids=fixed_generate_row_ids,
    )
    if args.save_last:
        save_checkpoint_bundle(model, visual_head, output_dir / "adapter_imitation_last", global_step=step if 'step' in locals() else 0, stage="imitation", kind="last", finetune_mode=args.finetune_mode, metrics=final_metrics)
        print(f"[unipath_native] saved_last path={output_dir / 'adapter_imitation_last'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
