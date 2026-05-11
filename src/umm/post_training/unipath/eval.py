from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from umm.post_training.unipath.train import (
    UniPathTokenDataset,
    WeightedCollator,
    build_context_text,
    build_language_model,
    compute_visual_summary_loss,
    maybe_load_lora_adapter,
    maybe_load_visual_head,
    resolve_jsonl_paths,
    weighted_language_model_loss,
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UniPath LoRA checkpoints on held-out validation data.")
    parser.add_argument("--val-jsonl", action="append", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--visual-head-path", required=True)
    parser.add_argument("--latent-cache-root", default=None)
    parser.add_argument("--latent-target-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-generate-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    cleaned = re.sub(r"[^\w\s./-]", "", cleaned)
    return cleaned.strip()


def is_image_answer(row: dict[str, Any]) -> bool:
    answer_type = str(row.get("answer_type") or "").lower()
    if answer_type == "image":
        return True
    answer = str(row.get("answer") or "").lower()
    return answer.endswith(IMAGE_EXTENSIONS)


def extract_answer_text(text: str) -> str:
    match = re.search(r"Answer:\s*(.*)", text, flags=re.DOTALL)
    if not match:
        return ""
    answer = match.group(1).strip()
    answer = answer.split("Task Understanding:")[0].strip()
    return answer


def format_is_valid(text: str) -> bool:
    return "Answer:" in text and any(tag in text for tag in ("Task Understanding:", "Text Thought:", "Visual Thought:", "Visual Hypothesis:"))


def evaluate_teacher_forced(
    model: torch.nn.Module,
    visual_head: torch.nn.Module,
    dataset: UniPathTokenDataset,
    tokenizer: Any,
    batch_size: int,
    latent_target_dim: int,
    device: torch.device,
) -> dict[str, float]:
    collator = WeightedCollator(tokenizer, latent_target_dim=latent_target_dim)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    totals = {
        "lm_loss_sum": 0.0,
        "visual_loss_sum": 0.0,
        "visual_thought_loss_sum": 0.0,
        "visual_answer_loss_sum": 0.0,
        "batches": 0.0,
    }

    model.eval()
    visual_head.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                use_cache=False,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]
            lm_loss = weighted_language_model_loss(
                logits=outputs.logits,
                labels=batch["labels"].to(device),
                loss_weights=batch["loss_weights"].to(device),
            )
            visual_loss = compute_visual_summary_loss(
                hidden_states=hidden,
                visual_spans=batch["visual_spans"].to(device),
                visual_targets=batch["visual_targets"].to(device),
                visual_mask=batch["visual_mask"].to(device),
                visual_head=visual_head,
                visual_types=batch["visual_types"].to(device),
            )
            visual_thought_loss = compute_visual_summary_loss(
                hidden_states=hidden,
                visual_spans=batch["visual_spans"].to(device),
                visual_targets=batch["visual_targets"].to(device),
                visual_mask=batch["visual_mask"].to(device),
                visual_head=visual_head,
                visual_types=batch["visual_types"].to(device),
                allowed_types=(0,),
            )
            visual_answer_loss = compute_visual_summary_loss(
                hidden_states=hidden,
                visual_spans=batch["visual_spans"].to(device),
                visual_targets=batch["visual_targets"].to(device),
                visual_mask=batch["visual_mask"].to(device),
                visual_head=visual_head,
                visual_types=batch["visual_types"].to(device),
                allowed_types=(1,),
            )
            totals["lm_loss_sum"] += float(lm_loss.item())
            totals["visual_loss_sum"] += float(visual_loss.item())
            totals["visual_thought_loss_sum"] += float(visual_thought_loss.item())
            totals["visual_answer_loss_sum"] += float(visual_answer_loss.item())
            totals["batches"] += 1.0

    denom = max(totals["batches"], 1.0)
    return {
        "lm_loss": totals["lm_loss_sum"] / denom,
        "visual_loss": totals["visual_loss_sum"] / denom,
        "visual_thought_mse": totals["visual_thought_loss_sum"] / denom,
        "visual_answer_mse": totals["visual_answer_loss_sum"] / denom,
    }


def evaluate_generation(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
    max_generate_samples: int,
) -> dict[str, Any]:
    candidates = [row for row in rows if not is_image_answer(row)]
    candidates = candidates[:max_generate_samples]

    stats = {
        "generated_text_samples": len(candidates),
        "format_pass_count": 0,
        "text_exact_match_count": 0,
    }
    by_level: dict[str, dict[str, int]] = defaultdict(lambda: {"samples": 0, "format_pass": 0, "exact_match": 0})

    model.eval()
    with torch.no_grad():
        for row in candidates:
            prompt_text = build_context_text(row)
            prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
            generated = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_text = tokenizer.decode(generated[0][prompt_ids.shape[1]:], skip_special_tokens=True)
            level = str(row.get("reasoning_level") or "unknown")
            by_level[level]["samples"] += 1
            if format_is_valid(generated_text):
                stats["format_pass_count"] += 1
                by_level[level]["format_pass"] += 1
            expected = normalize_text(str(row.get("answer") or ""))
            actual = normalize_text(extract_answer_text(generated_text))
            if expected and actual and expected == actual:
                stats["text_exact_match_count"] += 1
                by_level[level]["exact_match"] += 1

    denom = max(stats["generated_text_samples"], 1)
    summary = {
        "generated_text_samples": stats["generated_text_samples"],
        "format_pass_rate": stats["format_pass_count"] / denom,
        "text_exact_match_rate": stats["text_exact_match_count"] / denom,
        "per_level": {},
    }
    for level, values in by_level.items():
        level_denom = max(values["samples"], 1)
        summary["per_level"][level] = {
            "samples": values["samples"],
            "format_pass_rate": values["format_pass"] / level_denom,
            "text_exact_match_rate": values["exact_match"] / level_denom,
        }
    return summary


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path).expanduser()
    adapter_path = Path(args.adapter_path).expanduser()
    visual_head_path = Path(args.visual_head_path).expanduser()
    latent_cache_root = Path(args.latent_cache_root).expanduser() if args.latent_cache_root else None
    val_paths = resolve_jsonl_paths(args.val_jsonl, skip_missing=False)

    model, tokenizer = build_language_model(model_path, meta_only=False)
    model = maybe_load_lora_adapter(model, str(adapter_path), allow_missing=False)
    hidden_size = int(model.config.hidden_size)
    visual_head = torch.nn.Linear(hidden_size, args.latent_target_dim)
    visual_head = maybe_load_visual_head(visual_head, str(visual_head_path), allow_missing=False)

    dataset = UniPathTokenDataset(
        val_paths,
        tokenizer=tokenizer,
        seed=args.seed,
        max_length=args.max_length,
        thought_weight=1.0,
        answer_weight=1.0,
        latent_cache_root=latent_cache_root,
        latent_target_dim=args.latent_target_dim,
        max_samples=args.max_samples,
    )
    if len(dataset) == 0:
        raise ValueError("No validation samples found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    visual_head = visual_head.to(device)

    teacher_forced = evaluate_teacher_forced(
        model=model,
        visual_head=visual_head,
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        latent_target_dim=args.latent_target_dim,
        device=device,
    )

    rows: list[dict[str, Any]] = []
    for path in val_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    generation = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        device=device,
        max_new_tokens=args.max_new_tokens,
        max_generate_samples=args.max_generate_samples,
    )

    result = {
        "val_jsonl": [str(path) for path in val_paths],
        "num_val_samples": len(dataset),
        "teacher_forced": teacher_forced,
        "generation": generation,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
