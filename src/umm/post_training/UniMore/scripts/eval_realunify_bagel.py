#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SPLIT_NAMES = ("UEG_direct", "GEU_direct", "UEG_step", "GEU_step")
EXPECTED_COUNTS = {
    "UEG_direct": 600,
    "GEU_direct": 400,
    "UEG_step": 600,
    "GEU_step": 400,
}
REQUIRED_TASKS = {
    "UEG_direct": {
        "world_knowledge",
        "commonsense_reasoning",
        "mathematical_reasoning",
        "logical_reasoning",
        "scientific_reasoning",
        "code_to_image",
    },
    "GEU_direct": {
        "mental_reconstruction",
        "mental_tracking",
        "attentional_focusing",
        "cognitive_navigation",
    },
    "UEG_step": {
        "world_knowledge",
        "commonsense_reasoning",
        "mathematical_reasoning",
        "logical_reasoning",
        "scientific_reasoning",
        "code_to_image",
    },
    "GEU_step": {
        "mental_reconstruction",
        "mental_tracking",
        "attentional_focusing",
        "cognitive_navigation",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BAGEL checkpoints on the official RealUnify protocol."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate the downloaded dataset.")
    validate.add_argument("--data-root", type=Path, required=True)

    run = subparsers.add_parser("run", help="Run one checkpoint on one or more splits.")
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--checkpoint-name", required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--bagel-root", type=Path, required=True)
    run.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, required=True)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--max-samples", type=int, default=0)
    run.add_argument("--samples-per-task", type=int, default=0)
    run.add_argument("--base-seed", type=int, default=42)
    run.add_argument("--image-size", type=int, choices=(256, 512, 1024), default=1024)
    run.add_argument("--max-mem-per-gpu", default="78GiB")
    run.add_argument("--offload-folder", type=Path, required=True)
    run.add_argument("--data-revision", default="0b3609a3f5dbc460a341e78d833fe6272693c855")
    run.add_argument("--code-revision", default="")
    run.add_argument("--log-every", type=int, default=1)
    run.add_argument("--no-resume", action="store_true")

    aggregate = subparsers.add_parser(
        "aggregate", help="Merge completed shard JSONL files and calculate GEU accuracy."
    )
    aggregate.add_argument("--checkpoint-name", required=True)
    aggregate.add_argument("--data-root", type=Path, required=True)
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, required=True)
    aggregate.add_argument("--max-samples", type=int, default=0)
    aggregate.add_argument("--samples-per-task", type=int, default=0)
    aggregate.add_argument("--require-complete", action="store_true")

    return parser.parse_args()


def load_split(data_root: Path, split: str) -> list[dict[str, Any]]:
    path = data_root / f"{split}.json"
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"{path} must contain a JSON list")
    return payload


def item_key(split: str, item: dict[str, Any]) -> str:
    return f"{split}:{item['task_type']}:{int(item['index']):04d}"


def select_rows(
    rows: Iterable[dict[str, Any]],
    *,
    samples_per_task: int,
    max_samples: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    for row in rows:
        task_type = str(row["task_type"])
        if samples_per_task > 0 and task_counts[task_type] >= samples_per_task:
            continue
        selected.append(row)
        task_counts[task_type] += 1
        if max_samples > 0 and len(selected) >= max_samples:
            break
    return selected


def resolve_input_image(data_root: Path, raw_path: str) -> Path:
    normalized = raw_path.replace("\\", "/")
    if "/image/" in normalized:
        relative = normalized.split("/image/", 1)[1]
    elif normalized.startswith("image/"):
        relative = normalized[len("image/") :]
    else:
        relative = normalized
    resolved = (data_root / "image" / relative).resolve()
    image_root = (data_root / "image").resolve()
    if not resolved.is_relative_to(image_root):
        raise ValueError(f"Input image escapes the RealUnify image root: {raw_path}")
    if not resolved.is_file():
        raise FileNotFoundError(f"RealUnify input image not found: {resolved}")
    return resolved


def validate_dataset(data_root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "data_root": str(data_root.resolve()),
        "splits": {},
        "valid": True,
    }
    for split in SPLIT_NAMES:
        rows = load_split(data_root, split)
        keys = [item_key(split, row) for row in rows]
        task_counts = Counter(str(row["task_type"]) for row in rows)
        missing_images: list[str] = []
        if split.startswith("GEU"):
            for row in rows:
                try:
                    resolve_input_image(data_root, str(row["input_image"]))
                except FileNotFoundError as exc:
                    missing_images.append(str(exc))

        split_valid = (
            len(rows) == EXPECTED_COUNTS[split]
            and len(keys) == len(set(keys))
            and set(task_counts) == REQUIRED_TASKS[split]
            and all(count == 100 for count in task_counts.values())
            and not missing_images
        )
        summary["splits"][split] = {
            "count": len(rows),
            "expected_count": EXPECTED_COUNTS[split],
            "unique_keys": len(set(keys)),
            "task_counts": dict(sorted(task_counts.items())),
            "missing_image_count": len(missing_images),
            "missing_images": missing_images[:10],
            "valid": split_valid,
        }
        summary["valid"] = bool(summary["valid"] and split_valid)
    return summary


def stable_seed(base_seed: int, item: dict[str, Any], stage: str) -> int:
    value = f"{base_seed}:{item['task_type']}:{int(item['index'])}:{stage}".encode()
    digest = hashlib.sha256(value).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def extract_text(output: Any) -> str:
    if isinstance(output, dict):
        value = output.get("text")
    else:
        value = output
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"BAGEL returned an empty text response: {output!r}")
    return value.strip()


def image_output_path(output_root: Path, split: str, item: dict[str, Any]) -> Path:
    return output_root / split / "images" / str(item["task_type"]) / f"{int(item['index']):04d}.png"


def validate_generated_image(path: Path, image_size: int) -> None:
    from PIL import Image

    with Image.open(path) as image:
        actual_size = image.size
    expected_size = (image_size, image_size)
    if actual_size != expected_size:
        raise RuntimeError(
            f"Generated image has size {actual_size}, expected {expected_size}: {path}"
        )


def run_request(
    pipeline: Any,
    split: str,
    item: dict[str, Any],
    *,
    data_root: Path,
    output_root: Path,
    base_seed: int,
    image_size: int,
) -> dict[str, Any]:
    result = dict(item)
    input_image = None
    if split.startswith("GEU"):
        input_image = resolve_input_image(data_root, str(item["input_image"]))
        result["input_image"] = str(input_image)

    if split == "UEG_direct":
        output_path = image_output_path(output_root, split, item)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generation_seed = stable_seed(base_seed, item, "generation")
        set_seed(generation_seed)
        pipeline.run(
            {
                "backbone": "bagel",
                "task": "generation",
                "prompt": item["prompt"],
                "output_path": str(output_path),
                "params": {
                    "cfg_text_scale": 4.0,
                    "cfg_img_scale": 1.0,
                    "cfg_interval": [0.4, 1.0],
                    "timestep_shift": 3.0,
                    "num_timesteps": 50,
                    "cfg_renorm_min": 0.0,
                    "cfg_renorm_type": "global",
                    "image_shapes": [image_size, image_size],
                    "think": False,
                },
            }
        )
        if not output_path.is_file():
            raise RuntimeError(f"BAGEL did not write the generated image: {output_path}")
        validate_generated_image(output_path, image_size)
        result["generated_image"] = str(output_path)
        result["_seed"] = generation_seed

    elif split == "GEU_direct":
        output = pipeline.run(
            {
                "backbone": "bagel",
                "task": "understanding",
                "prompt": item["evaluation_prompt"],
                "images": [str(input_image)],
                "params": {"max_think_token_n": 1000, "do_sample": False},
            }
        )
        result["response"] = extract_text(output)

    elif split == "UEG_step":
        refined = pipeline.run(
            {
                "backbone": "bagel",
                "task": "understanding",
                "prompt": item["new_prompt"],
                "params": {"max_think_token_n": 1000, "do_sample": False},
            }
        )
        result["response"] = extract_text(refined)
        output_path = image_output_path(output_root, split, item)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generation_seed = stable_seed(base_seed, item, "generation")
        set_seed(generation_seed)
        pipeline.run(
            {
                "backbone": "bagel",
                "task": "generation",
                "prompt": result["response"],
                "output_path": str(output_path),
                "params": {
                    "cfg_text_scale": 4.0,
                    "cfg_img_scale": 1.0,
                    "cfg_interval": [0.4, 1.0],
                    "timestep_shift": 3.0,
                    "num_timesteps": 50,
                    "cfg_renorm_min": 0.0,
                    "cfg_renorm_type": "global",
                    "image_shapes": [image_size, image_size],
                    "think": False,
                },
            }
        )
        if not output_path.is_file():
            raise RuntimeError(f"BAGEL did not write the generated image: {output_path}")
        validate_generated_image(output_path, image_size)
        result["generated_image"] = str(output_path)
        result["_seed"] = generation_seed

    elif split == "GEU_step":
        output_path = image_output_path(output_root, split, item)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        editing_seed = stable_seed(base_seed, item, "editing")
        set_seed(editing_seed)
        pipeline.run(
            {
                "backbone": "bagel",
                "task": "editing",
                "prompt": item["edit_prompt"],
                "images": [str(input_image)],
                "output_path": str(output_path),
                "params": {
                    "cfg_text_scale": 4.0,
                    "cfg_img_scale": 2.0,
                    "cfg_interval": [0.0, 1.0],
                    "timestep_shift": 3.0,
                    "num_timesteps": 50,
                    "cfg_renorm_min": 0.0,
                    "cfg_renorm_type": "text_channel",
                    "think": False,
                },
            }
        )
        if not output_path.is_file():
            raise RuntimeError(f"BAGEL did not write the edited image: {output_path}")
        result["edit_image"] = str(output_path)
        output = pipeline.run(
            {
                "backbone": "bagel",
                "task": "understanding",
                "prompt": item["evaluation_prompt"],
                "images": [str(output_path)],
                "params": {"max_think_token_n": 1000, "do_sample": False},
            }
        )
        result["response"] = extract_text(output)
        result["_seed"] = editing_seed

    else:
        raise ValueError(f"Unsupported split: {split}")

    return result


def read_jsonl(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    records: dict[str, dict[str, Any]] = {}
    total = 0
    if not path.is_file():
        return records, total
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            key = record.get("_realunify_key")
            if not isinstance(key, str):
                raise ValueError(f"Missing _realunify_key at {path}:{line_number}")
            records[key] = record
            total += 1
    return records, total


def record_is_complete(split: str, record: dict[str, Any]) -> bool:
    if split.startswith("UEG"):
        path = record.get("generated_image")
        return isinstance(path, str) and Path(path).is_file()
    if split == "GEU_step":
        path = record.get("edit_image")
        return (
            isinstance(path, str)
            and Path(path).is_file()
            and isinstance(record.get("response"), str)
            and bool(record["response"].strip())
        )
    return isinstance(record.get("response"), str) and bool(record["response"].strip())


def write_json_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def build_pipeline(args: argparse.Namespace) -> Any:
    from umm.inference import InferencePipeline

    return InferencePipeline(
        backbone_name="bagel",
        backbone_cfg={
            "model_path": str(args.model_path.resolve()),
            "bagel_root": str(args.bagel_root.resolve()),
            "max_mem_per_gpu": args.max_mem_per_gpu,
            "offload_folder": str(args.offload_folder.resolve()),
            "seed": args.base_seed,
        },
    )


def run_split(args: argparse.Namespace, pipeline: Any, split: str) -> dict[str, Any]:
    rows = select_rows(
        load_split(args.data_root, split),
        samples_per_task=args.samples_per_task,
        max_samples=args.max_samples,
    )
    shard_rows = rows[args.shard_index :: args.num_shards]
    split_root = args.output_root / args.checkpoint_name / split
    shard_path = (
        split_root / "shards" / f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}.jsonl"
    )
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    completed, existing_lines = read_jsonl(shard_path) if not args.no_resume else ({}, 0)

    manifest = {
        "checkpoint_name": args.checkpoint_name,
        "checkpoint_path": str(args.model_path.resolve()),
        "checkpoint_size_bytes": (args.model_path / "ema.safetensors").stat().st_size,
        "data_root": str(args.data_root.resolve()),
        "data_revision": args.data_revision,
        "code_revision": args.code_revision,
        "split": split,
        "selected_count": len(rows),
        "shard_count": len(shard_rows),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "base_seed": args.base_seed,
        "generation": {
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.0,
            "cfg_interval": [0.4, 1.0],
            "timestep_shift": 3.0,
            "num_timesteps": 50,
            "image_shapes": [args.image_size, args.image_size],
        },
        "editing": {
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 2.0,
            "cfg_interval": [0.0, 1.0],
            "timestep_shift": 3.0,
            "num_timesteps": 50,
        },
        "understanding": {"max_think_token_n": 1000, "do_sample": False},
    }
    write_json_atomic(
        manifest,
        split_root / "manifests" / f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}.json",
    )

    started = time.monotonic()
    processed = 0
    skipped = 0
    with shard_path.open("a", encoding="utf-8") as output:
        for position, item in enumerate(shard_rows, start=1):
            key = item_key(split, item)
            old_record = completed.get(key)
            if old_record is not None and record_is_complete(split, old_record):
                skipped += 1
                continue

            item_started = time.monotonic()
            record = run_request(
                pipeline,
                split,
                item,
                data_root=args.data_root,
                output_root=args.output_root / args.checkpoint_name,
                base_seed=args.base_seed,
                image_size=args.image_size,
            )
            elapsed = time.monotonic() - item_started
            record["_realunify_key"] = key
            record["_checkpoint_name"] = args.checkpoint_name
            record["_split"] = split
            record["_shard_index"] = args.shard_index
            record["_num_shards"] = args.num_shards
            record["_image_size"] = args.image_size
            record["_elapsed_seconds"] = round(elapsed, 6)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            completed[key] = record
            processed += 1

            if args.log_every > 0 and (
                processed % args.log_every == 0 or position == len(shard_rows)
            ):
                total_elapsed = time.monotonic() - started
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "split": split,
                            "checkpoint": args.checkpoint_name,
                            "position": position,
                            "shard_count": len(shard_rows),
                            "processed": processed,
                            "resumed": skipped,
                            "last_seconds": round(elapsed, 3),
                            "total_seconds": round(total_elapsed, 3),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    return {
        "split": split,
        "shard_path": str(shard_path),
        "shard_count": len(shard_rows),
        "processed": processed,
        "resumed": skipped,
        "existing_jsonl_lines": existing_lines,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def extract_choice(response: str) -> str:
    cleaned = response.strip()
    prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer",
        "Answer is",
    ]
    for prefix in prefixes:
        cleaned = cleaned.replace(prefix, "")
    if len(cleaned.split()) > 10 and re.search(r"[ABCD]", cleaned) is None:
        return ""
    match = re.search(r"[ABCD]", cleaned)
    return match.group(0) if match else ""


def aggregate_split(args: argparse.Namespace, split: str) -> dict[str, Any]:
    expected_rows = select_rows(
        load_split(args.data_root, split),
        samples_per_task=args.samples_per_task,
        max_samples=args.max_samples,
    )
    expected_by_key = {item_key(split, row): row for row in expected_rows}
    split_root = args.output_root / args.checkpoint_name / split
    shard_paths = sorted((split_root / "shards").glob("shard-*.jsonl"))
    records: dict[str, dict[str, Any]] = {}
    total_lines = 0
    for shard_path in shard_paths:
        shard_records, shard_lines = read_jsonl(shard_path)
        records.update(shard_records)
        total_lines += shard_lines

    unexpected = sorted(set(records) - set(expected_by_key))
    missing = sorted(set(expected_by_key) - set(records))
    incomplete = sorted(
        key for key, record in records.items() if not record_is_complete(split, record)
    )
    ordered = [
        records[key]
        for key in expected_by_key
        if key in records and record_is_complete(split, records[key])
    ]

    write_json_atomic(ordered, split_root / "results.json")
    summary: dict[str, Any] = {
        "checkpoint_name": args.checkpoint_name,
        "split": split,
        "expected_count": len(expected_rows),
        "result_count": len(ordered),
        "shard_file_count": len(shard_paths),
        "jsonl_line_count": total_lines,
        "duplicate_line_count": total_lines - len(records),
        "missing_count": len(missing),
        "missing_keys": missing,
        "incomplete_count": len(incomplete),
        "incomplete_keys": incomplete,
        "unexpected_count": len(unexpected),
        "unexpected_keys": unexpected,
        "complete": not missing and not incomplete and not unexpected,
    }

    if split.startswith("GEU"):
        correct_by_task: Counter[str] = Counter()
        total_by_task: Counter[str] = Counter()
        predictions: Counter[str] = Counter()
        for record in ordered:
            prediction = extract_choice(str(record["response"]))
            predictions[prediction or "<empty>"] += 1
            task_type = str(record["task_type"])
            total_by_task[task_type] += 1
            if prediction == record["answer"]:
                correct_by_task[task_type] += 1
        total_correct = sum(correct_by_task.values())
        total_count = sum(total_by_task.values())
        summary["official_local_accuracy"] = total_correct / total_count if total_count else None
        summary["correct"] = total_correct
        summary["total"] = total_count
        summary["accuracy_by_task"] = {
            task: {
                "correct": correct_by_task[task],
                "total": total_by_task[task],
                "accuracy": (
                    correct_by_task[task] / total_by_task[task] if total_by_task[task] else None
                ),
            }
            for task in sorted(total_by_task)
        }
        summary["prediction_counts"] = dict(sorted(predictions.items()))
        summary["scoring_protocol"] = "RealUnify eval/eval_understanding.py regex"
    else:
        summary["generated_image_count"] = len(ordered)
        summary["official_score"] = None
        summary["scoring_protocol"] = (
            "Pending Gemini 2.5 Pro polling over every question_list item; "
            "a sample scores 1 only when every yes/no answer matches."
        )

    write_json_atomic(summary, split_root / "summary.json")
    if getattr(args, "require_complete", False) and not summary["complete"]:
        raise RuntimeError(
            f"{split} aggregation is incomplete: "
            f"missing={len(missing)}, incomplete={len(incomplete)}, "
            f"unexpected={len(unexpected)}"
        )
    return summary


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.resolve()

    if args.command == "validate":
        summary = validate_dataset(args.data_root)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["valid"] else 1

    args.output_root = args.output_root.resolve()
    if args.command == "aggregate":
        summaries = [aggregate_split(args, split) for split in args.splits]
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return 0

    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num_shards")
    if args.max_samples < 0 or args.samples_per_task < 0:
        raise ValueError("sample limits cannot be negative")
    if not (args.model_path / "ema.safetensors").is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.model_path / 'ema.safetensors'}")

    data_summary = validate_dataset(args.data_root)
    if not data_summary["valid"]:
        raise RuntimeError(f"RealUnify dataset validation failed: {data_summary}")

    args.offload_folder.mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline(args)
    run_summaries = [run_split(args, pipeline, split) for split in args.splits]
    aggregate_summaries = []
    if args.num_shards == 1:
        aggregate_summaries = [aggregate_split(args, split) for split in args.splits]
    print(
        json.dumps(
            {"runs": run_summaries, "aggregates": aggregate_summaries},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
