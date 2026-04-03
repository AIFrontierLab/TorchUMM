#!/usr/bin/env python3
"""ImgEdit-Bench scoring — evaluate edited images using Qwen2.5-VL-72B.

Supports three suites:
  - singleturn: type-specific prompts from prompts.json (9 edit types)
  - uge:        unified prompt from uge_prompt.txt
  - multiturn:  unified prompt, scores each turn's (input, output) pair
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

_MULTITURN_CATEGORIES = ["content_memory", "content_understand", "version_backtrace"]


def load_model(model_path: str | None = None):
    """Load Qwen2.5-VL model — reuse the wrapper from GEdit's VIEScore."""
    from viescore.mllm_tools.qwen25vl_eval import Qwen25VL
    return Qwen25VL(model_path=model_path)


def _score_pair(model, prompt: str, origin_path: str, edited_path: str) -> str | None:
    """Score a single (original, edited) image pair."""
    messages = model.prepare_prompt(
        image_links=[origin_path, edited_path],
        text_prompt=prompt,
    )
    try:
        return model.get_parsed_output(messages)
    except Exception as e:
        print(f"[imgedit score] error: {e}", flush=True)
        return None


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Singleturn scoring
# ---------------------------------------------------------------------------

def score_singleturn(model, args, benchmark_data: Path, edited_images_dir: Path,
                     save_dir: Path, origin_img_root: Path) -> None:
    edit_json = benchmark_data / "basic_edit.json"
    prompts_json = benchmark_data / "prompts.json"
    with open(edit_json, "r", encoding="utf-8") as f:
        edit_data = json.load(f)
    with open(prompts_json, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    result_path = save_dir / "result.json"
    results = {}
    if result_path.is_file():
        with open(result_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    if args.edit_type != "all":
        items = {k: v for k, v in edit_data.items() if v.get("edit_type") == args.edit_type}
    else:
        items = edit_data

    to_score = {k: v for k, v in items.items() if k not in results}
    print(f"[singleturn score] total={len(items)}, to_score={len(to_score)}, done={len(items)-len(to_score)}", flush=True)

    for key, item in tqdm(to_score.items(), desc="[singleturn score]"):
        edit_type = item.get("edit_type", "")
        if edit_type not in prompts:
            continue
        edited_path = edited_images_dir / f"{key}.png"
        if not edited_path.is_file():
            continue
        origin_path = origin_img_root / "singleturn" / item["id"]
        if not origin_path.is_file():
            continue

        full_prompt = prompts[edit_type].replace("<edit_prompt>", item["prompt"])
        response = _score_pair(model, full_prompt, str(origin_path), str(edited_path))
        if response is not None:
            results[key] = response
        if len(results) % 50 == 0:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[singleturn score] done — {len(results)} items saved to {result_path}", flush=True)


# ---------------------------------------------------------------------------
# UGE scoring
# ---------------------------------------------------------------------------

def score_uge(model, benchmark_data: Path, edited_images_dir: Path,
              save_dir: Path, origin_img_root: Path) -> None:
    uge_prompt_path = benchmark_data / "uge_prompt.txt"
    uge_prompt_template = uge_prompt_path.read_text(encoding="utf-8")

    # Load UGE annotations from Benchmark/hard/annotation.jsonl (ground truth)
    uge_img_root = origin_img_root / "hard"
    ann_path = uge_img_root / "annotation.jsonl"
    items = _load_jsonl(ann_path)

    result_path = save_dir / "result.json"
    results = {}
    if result_path.is_file():
        with open(result_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    to_score = [(i, item) for i, item in enumerate(items)
                if f"uge_{Path(item['id']).stem}" not in results]
    print(f"[uge score] total={len(items)}, to_score={len(to_score)}, done={len(items)-len(to_score)}", flush=True)

    for idx, item in tqdm(to_score, desc="[uge score]"):
        img_id = item["id"]
        key = Path(img_id).stem
        edited_path = edited_images_dir / f"uge_{key}.png"
        if not edited_path.is_file():
            continue
        origin_path = uge_img_root / img_id
        if not origin_path.is_file():
            continue

        full_prompt = uge_prompt_template.replace("<edit_prompt>", item["prompt"])
        response = _score_pair(model, full_prompt, str(origin_path), str(edited_path))
        if response is not None:
            results[f"uge_{key}"] = response
        if len(results) % 20 == 0:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[uge score] done — {len(results)} items saved to {result_path}", flush=True)


# ---------------------------------------------------------------------------
# Multiturn scoring
# ---------------------------------------------------------------------------

def score_multiturn(model, benchmark_data: Path, edited_images_dir: Path,
                    save_dir: Path, origin_img_root: Path) -> None:
    uge_prompt_path = benchmark_data / "uge_prompt.txt"
    uge_prompt_template = uge_prompt_path.read_text(encoding="utf-8")

    multiturn_root = origin_img_root / "multiturn"

    result_path = save_dir / "result.json"
    results = {}
    if result_path.is_file():
        with open(result_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    for category in _MULTITURN_CATEGORIES:
        cat_dir = multiturn_root / category
        ann_path = cat_dir / "annotation.json"
        if not ann_path.is_file():
            continue

        items = _load_jsonl(ann_path)
        mt_images_dir = edited_images_dir / f"multiturn_{category}"

        for item in tqdm(items, desc=f"[multiturn score] {category}"):
            img_id = item["id"]
            base_name = Path(img_id).stem
            origin_path = cat_dir / img_id

            if not origin_path.is_file():
                continue

            # Collect turns
            turns = []
            for t in range(1, 10):
                if f"turn{t}" in item:
                    turns.append(item[f"turn{t}"])
                else:
                    break

            # Score each turn: compare (previous input, turn output)
            prev_image_path = str(origin_path)
            for turn_idx, turn_prompt in enumerate(turns, start=1):
                result_key = f"mt_{category}_{base_name}_turn{turn_idx}"
                if result_key in results:
                    # Update prev for next turn
                    turn_output = mt_images_dir / f"{base_name}_turn{turn_idx}.png"
                    if turn_output.is_file():
                        prev_image_path = str(turn_output)
                    continue

                turn_output = mt_images_dir / f"{base_name}_turn{turn_idx}.png"
                if not turn_output.is_file():
                    break  # Cannot score subsequent turns

                full_prompt = uge_prompt_template.replace("<edit_prompt>", turn_prompt)
                response = _score_pair(model, full_prompt, prev_image_path, str(turn_output))
                if response is not None:
                    results[result_key] = response

                prev_image_path = str(turn_output)

        # Save after each category
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[multiturn score] done — {len(results)} items saved to {result_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ImgEdit-Bench Scoring with Qwen2.5-VL")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--edited_images_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--origin_img_root", required=True)
    parser.add_argument("--benchmark_data", required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--suite", default="singleturn", choices=["singleturn", "uge", "multiturn"])
    parser.add_argument("--edit_type", default="all")
    args = parser.parse_args()

    edited_images_dir = Path(args.edited_images_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    origin_img_root = Path(args.origin_img_root)
    benchmark_data = Path(args.benchmark_data)

    print(f"[imgedit score] loading Qwen2.5-VL: {args.model_path or 'default'}", flush=True)
    model = load_model(args.model_path)

    if args.suite == "singleturn":
        score_singleturn(model, args, benchmark_data, edited_images_dir, save_dir, origin_img_root)
    elif args.suite == "uge":
        score_uge(model, benchmark_data, edited_images_dir, save_dir, origin_img_root)
    elif args.suite == "multiturn":
        score_multiturn(model, benchmark_data, edited_images_dir, save_dir, origin_img_root)


if __name__ == "__main__":
    main()
