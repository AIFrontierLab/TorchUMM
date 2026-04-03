#!/usr/bin/env python3
"""ImgEdit-Bench statistics — aggregate scoring results.

Supports three suites:
  - singleturn: per-image avg → per-edit-type avg → overall
  - uge:        per-image score → overall avg
  - multiturn:  per-turn score → per-category avg → overall
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def extract_scores_and_average(entry: str) -> float | None:
    """Parse a Qwen2.5-VL response to extract numeric scores and average them."""
    lines = entry.splitlines()
    scores = []
    for line in lines:
        parts = line.strip().split(": ")
        if len(parts) == 2 and parts[1].strip().isdigit():
            scores.append(int(parts[1].strip()))
    if scores:
        return round(sum(scores) / len(scores), 2)
    # Fallback: find any standalone digit 1-5
    if not scores:
        matches = re.findall(r'\b([1-5])\b', entry)
        if matches:
            scores = [int(m) for m in matches]
            return round(sum(scores) / len(scores), 2)
    return None


def _print_table(title: str, data: dict, count_fn=None) -> None:
    print(f"\n[imgedit stats] {title}:", flush=True)
    print(f"{'Category':<25} {'Score':>6} {'Count':>6}", flush=True)
    print(f"{'-'*40}", flush=True)
    total_score = 0.0
    total_count = 0
    for key, score in sorted(data.items()):
        count = count_fn(key) if count_fn else 1
        print(f"{key:<25} {score:>6.2f} {count:>6}", flush=True)
        total_score += score * count
        total_count += count
    if total_count > 0:
        overall = round(total_score / total_count, 2)
        print(f"{'-'*40}", flush=True)
        print(f"{'Overall':<25} {overall:>6.2f} {total_count:>6}", flush=True)


# ---------------------------------------------------------------------------
# Singleturn statistics
# ---------------------------------------------------------------------------

def stats_singleturn(scores_dir: Path, benchmark_data: Path) -> None:
    result_path = scores_dir / "result.json"
    if not result_path.is_file():
        print("[singleturn stats] result.json not found, skipping", flush=True)
        return
    with open(result_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)
    with open(benchmark_data / "basic_edit.json", "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # Per-image average
    avg_scores = {}
    for key, value in result_data.items():
        if isinstance(value, str):
            avg = extract_scores_and_average(value)
            if avg is not None:
                avg_scores[key] = avg

    avg_path = scores_dir / "average_score.json"
    with open(avg_path, "w", encoding="utf-8") as f:
        json.dump(avg_scores, f, indent=2, ensure_ascii=False)

    # Per-type average
    type_scores = defaultdict(list)
    for key, score in avg_scores.items():
        etype = meta_data.get(key, {}).get("edit_type")
        if etype:
            type_scores[etype].append(score)
    type_avg = {t: round(sum(s) / len(s), 2) for t, s in type_scores.items() if s}

    type_path = scores_dir / "typescore.json"
    with open(type_path, "w", encoding="utf-8") as f:
        json.dump(type_avg, f, indent=2, ensure_ascii=False)

    _print_table("Singleturn per-type scores", type_avg,
                 count_fn=lambda t: len(type_scores[t]))
    print(f"[singleturn stats] saved: {avg_path}, {type_path}", flush=True)


# ---------------------------------------------------------------------------
# UGE statistics
# ---------------------------------------------------------------------------

def stats_uge(scores_dir: Path) -> None:
    result_path = scores_dir / "result.json"
    if not result_path.is_file():
        print("[uge stats] result.json not found, skipping", flush=True)
        return
    with open(result_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    scores = {}
    for key, value in result_data.items():
        if isinstance(value, str):
            avg = extract_scores_and_average(value)
            if avg is not None:
                scores[key] = avg

    avg_path = scores_dir / "average_score.json"
    with open(avg_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    if scores:
        overall = round(sum(scores.values()) / len(scores), 2)
        print(f"\n[uge stats] UGE overall: {overall:.2f} ({len(scores)} items)", flush=True)
    print(f"[uge stats] saved: {avg_path}", flush=True)


# ---------------------------------------------------------------------------
# Multiturn statistics
# ---------------------------------------------------------------------------

def stats_multiturn(scores_dir: Path) -> None:
    result_path = scores_dir / "result.json"
    if not result_path.is_file():
        print("[multiturn stats] result.json not found, skipping", flush=True)
        return
    with open(result_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    # Group by category: keys are mt_{category}_{basename}_turn{n}
    cat_scores = defaultdict(list)
    for key, value in result_data.items():
        if not isinstance(value, str):
            continue
        avg = extract_scores_and_average(value)
        if avg is None:
            continue
        # Extract category from key: mt_{category}_{rest}
        parts = key.split("_", 2)
        if len(parts) >= 2:
            category = parts[1]
            # Handle multi-word categories like content_memory
            for cat in ["content_memory", "content_understand", "version_backtrace"]:
                if key.startswith(f"mt_{cat}_"):
                    category = cat
                    break
            cat_scores[category].append(avg)

    cat_avg = {c: round(sum(s) / len(s), 2) for c, s in cat_scores.items() if s}

    avg_path = scores_dir / "category_score.json"
    with open(avg_path, "w", encoding="utf-8") as f:
        json.dump(cat_avg, f, indent=2, ensure_ascii=False)

    _print_table("Multiturn per-category scores", cat_avg,
                 count_fn=lambda c: len(cat_scores[c]))
    print(f"[multiturn stats] saved: {avg_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ImgEdit-Bench Statistics")
    parser.add_argument("--scores_dir", required=True)
    parser.add_argument("--benchmark_data", required=True)
    parser.add_argument("--suite", default="singleturn", choices=["singleturn", "uge", "multiturn"])
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    benchmark_data = Path(args.benchmark_data)

    if args.suite == "singleturn":
        stats_singleturn(scores_dir, benchmark_data)
    elif args.suite == "uge":
        stats_uge(scores_dir)
    elif args.suite == "multiturn":
        stats_multiturn(scores_dir)


if __name__ == "__main__":
    main()
