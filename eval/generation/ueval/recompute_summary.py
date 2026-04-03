#!/usr/bin/env python3
"""
Recompute per-task UEval scores from existing eval_results.json
by matching item IDs against the dataset to get task labels.

Usage:
    python eval/generation/ueval/recompute_summary.py \
        --eval_results /path/to/eval_results.json \
        --dataset_cache /path/to/UEval   # or --hf_dataset primerL/UEval-all
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_results", required=True)
    parser.add_argument("--hf_dataset", default="primerL/UEval-all")
    parser.add_argument("--dataset_cache", default=None)
    args = parser.parse_args()

    # Load eval results
    eval_data = json.loads(Path(args.eval_results).read_text())
    results = eval_data.get("results", [])
    print(f"Loaded {len(results)} results")

    # Load dataset for id -> task mapping
    from datasets import load_dataset
    if args.dataset_cache and Path(args.dataset_cache).is_dir():
        ds = load_dataset(args.dataset_cache, split="test")
    else:
        ds = load_dataset(args.hf_dataset, split="test")

    id_to_task = {}
    for item in ds:
        item_id = item.get("id")
        task = item.get("task") or item.get("task_type") or ""
        if item_id is not None:
            id_to_task[str(item_id)] = task.lower().strip()
            id_to_task[item_id] = task.lower().strip()

    # Per-task scoring
    task_text: Dict[str, List[float]] = {}
    task_image: Dict[str, List[float]] = {}
    task_overall: Dict[str, List[float]] = {}

    no_task_count = 0
    for r in results:
        rid = r.get("id")
        task = id_to_task.get(str(rid), "") if rid is not None else ""
        if not task:
            task = id_to_task.get(rid, "") if rid is not None else ""
        if not task:
            no_task_count += 1
            task = "unknown"

        t_rate = r.get("text_score", {}).get("rate")
        i_rate = r.get("image_score", {}).get("rate")

        if t_rate is not None:
            task_text.setdefault(task, []).append(t_rate)
            task_overall.setdefault(task, []).append(t_rate)
        if i_rate is not None:
            task_image.setdefault(task, []).append(i_rate)
            task_overall.setdefault(task, []).append(i_rate)

    if no_task_count:
        print(f"WARNING: {no_task_count} items could not be matched to a task")

    # Print table
    all_tasks = sorted(set(list(task_text.keys()) + list(task_image.keys())))

    print(f"\n{'Task':12s} | {'Text':>8s} | {'Image':>8s} | {'Overall':>8s} | {'Items':>5s}")
    print("-" * 55)

    total_text_rates = []
    total_image_rates = []

    for task in all_tasks:
        t_rates = task_text.get(task, [])
        i_rates = task_image.get(task, [])
        o_rates = task_overall.get(task, [])

        t_avg = sum(t_rates) / len(t_rates) * 100 if t_rates else 0
        i_avg = sum(i_rates) / len(i_rates) * 100 if i_rates else 0
        o_avg = sum(o_rates) / len(o_rates) * 100 if o_rates else 0
        n_items = max(len(t_rates), len(i_rates))

        total_text_rates.extend(t_rates)
        total_image_rates.extend(i_rates)

        print(f"{task:12s} | {t_avg:7.1f}% | {i_avg:7.1f}% | {o_avg:7.1f}% | {n_items:5d}")

    all_overall = total_text_rates + total_image_rates
    t_total = sum(total_text_rates) / len(total_text_rates) * 100 if total_text_rates else 0
    i_total = sum(total_image_rates) / len(total_image_rates) * 100 if total_image_rates else 0
    o_total = sum(all_overall) / len(all_overall) * 100 if all_overall else 0

    print("-" * 55)
    print(f"{'TOTAL':12s} | {t_total:7.1f}% | {i_total:7.1f}% | {o_total:7.1f}% | {len(results):5d}")

    # Also check a few image results for debugging
    print(f"\n--- Sample image results (first 3 items with image rubrics) ---")
    count = 0
    for r in results:
        img_results = r.get("image_results", [])
        if img_results:
            print(f"\nItem {r.get('id')} (task={id_to_task.get(str(r.get('id')), '?')}):")
            print(f"  image_outputs: {r.get('image_outputs', [])}")
            for ir in img_results[:2]:
                print(f"  criterion: {ir.get('criterion', '')[:80]}...")
                print(f"  criteria_met: {ir.get('criteria_met')}")
                print(f"  raw_response: {str(ir.get('raw_response', ''))[:120]}...")
            count += 1
            if count >= 3:
                break


if __name__ == "__main__":
    main()
