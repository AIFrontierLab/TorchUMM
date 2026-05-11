from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from umm.post_training.unipath.planner.models import PATHS, load_planner, planner_probs, prepare_feature_tensor
from umm.post_training.unipath.planner.qfcp_policy import (
    QFCP_REFINED_PRAGMATIC_SAFE_POLICY,
    build_mmmu_query,
    choose_qfcp_refined_pragmatic_safe_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay MMMU 5-path cached outputs with a UniPath planner.")
    parser.add_argument("--planner-path", required=True)
    parser.add_argument("--features-pt", required=True, help="MMMU planner feature cache.")
    parser.add_argument("--results-root", required=True, help="Root containing mmmu_direct/mmmu_l0/... cached cases.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy", choices=["qfcp_refined_pragmatic_safe", "raw_planner"], default="qfcp_refined_pragmatic_safe")
    parser.add_argument(
        "--query-mode",
        choices=["raw", "mmmu_query"],
        default="raw",
        help="raw matches the reported MMMU offline replay; mmmu_query appends options and answer instruction.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--planner-predictions-jsonl", default="", help="Optional cached planner scores to reuse.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_case_maps(results_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    by_path: dict[str, dict[str, dict[str, Any]]] = {}
    expected_ids: set[str] | None = None
    for path_name in PATHS:
        candidates = [
            results_root / f"mmmu_{path_name}" / "cases_with_lengths.jsonl",
            results_root / f"mmmu_{path_name}" / "cases.jsonl",
            results_root / path_name / "cases_with_lengths.jsonl",
            results_root / path_name / "cases.jsonl",
        ]
        cases_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if cases_path is None:
            raise FileNotFoundError(candidates[0])
        rows = read_jsonl(cases_path)
        case_map = {str(row["data_id"]): row for row in rows}
        ids = set(case_map)
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"MMMU path ids mismatch for {path_name}: {len(ids)} vs {len(expected_ids)}")
        by_path[path_name] = case_map
    return by_path


def load_feature_map(features_pt: Path, bagel_feature_mode: str) -> tuple[dict[str, torch.Tensor], list[str]]:
    payload = torch.load(features_pt, map_location="cpu")
    features = prepare_feature_tensor(payload, bagel_feature_mode)
    metadata = list(payload["metadata"])
    if len(metadata) != int(features.shape[0]):
        raise ValueError(f"Feature/metadata length mismatch: {features.shape[0]} vs {len(metadata)}")
    return {str(row["data_id"]): features[idx] for idx, row in enumerate(metadata)}, [str(row["data_id"]) for row in metadata]


def default_choose_path(planner_scores: dict[str, float], margin: float) -> str:
    best = max(float(v) for v in planner_scores.values())
    candidates = [path for path in PATHS if float(planner_scores[path]) >= best - margin]
    return min(candidates, key=lambda path: PATHS.index(path))


@torch.no_grad()
def build_planner_predictions(args: argparse.Namespace, case_maps: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    if args.planner_predictions_jsonl:
        return read_jsonl(Path(args.planner_predictions_jsonl))

    device = torch.device(args.device)
    model, payload = load_planner(args.planner_path, device=device)
    margin = float(payload.get("margin", 0.03))
    bagel_feature_mode = str(payload.get("bagel_feature_mode") or "joint_text_mean_last_image")
    features_by_id, feature_ids = load_feature_map(Path(args.features_pt), bagel_feature_mode)

    ordered_ids = [data_id for data_id in feature_ids if data_id in case_maps["direct"]]
    if args.max_samples > 0:
        ordered_ids = ordered_ids[: int(args.max_samples)]
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(ordered_ids), int(args.batch_size)):
        batch_ids = ordered_ids[start : start + int(args.batch_size)]
        x = torch.stack([features_by_id[data_id].float() for data_id in batch_ids]).to(device)
        probs = planner_probs(model, x).detach().cpu()
        for offset, data_id in enumerate(batch_ids):
            case = case_maps["direct"][data_id]
            scores = {PATHS[j]: float(probs[offset, j].item()) for j in range(len(PATHS))}
            default_selected = default_choose_path(scores, margin)
            if args.query_mode == "mmmu_query":
                query = build_mmmu_query(
                    str(case.get("question") or ""),
                    str(case.get("question_type") or ""),
                    list(case.get("options") or []),
                )
            else:
                query = str(case.get("question") or "").strip()
            predictions.append(
                {
                    "data_id": data_id,
                    "query": query,
                    "query_mode": args.query_mode,
                    "question_type": str(case.get("question_type") or ""),
                    "planner_scores": scores,
                    "selected_path": default_selected,
                    "default_selected_path": default_selected,
                    "answer_ok_by_path": {path: bool(case_maps[path][data_id].get("answer_ok")) for path in PATHS},
                }
            )
    return predictions


def select_path(policy: str, planner_scores: dict[str, float], query: str, default_path: str) -> tuple[str, str, str]:
    if policy == "raw_planner":
        return default_path, "raw_planner", "raw_planner"
    return choose_qfcp_refined_pragmatic_safe_path(planner_scores, query)


def aggregate_predictions(
    predictions: list[dict[str, Any]],
    case_maps: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
    policy: str,
) -> dict[str, Any]:
    path_counts = Counter()
    bucket_counts = Counter()
    selector_counts = Counter()
    answer_hits = 0
    token_sum = 0
    cases_out: list[dict[str, Any]] = []
    routed_predictions: list[dict[str, Any]] = []
    for row in predictions:
        data_id = str(row["data_id"])
        selected_path, bucket, selector = select_path(
            policy,
            {path: float(row["planner_scores"][path]) for path in PATHS},
            str(row.get("query") or ""),
            str(row.get("default_selected_path") or row.get("selected_path") or "direct"),
        )
        chosen = dict(case_maps[selected_path][data_id])
        answer_ok = bool(chosen.get("answer_ok"))
        path_counts[selected_path] += 1
        bucket_counts[bucket] += 1
        selector_counts[selector] += 1
        answer_hits += int(answer_ok)
        token_sum += int(chosen.get("raw_response_token_len", 0))
        routed_predictions.append({**row, "selected_path": selected_path, "bucket": bucket, "selector": selector})
        cases_out.append(
            {
                "data_id": data_id,
                "selected_path": selected_path,
                "bucket": bucket,
                "selector": selector,
                "planner_scores": row["planner_scores"],
                "answer_ok": answer_ok,
                "pred_answer": chosen.get("pred_answer"),
                "gt_answer": chosen.get("gt_answer"),
                "raw_response": chosen.get("raw_response"),
                "raw_response_token_len": chosen.get("raw_response_token_len"),
                "question": chosen.get("question"),
                "options": chosen.get("options"),
                "question_type": chosen.get("question_type"),
                "image_paths": chosen.get("image_paths"),
                "prompt_used": chosen.get("prompt_used"),
                "source_case_path": selected_path,
            }
        )

    write_jsonl(output_dir / "planner_predictions_with_policy.jsonl", routed_predictions)
    write_jsonl(output_dir / "cases.jsonl", cases_out)
    return {
        "benchmark": "mmmu",
        "policy": policy,
        "query_mode": str(predictions[0].get("query_mode") or "") if predictions else "",
        "num_samples": len(predictions),
        "answer_accuracy": answer_hits / max(len(predictions), 1),
        "correct": answer_hits,
        "path_counts": dict(path_counts),
        "bucket_counts": dict(bucket_counts),
        "selector_counts": dict(selector_counts),
        "mean_raw_response_token_len": token_sum / max(len(predictions), 1),
        "cases_jsonl": str(output_dir / "cases.jsonl"),
        "planner_predictions_with_policy_jsonl": str(output_dir / "planner_predictions_with_policy.jsonl"),
    }


def main() -> int:
    args = parse_args()
    started_at = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_maps = load_case_maps(Path(args.results_root))
    predictions = build_planner_predictions(args, case_maps)
    metrics = aggregate_predictions(predictions, case_maps, output_dir, args.policy)
    metrics.update(
        {
            "planner_path": str(Path(args.planner_path)),
            "features_pt": str(Path(args.features_pt)),
            "results_root": str(Path(args.results_root)),
            "query_mode": args.query_mode,
            "qfcp_policy": QFCP_REFINED_PRAGMATIC_SAFE_POLICY if args.policy == "qfcp_refined_pragmatic_safe" else None,
            "elapsed_seconds": time.time() - started_at,
        }
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
