#!/usr/bin/env python3
"""
Unified-Bench scoring — computes similarity between reference images and the
round-trip-generated images produced by run_generation.py.

For each enabled scorer (clip / dinov2 / dinov3 / longclip), computes per-image
cosine similarity and aggregates min / max / average. Output JSON format
mirrors model/UAE/Unified-Bench/results/example.json for direct comparison
with the UAE paper numbers.

Called via subprocess from cli/unified_bench.py.

Usage:
    python eval/generation/unified_bench/run_scoring.py --config <config.yaml>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from umm.core.config import load_config

# Make scorers importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorers import load_scorer  # noqa: E402


def _resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else repo_root / path


def _natural_key(path: Path) -> Tuple[int, str]:
    stem = path.stem
    m = re.match(r"^(\d+)$", stem)
    if m:
        return (0, f"{int(m.group(1)):012d}")
    return (1, stem)


def _build_pairs(ref_dir: Path, gen_dir: Path) -> List[Tuple[str, Path, Path]]:
    """Match each reference image with a generated image by stem.

    Ref files may have arbitrary extensions (jpg/png); generated images are
    always .png (saved by run_generation.py). Pairs with a missing gen image
    are skipped and logged.
    """
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ref_files = sorted(
        [p for p in ref_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=_natural_key,
    )

    pairs: List[Tuple[str, Path, Path]] = []
    missing: List[str] = []
    for ref in ref_files:
        gen = gen_dir / f"{ref.stem}.png"
        if not gen.is_file():
            # Try alternative extensions in case the backbone saved .jpg etc.
            alt = None
            for ext in (".jpg", ".jpeg", ".webp"):
                cand = gen_dir / f"{ref.stem}{ext}"
                if cand.is_file():
                    alt = cand
                    break
            if alt is None:
                missing.append(ref.stem)
                continue
            gen = alt
        pairs.append((ref.stem, ref, gen))

    if missing:
        print(f"[unified_bench score] WARNING: {len(missing)} gen images missing: {missing[:10]}...")
    return pairs


def _score_with(scorer, pairs: List[Tuple[str, Path, Path]], name: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    per_image: Dict[str, float] = {}
    first_logged = False
    for item_id, ref, gen in tqdm(pairs, desc=f"[{name}]"):
        try:
            sim = scorer.score_pair(str(ref), str(gen))
            sim_f = float(sim)
            per_image[f"{gen.name}"] = sim_f
            if not first_logged:
                print(f"[{name}] first pair sample — id={item_id}, sim={sim_f:.4f}")
                first_logged = True
        except Exception as exc:
            print(f"[{name}] id={item_id} scoring error: {exc}")
            if not first_logged:
                # Show full traceback for the first failure only, to pinpoint root cause.
                traceback.print_exc()
                first_logged = True
            per_image[f"{gen.name}"] = float("nan")

    valid = [v for v in per_image.values() if v == v]  # filter NaN
    n_errors = len(pairs) - len(valid)
    if n_errors:
        print(f"[{name}] WARNING: {n_errors}/{len(pairs)} pairs failed scoring (NaN)")
    stats = {
        "total_images": len(pairs),
        "valid_pairs": len(valid),
        "average_similarity": float(sum(valid) / len(valid)) if valid else 0.0,
        "min_similarity": float(min(valid)) if valid else 0.0,
        "max_similarity": float(max(valid)) if valid else 0.0,
    }
    return per_image, stats


def _load_eval_cfg(config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw_cfg = load_config(config_path)
    bench_cfg = raw_cfg.get("unified_bench", {}) if isinstance(raw_cfg.get("unified_bench"), dict) else {}
    inference_cfg = raw_cfg.get("inference", {}) if isinstance(raw_cfg.get("inference"), dict) else {}
    return bench_cfg, inference_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified-Bench scoring")
    parser.add_argument("--config", required=True, help="Path to eval config YAML")
    args = parser.parse_args()

    bench_cfg, inference_cfg = _load_eval_cfg(args.config)
    repo_root = Path(__file__).resolve().parents[3]

    backbone = str(inference_cfg.get("backbone", "unknown")).strip()

    ref_dir = _resolve_path(
        str(bench_cfg.get("ref_dir", "/datasets/unified_bench/Image")), repo_root
    )
    out_dir = _resolve_path(
        str(bench_cfg.get("out_dir", f"output/unified_bench/{backbone}")), repo_root
    )
    gen_dir = out_dir / "images"

    score_cfg = bench_cfg.get("scoring", {}) or {}
    if not isinstance(score_cfg, dict):
        score_cfg = {}
    model_names: List[str] = list(score_cfg.get("models", ["clip", "dinov2", "dinov3", "longclip"]))
    model_paths: Dict[str, str] = dict(score_cfg.get("model_paths", {}) or {})
    device = str(score_cfg.get("device", "cuda"))

    score_output_path_raw = bench_cfg.get("score_output_path")
    if score_output_path_raw:
        score_output_path = _resolve_path(str(score_output_path_raw), repo_root)
    else:
        score_output_path = out_dir / "summary.json"
    score_output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[unified_bench score] backbone={backbone}, ref_dir={ref_dir}, gen_dir={gen_dir}")
    print(f"[unified_bench score] scorers={model_names}")

    if not gen_dir.is_dir():
        raise FileNotFoundError(f"Generated images dir not found: {gen_dir}")
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"Reference images dir not found: {ref_dir}")

    pairs = _build_pairs(ref_dir, gen_dir)
    if not pairs:
        raise RuntimeError(
            f"No (ref, gen) pairs matched. ref_dir={ref_dir}, gen_dir={gen_dir}"
        )
    print(f"[unified_bench score] matched {len(pairs)} pairs")

    similarities_by_model: Dict[str, Dict[str, float]] = {}
    statistics_by_model: Dict[str, Dict[str, float]] = {}
    skipped_models: Dict[str, str] = {}

    for name in model_names:
        path = model_paths.get(name)
        if not path:
            skipped_models[name] = "no model_path configured"
            print(f"[unified_bench score] SKIP {name}: no model_path in config")
            continue
        try:
            scorer = load_scorer(name, model_path=path, device=device)
        except Exception as exc:
            skipped_models[name] = f"load failed: {exc}"
            print(f"[unified_bench score] SKIP {name}: {exc}")
            traceback.print_exc()
            continue

        per_image, stats = _score_with(scorer, pairs, name)
        similarities_by_model[name] = per_image
        statistics_by_model[name] = stats
        # Free GPU memory between scorers.
        try:
            import torch
            del scorer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    summary_averages = {
        name: stats.get("average_similarity", 0.0)
        for name, stats in statistics_by_model.items()
    }
    if summary_averages:
        best = max(summary_averages.items(), key=lambda kv: kv[1])
        worst = min(summary_averages.items(), key=lambda kv: kv[1])
        overall_avg = sum(summary_averages.values()) / len(summary_averages)
    else:
        best = worst = ("none", 0.0)
        overall_avg = 0.0

    output = {
        "backbone": backbone,
        "ref_dir": str(ref_dir),
        "gen_dir": str(gen_dir),
        "model_types": list(similarities_by_model.keys()),
        "similarities_by_model": similarities_by_model,
        "statistics_by_model": statistics_by_model,
        "summary": {
            "overall_average": float(overall_avg),
            "best_model": [best[0], float(best[1])],
            "worst_model": [worst[0], float(worst[1])],
            "averages_by_model": summary_averages,
        },
        "skipped_models": skipped_models,
    }

    score_output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[unified_bench score] wrote {score_output_path}")
    print(f"[unified_bench score] averages: {summary_averages}")
    print(f"[unified_bench score] skipped: {skipped_models}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
