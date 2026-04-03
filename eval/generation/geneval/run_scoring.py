#!/usr/bin/env python3
"""
GenEval scoring script — evaluates generated images using Mask2Former object
detection (evaluate_images.py) and computes summary scores (summary_scores.py).

Called via subprocess from the geneval.py CLI wrapper.

Usage:
    python eval/generation/geneval/run_scoring.py --config configs/eval/geneval/geneval_bagel.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from umm.core.config import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


# ---------------------------------------------------------------------------
# Object-detection evaluation (Mask2Former via evaluate_images.py)
# ---------------------------------------------------------------------------

def _run_object_detection_eval(
    eval_dir: Path,
    images_dir: Path,
    results_file: Path,
    eval_cfg: Dict[str, Any],
) -> bool:
    """Run GenEval evaluate_images.py using Mask2Former."""
    eval_script = eval_dir / "evaluate_images.py"
    if not eval_script.exists():
        print(f"[geneval-score] evaluate_images.py not found at {eval_script}")
        return False

    model_path = str(eval_cfg.get("model_path", "./"))
    model_config = eval_cfg.get("model_config")
    eval_python = str(eval_cfg.get("eval_python", sys.executable))

    cmd = [
        eval_python, str(eval_script),
        str(images_dir),
        "--outfile", str(results_file),
        "--model-path", model_path,
    ]
    if model_config:
        cmd += ["--model-config", str(model_config)]

    options = eval_cfg.get("options", {})
    if isinstance(options, dict) and options:
        cmd.append("--options")
        for k, v in options.items():
            cmd.append(f"{k}={v}")

    results_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[geneval-score] running evaluate_images.py ...")
    print(f"[geneval-score] command: {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(eval_dir)).returncode
    if rc != 0:
        print(f"[geneval-score] evaluate_images.py returned rc={rc}")
        return False
    return True


# ---------------------------------------------------------------------------
# Summary scoring (summary_scores.py)
# ---------------------------------------------------------------------------

def _run_summary_scores(
    eval_dir: Path,
    results_file: Path,
    eval_cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Run GenEval summary_scores.py and parse output."""
    summary_script = eval_dir / "summary_scores.py"
    if not summary_script.exists():
        print(f"[geneval-score] summary_scores.py not found at {summary_script}")
        return None
    if not results_file.exists():
        print(f"[geneval-score] results file not found: {results_file}")
        return None

    eval_python = str(eval_cfg.get("eval_python", sys.executable))
    cmd = [eval_python, str(summary_script), str(results_file)]

    print("[geneval-score] running summary_scores.py ...")
    proc = subprocess.run(cmd, cwd=str(eval_dir), capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(f"[geneval-score] summary_scores.py failed (rc={proc.returncode}): {proc.stderr}")
        return None

    scores: Dict[str, Any] = {"stdout": proc.stdout}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "Overall score" in line and ":" in line:
            try:
                scores["overall"] = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "=" in line and not line.startswith("="):
            parts = line.split("=")
            if len(parts) == 2:
                tag = parts[0].strip()
                val_str = parts[1].strip()
                paren_idx = val_str.find("(")
                if paren_idx != -1:
                    val_str = val_str[:paren_idx].strip()
                try:
                    scores[tag] = float(val_str.strip("%")) / 100
                except ValueError:
                    pass
    return scores


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_eval_cfg(
    config_path: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw_cfg = load_config(config_path)
    geneval_cfg = (
        raw_cfg.get("geneval", {})
        if isinstance(raw_cfg.get("geneval"), dict)
        else {}
    )
    inference_cfg = (
        raw_cfg.get("inference", {})
        if isinstance(raw_cfg.get("inference"), dict)
        else {}
    )
    return geneval_cfg, inference_cfg


def _get_backbone_name(inference_cfg: Dict[str, Any], repo_root: Path) -> str:
    """Resolve backbone name, following infer_config reference if needed."""
    cfg = inference_cfg
    infer_config_ref = cfg.get("infer_config")
    if isinstance(infer_config_ref, str) and infer_config_ref:
        infer_cfg_path = _resolve_path(infer_config_ref, repo_root)
        resolved_cfg = load_config(infer_cfg_path)
        resolved_inf = resolved_cfg.get("inference", resolved_cfg)
        if isinstance(resolved_inf, dict):
            cfg = resolved_inf

    backbone_raw = cfg.get("backbone", "unknown")
    normalized = str(backbone_raw).strip().lower().replace("-", "_")
    aliases = {
        "showo2": "show_o2", "showo": "show_o2",
        "janus": "janus_pro", "januspro": "janus_pro",
        "omnigen": "omnigen2",
        "blip3": "blip3o", "blip3_o": "blip3o",
        "token_flow": "tokenflow",
    }
    return aliases.get(normalized, normalized)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="GenEval scoring (evaluate + summarize)")
    parser.add_argument("--config", required=True, help="Path to eval config YAML")
    args = parser.parse_args()

    geneval_cfg, inference_cfg = _load_eval_cfg(args.config)
    # eval/generation/geneval/run_scoring.py -> parents[3] = repo root
    repo_root = Path(__file__).resolve().parents[3]
    # The evaluation scripts (evaluate_images.py, summary_scores.py) live
    # in the evaluation/ subdirectory next to this file.
    eval_dir = Path(__file__).resolve().parent / "evaluation"

    backbone = _get_backbone_name(inference_cfg, repo_root)

    # --- Paths ---
    out_dir = _resolve_path(
        str(geneval_cfg.get("out_dir", f"output/geneval/{backbone}")), repo_root
    )
    images_dir = out_dir / "images"
    results_file = out_dir / "results.jsonl"
    score_output_path = geneval_cfg.get("score_output_path")

    # --- Detection eval config ---
    detection_eval_cfg = geneval_cfg.get("detection_eval", {})
    if not isinstance(detection_eval_cfg, dict):
        detection_eval_cfg = {}

    print(f"[geneval-score] backbone={backbone}, images_dir={images_dir}")

    if not images_dir.is_dir():
        print(f"[geneval-score] ERROR: images_dir not found: {images_dir}")
        print("[geneval-score] Run generation first (mode: generate or full).")
        return 1

    # --- Step 1: Object detection evaluation ---
    detection_ok = _run_object_detection_eval(
        eval_dir=eval_dir,
        images_dir=images_dir,
        results_file=results_file,
        eval_cfg=detection_eval_cfg,
    )

    if not detection_ok:
        print("[geneval-score] detection evaluation failed or skipped")
        return 1

    # --- Step 2: Summary scoring ---
    scores = _run_summary_scores(
        eval_dir=eval_dir,
        results_file=results_file,
        eval_cfg=detection_eval_cfg,
    )

    # --- Save summary ---
    summary: Dict[str, Any] = {
        "benchmark": "geneval",
        "backbone": backbone,
        "out_dir": str(out_dir),
        "images_dir": str(images_dir),
        "results_file": str(results_file),
    }
    if scores:
        summary["scores"] = scores

    summary_path = out_dir / "score_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[geneval-score] saved score summary to {summary_path}")

    if isinstance(score_output_path, str) and score_output_path:
        score_path = _resolve_path(score_output_path, repo_root)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[geneval-score] wrote score to {score_path}")

    print("[geneval-score] scoring completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
