"""
Unified-Bench evaluation wrapper — thin orchestrator that calls subprocess scripts.

Unified-Bench (from PKU-YuanGroup's UAE paper) measures the round-trip fidelity of
unified multimodal models: each backbone captions a reference image via its own I2T
module, then regenerates an image from that caption via its T2I module. Similarity
between the reference image and the regenerated image is scored with CLIP, DINOv2,
DINOv3, and LongCLIP.

Generation: eval/generation/unified_bench/run_generation.py (uses InferencePipeline)
Scoring:    eval/generation/unified_bench/run_scoring.py    (CLIP/DINOv2/v3/LongCLIP)

Follows the same subprocess pattern as ueval_eval.py.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from umm.core.config import load_config


def run_unified_bench_eval_command(args: Any) -> int:
    """Entry point called by eval.py dispatcher."""
    config_path = str(args.config)
    raw_cfg = load_config(config_path)
    bench_cfg = raw_cfg.get("unified_bench", {})
    if not isinstance(bench_cfg, dict):
        bench_cfg = {}

    mode = str(bench_cfg.get("mode", "full")).strip().lower()
    if mode not in ("full", "generate", "score"):
        print(f"[unified_bench] unknown mode '{mode}', defaulting to 'full'")
        mode = "full"

    repo_root = Path(__file__).resolve().parents[3]

    env = os.environ.copy()
    src_dir = str(repo_root / "src")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if not existing_pp else f"{src_dir}:{existing_pp}"

    print(f"[unified_bench] mode={mode}")

    if mode in ("full", "generate"):
        gen_script = repo_root / "eval" / "generation" / "unified_bench" / "run_generation.py"
        if not gen_script.exists():
            raise FileNotFoundError(f"Unified-Bench generation script not found: {gen_script}")
        cmd = [sys.executable, str(gen_script), "--config", config_path]
        print(f"[unified_bench] running generation: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(repo_root), env=env)
        if result.returncode != 0:
            return result.returncode

    if mode in ("full", "score"):
        score_script = repo_root / "eval" / "generation" / "unified_bench" / "run_scoring.py"
        if not score_script.exists():
            raise FileNotFoundError(f"Unified-Bench scoring script not found: {score_script}")
        cmd = [sys.executable, str(score_script), "--config", config_path]
        print(f"[unified_bench] running scoring: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(repo_root), env=env)
        if result.returncode != 0:
            return result.returncode

    print("[unified_bench] completed.")
    return 0
