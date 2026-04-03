"""
UEval evaluation wrapper — thin orchestrator that calls subprocess scripts.

Generation: eval/generation/ueval/run_generation.py (uses InferencePipeline)
Scoring:    eval/generation/ueval/run_scoring.py    (Qwen3-32B + Qwen2.5-VL-72B)

Follows the same subprocess pattern as dpg_bench.py.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from umm.core.config import load_config


def run_ueval_eval_command(args: Any) -> int:
    """Entry point called by eval.py dispatcher."""
    config_path = str(args.config)
    raw_cfg = load_config(config_path)
    ueval_cfg = raw_cfg.get("ueval", {})
    if not isinstance(ueval_cfg, dict):
        ueval_cfg = {}

    # mode: full / generate / score (backward compat: score_only -> score)
    if ueval_cfg.get("score_only"):
        mode = "score"
    else:
        mode = str(ueval_cfg.get("mode", "full")).strip().lower()
    if mode not in ("full", "generate", "score"):
        print(f"[ueval] unknown mode '{mode}', defaulting to 'full'")
        mode = "full"

    repo_root = Path(__file__).resolve().parents[3]

    # Set up PYTHONPATH so subprocess can import umm.*
    env = os.environ.copy()
    src_dir = str(repo_root / "src")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if not existing_pp else f"{src_dir}:{existing_pp}"

    print(f"[ueval] mode={mode}")

    # --- Generation phase ---
    if mode in ("full", "generate"):
        gen_script = repo_root / "eval" / "generation" / "ueval" / "run_generation.py"
        if not gen_script.exists():
            raise FileNotFoundError(f"UEval generation script not found: {gen_script}")

        cmd = [sys.executable, str(gen_script), "--config", config_path]
        print(f"[ueval] running generation: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(repo_root), env=env)
        if result.returncode != 0:
            return result.returncode

    # --- Scoring phase (Qwen-based local scoring) ---
    if mode in ("full", "score"):
        score_script = repo_root / "eval" / "generation" / "ueval" / "run_scoring.py"
        if not score_script.exists():
            raise FileNotFoundError(f"UEval scoring script not found: {score_script}")

        cmd = [sys.executable, str(score_script), "--config", config_path]
        print(f"[ueval] running scoring: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(repo_root), env=env)
        if result.returncode != 0:
            return result.returncode

    print("[ueval] completed.")
    return 0
