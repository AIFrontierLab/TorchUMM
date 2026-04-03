"""
GenEval evaluation wrapper — thin orchestrator that calls subprocess scripts.

Generation: eval/generation/geneval/run_generation.py  (uses InferencePipeline)
Scoring:    eval/generation/geneval/run_scoring.py     (Mask2Former detection + summary)

Follows the same subprocess pattern as dpg_bench.py.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from umm.core.config import load_config


def run_eval_command(args: Any) -> int:
    """Entry point called by eval.py dispatcher."""
    repo_root = Path(__file__).resolve().parents[3]
    config_path = str(args.config)
    raw_cfg = load_config(config_path)

    geneval_cfg = raw_cfg.get("geneval", {})
    if not isinstance(geneval_cfg, dict):
        geneval_cfg = {}

    # mode: full / generate / score
    mode = str(geneval_cfg.get("mode", "full")).strip().lower()
    if mode not in ("full", "generate", "score"):
        print(f"[geneval] unknown mode '{mode}', defaulting to 'full'")
        mode = "full"

    # Set up PYTHONPATH so subprocess can import umm.*
    env = os.environ.copy()
    src_dir = str(repo_root / "src")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if not existing_pp else f"{src_dir}:{existing_pp}"

    geneval_dir = repo_root / "eval" / "generation" / "geneval"

    print(f"[geneval] mode={mode}")

    # --- Generation phase ---
    if mode in ("full", "generate"):
        gen_script = geneval_dir / "run_generation.py"
        if not gen_script.exists():
            raise FileNotFoundError(f"GenEval generation script not found: {gen_script}")

        cmd = [sys.executable, str(gen_script), "--config", config_path]
        print(f"[geneval] running generation: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(repo_root), env=env)
        if result.returncode != 0:
            return result.returncode

    # --- Scoring phase (Mask2Former detection + summary) ---
    if mode in ("full", "score"):
        score_script = geneval_dir / "run_scoring.py"
        if not score_script.exists():
            raise FileNotFoundError(f"GenEval scoring script not found: {score_script}")

        cmd = [sys.executable, str(score_script), "--config", config_path]
        print(f"[geneval] running scoring: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(repo_root), env=env)
        if result.returncode != 0:
            return result.returncode

    print("[geneval] completed.")
    return 0
