"""Thin CLI shim for Uni-MMMU evaluation.

Follows the DPG-Bench subprocess pattern: launches run_generation.py
and/or run_scoring.py as subprocesses depending on the ``mode`` setting
in the config file.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from umm.core.config import load_config


def _resolve_config_path(config_path: str, repo_root: Path) -> Path:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def run_eval_command(args: Any) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    config_path = _resolve_config_path(str(args.config), repo_root)
    raw_cfg = load_config(config_path)

    uni_mmmu_cfg = raw_cfg.get("uni_mmmu", {})
    if not isinstance(uni_mmmu_cfg, dict):
        uni_mmmu_cfg = {}

    mode = str(uni_mmmu_cfg.get("mode", "generate")).strip().lower()
    if mode not in ("generate", "score", "full"):
        raise ValueError(
            f"`uni_mmmu.mode` must be 'generate', 'score', or 'full', got '{mode}'."
        )

    env = os.environ.copy()
    src_dir = str(repo_root / "src")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if not existing_pp else f"{src_dir}:{existing_pp}"

    run_generate = mode in ("generate", "full")
    run_score = mode in ("score", "full")

    # ── Generation phase ──
    if run_generate:
        gen_script = (
            repo_root / "eval" / "generation" / "uni_mmmu" / "run_generation.py"
        )
        if not gen_script.exists():
            raise FileNotFoundError(
                f"Uni-MMMU generation runner not found: {gen_script}"
            )
        cmd = [sys.executable, str(gen_script), "--config", str(config_path)]
        print(f"[umm eval] running Uni-MMMU generation")
        print(f"[umm eval] command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=repo_root, env=env)
        if result.returncode != 0:
            return int(result.returncode)

    # ── Scoring phase ──
    if run_score:
        score_script = (
            repo_root / "eval" / "generation" / "uni_mmmu" / "run_scoring.py"
        )
        if not score_script.exists():
            raise FileNotFoundError(
                f"Uni-MMMU scoring runner not found: {score_script}"
            )

        scoring_cfg = uni_mmmu_cfg.get("scoring", {})
        if not isinstance(scoring_cfg, dict):
            scoring_cfg = {}

        score_python = scoring_cfg.get("python_executable")
        py = str(Path(score_python).expanduser()) if score_python else sys.executable

        cmd = [py, str(score_script), "--config", str(config_path)]
        print(f"[umm eval] running Uni-MMMU scoring")
        if score_python:
            print(f"[umm eval] using scoring python: {py}")
        print(f"[umm eval] command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=repo_root, env=env)
        if result.returncode != 0:
            return int(result.returncode)

    return 0
