"""CLI entry point for ImgEdit-Bench — thin subprocess wrapper.

Follows the same pattern as cli/gedit.py:
reads config, builds subprocess command, calls eval/edit/imgedit/run_imgedit_eval.py.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from umm.core.config import load_config


def _resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def _normalize_backbone_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "showo2": "show_o2",
        "showo": "show_o2",
        "janus": "janus_pro",
    }
    return aliases.get(normalized, normalized)


def run_imgedit_eval_command(args: Any) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    config_path = _resolve_path(str(args.config), repo_root)
    raw_cfg = load_config(config_path)

    # Read backbone for logging
    inference_cfg = raw_cfg.get("inference", {})
    if isinstance(inference_cfg, dict):
        backbone = _normalize_backbone_name(str(inference_cfg.get("backbone", "")))
    else:
        backbone = "unknown"

    script_path = repo_root / "eval" / "edit" / "imgedit" / "run_imgedit_eval.py"
    if not script_path.exists():
        raise FileNotFoundError(f"ImgEdit runner not found: {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
    ]

    env = os.environ.copy()
    src_dir = str(repo_root / "src")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if not existing_pp else f"{src_dir}:{existing_pp}"

    print(f"[umm eval] running ImgEdit-Bench, backbone={backbone}")
    print(f"[umm eval] command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=repo_root, env=env)
    return int(result.returncode)
