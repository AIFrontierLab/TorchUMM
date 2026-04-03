from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


def _build_args(args: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in args.items():
        flag = f"--{key}"
        if value is None:
            continue
        if isinstance(value, bool):
            out.extend([flag, "True" if value else "False"])
        elif isinstance(value, (list, tuple)):
            for item in value:
                out.extend([flag, str(item)])
        else:
            out.extend([flag, str(value)])
    return out


def _find_repo_root(start: Path) -> Path | None:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def _resolve_cwd(config_path: str | None, cwd: str | None) -> Path:
    base_dir = Path(config_path).resolve().parent if config_path else Path.cwd()
    repo_root = _find_repo_root(base_dir) or base_dir
    if cwd:
        cwd_path = Path(cwd)
        if not cwd_path.is_absolute():
            candidate = (base_dir / cwd_path).resolve()
            if candidate.exists():
                cwd_path = candidate
            else:
                cwd_path = (repo_root / cwd_path).resolve()
    else:
        cwd_path = base_dir
    if not cwd_path.exists():
        raise FileNotFoundError(cwd_path)
    return cwd_path


def run_irg_train(cfg: dict[str, Any], config_path: str | None = None) -> None:
    script = cfg.get("script")
    if not isinstance(script, str) or not script:
        raise ValueError("Training config requires non-empty `script`.")

    cwd_path = _resolve_cwd(config_path, cfg.get("cwd"))

    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = (cwd_path / script_path).resolve()

    entrypoint = cfg.get("entrypoint", "bash")
    if entrypoint not in ("bash", "python", "torchrun"):
        raise ValueError("`entrypoint` must be one of: bash, python, torchrun")

    cmd: list[str]
    if entrypoint == "bash":
        cmd = ["bash", str(script_path)]
    elif entrypoint == "python":
        cmd = ["python", str(script_path)]
    else:
        torchrun_cfg = cfg.get("torchrun", {}) or {}
        if not isinstance(torchrun_cfg, dict):
            raise ValueError("`torchrun` must be a dict if provided.")
        cmd = ["torchrun"]
        defaults = {
            "nnodes": 1,
            "node_rank": 0,
            "nproc_per_node": 1,
        }
        for key, default in defaults.items():
            value = torchrun_cfg.get(key, default)
            cmd.append(f"--{key}={value}")
        for key in ("master_addr", "master_port", "rdzv_backend", "rdzv_endpoint"):
            value = torchrun_cfg.get(key)
            if value is not None:
                cmd.append(f"--{key}={value}")
        extra_torchrun = torchrun_cfg.get("extra_args", [])
        if isinstance(extra_torchrun, list):
            cmd.extend([str(x) for x in extra_torchrun])
        cmd.append(str(script_path))

    args_dict = cfg.get("args", {})
    if args_dict is None:
        args_dict = {}
    if not isinstance(args_dict, dict):
        raise ValueError("`args` must be a dict if provided.")
    cmd.extend(_build_args(args_dict))

    extra_args = cfg.get("extra_args", [])
    if isinstance(extra_args, list):
        cmd.extend([str(x) for x in extra_args])

    env = os.environ.copy()
    env_update = cfg.get("env", {})
    if isinstance(env_update, dict):
        for key, value in env_update.items():
            if value is not None:
                env[str(key)] = str(value)

    print(f"[umm train] running: {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd_path), env=env, check=True)
