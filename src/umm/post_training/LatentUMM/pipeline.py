from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


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
        cwd_path = repo_root
    if not cwd_path.exists():
        raise FileNotFoundError(cwd_path)
    return cwd_path


def _format_flag(key: str, flag_style: str) -> str:
    if flag_style == "underscore":
        return f"--{key.replace('-', '_')}"
    if flag_style == "raw":
        return f"--{key}"
    return f"--{key.replace('_', '-')}"


def _build_args(args: dict[str, Any], *, flag_style: str, bool_style: str) -> list[str]:
    out: list[str] = []
    for key, value in args.items():
        if value is None:
            continue
        flag = _format_flag(str(key), flag_style)
        if isinstance(value, bool):
            if bool_style == "value":
                out.extend([flag, "True" if value else "False"])
            elif value:
                out.append(flag)
        elif isinstance(value, (list, tuple)):
            for item in value:
                out.extend([flag, str(item)])
        else:
            out.extend([flag, str(value)])
    return out


def _build_launcher_prefix(torchrun_cfg: dict[str, Any], use_torchrun: bool) -> list[str]:
    if not use_torchrun:
        return [sys.executable]

    cmd: list[str] = [sys.executable, "-m", "torch.distributed.run"]
    defaults = {
        "nnodes": 1,
        "node_rank": 0,
        "nproc_per_node": 1,
    }
    for key, default in defaults.items():
        cmd.append(f"--{key}={torchrun_cfg.get(key, default)}")
    for key in ("master_addr", "master_port", "rdzv_backend", "rdzv_endpoint"):
        value = torchrun_cfg.get(key)
        if value is not None:
            cmd.append(f"--{key}={value}")
    extra_torchrun = torchrun_cfg.get("extra_args", [])
    if isinstance(extra_torchrun, list):
        cmd.extend([str(x) for x in extra_torchrun])
    return cmd


def run_latentumm_train(cfg: dict[str, Any], config_path: str | None = None) -> None:
    module = cfg.get("module")
    if not isinstance(module, str) or not module:
        raise ValueError("LatentUMM training config requires non-empty `module`.")

    cwd_path = _resolve_cwd(config_path, cfg.get("cwd"))
    torchrun_cfg = cfg.get("torchrun", {}) or {}
    if not isinstance(torchrun_cfg, dict):
        raise ValueError("`torchrun` must be a dict if provided.")

    use_torchrun = bool(cfg.get("use_torchrun", False))
    flag_style = str(cfg.get("flag_style", "hyphen"))
    bool_style = str(cfg.get("bool_style", "flag"))
    args_dict = cfg.get("args", {}) or {}
    if not isinstance(args_dict, dict):
        raise ValueError("`args` must be a dict if provided.")

    env = os.environ.copy()
    env_update = cfg.get("env", {})
    if isinstance(env_update, dict):
        for key, value in env_update.items():
            if value is not None:
                env[str(key)] = str(value)

    cmd = _build_launcher_prefix(torchrun_cfg=torchrun_cfg, use_torchrun=use_torchrun)
    cmd.extend(["-m", module])
    cmd.extend(_build_args(dict(args_dict), flag_style=flag_style, bool_style=bool_style))

    extra_args = cfg.get("extra_args", [])
    if isinstance(extra_args, list):
        cmd.extend([str(x) for x in extra_args])

    print(f"[umm train] cwd: {cwd_path}")
    print(f"[umm train] running: {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd_path), env=env, check=True)
