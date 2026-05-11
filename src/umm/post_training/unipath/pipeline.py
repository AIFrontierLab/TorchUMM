from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

COMPACT_LIST_ARGS = {"target_modules"}


def _build_args(args: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, value in args.items():
        flag = f"--{key.replace('_', '-')}"
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                out.append(flag)
        elif isinstance(value, (list, tuple)) and key in COMPACT_LIST_ARGS:
            out.append(flag)
            out.extend([str(item) for item in value])
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
        cwd_path = repo_root
    if not cwd_path.exists():
        raise FileNotFoundError(cwd_path)
    return cwd_path


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


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _has_unexpanded_env(value: Any) -> bool:
    return isinstance(value, str) and "${" in value


def _path_from_arg(value: Any) -> Path | None:
    if value is None or _has_unexpanded_env(value) or str(value).strip() == "":
        return None
    return Path(str(value)).expanduser()


def _cache_output_path(manifest_dir: Path, input_path: Path, seen: set[str]) -> Path:
    stem = input_path.stem
    candidate = manifest_dir / f"{stem}.with_latents.jsonl"
    if str(candidate) not in seen:
        seen.add(str(candidate))
        return candidate
    index = 1
    while True:
        candidate = manifest_dir / f"{stem}.{index}.with_latents.jsonl"
        if str(candidate) not in seen:
            seen.add(str(candidate))
            return candidate
        index += 1


def _maybe_generate_latent_cache(
    *,
    module: str,
    args_dict: dict[str, Any],
    cfg: dict[str, Any],
    cwd_path: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    if not module.startswith("umm.post_training.unipath.") or module.endswith(".cache_latents"):
        return args_dict
    auto_generate = bool(args_dict.pop("auto_generate_latent_cache", cfg.get("auto_generate_latent_cache", True)))
    max_rows_raw = args_dict.pop("cache_latents_max_rows", cfg.get("cache_latents_max_rows", None))
    overwrite = bool(args_dict.pop("cache_latents_overwrite", cfg.get("cache_latents_overwrite", False)))
    progress_every = int(args_dict.pop("cache_latents_progress_every", cfg.get("cache_latents_progress_every", 50)))
    manifest_dir_raw = args_dict.pop("latent_manifest_dir", cfg.get("latent_manifest_dir", None))
    if not auto_generate:
        return args_dict

    jsonl_keys = ("train_jsonl", "val_jsonl")
    input_paths = [Path(path).expanduser() for key in jsonl_keys for path in _as_list(args_dict.get(key))]
    existing_inputs = [path for path in input_paths if path.exists()]
    if not existing_inputs:
        return args_dict

    cache_dir = _path_from_arg(args_dict.get("latent_cache_root"))
    if cache_dir is None:
        output_dir = _path_from_arg(args_dict.get("output_dir"))
        if output_dir is None:
            return args_dict
        cache_dir = output_dir / "latent_cache" / "cache"
        args_dict["latent_cache_root"] = str(cache_dir)

    max_rows = int(max_rows_raw) if max_rows_raw is not None else None
    from umm.post_training.unipath.cache_latents import jsonl_has_latent_references, jsonl_needs_latent_cache

    needs_cache: list[Path] = []
    for path in existing_inputs:
        if jsonl_has_latent_references(path, max_rows=max_rows) and jsonl_needs_latent_cache(path, cache_dir, max_rows=max_rows):
            needs_cache.append(path)
    if not needs_cache:
        return args_dict

    if manifest_dir_raw is not None and not _has_unexpanded_env(manifest_dir_raw):
        manifest_dir = Path(str(manifest_dir_raw)).expanduser()
    else:
        output_dir = _path_from_arg(args_dict.get("output_dir")) or cwd_path
        manifest_dir = output_dir / "latent_cache_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    replacement_by_input: dict[str, str] = {}
    seen_outputs: set[str] = set()
    cmd = [sys.executable, "-m", "umm.post_training.unipath.cache_latents"]
    for input_path in needs_cache:
        output_path = _cache_output_path(manifest_dir, input_path, seen_outputs)
        replacement_by_input[str(input_path)] = str(output_path)
        cmd.extend(["--input-jsonl", str(input_path), "--output-jsonl", str(output_path)])
    model_path = args_dict.get("model_path")
    if not model_path or _has_unexpanded_env(model_path):
        raise ValueError("Auto latent caching requires a resolved `model_path`.")
    cmd.extend(["--model-path", str(model_path), "--cache-dir", str(cache_dir), "--progress-every", str(progress_every)])
    if max_rows is not None:
        cmd.extend(["--max-rows", str(max_rows)])
    if overwrite:
        cmd.append("--overwrite")

    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    print(f"[umm train] auto latent cache: {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd_path), env=env, check=True)

    updated_args = dict(args_dict)
    for key in jsonl_keys:
        original = _as_list(args_dict.get(key))
        if not original:
            continue
        replaced = [replacement_by_input.get(str(Path(path).expanduser()), str(path)) for path in original]
        updated_args[key] = replaced
    return updated_args


def run_unipath_train(cfg: dict[str, Any], config_path: str | None = None) -> None:
    cwd_path = _resolve_cwd(config_path, cfg.get("cwd"))
    module = cfg.get("module", "umm.post_training.unipath.train")
    if not isinstance(module, str) or not module:
        raise ValueError("Training config requires non-empty `module`.")

    torchrun_cfg = cfg.get("torchrun", {}) or {}
    if not isinstance(torchrun_cfg, dict):
        raise ValueError("`torchrun` must be a dict if provided.")
    use_torchrun = bool(cfg.get("use_torchrun", False))
    args_dict = cfg.get("args", {}) or {}
    if not isinstance(args_dict, dict):
        raise ValueError("`args` must be a dict if provided.")
    args_dict = dict(args_dict)

    extra_args = cfg.get("extra_args", [])

    env = os.environ.copy()
    env_update = cfg.get("env", {})
    if isinstance(env_update, dict):
        for key, value in env_update.items():
            if value is not None:
                env[str(key)] = str(value)

    args_dict = _maybe_generate_latent_cache(module=module, args_dict=args_dict, cfg=cfg, cwd_path=cwd_path, env=env)

    cmd = _build_launcher_prefix(torchrun_cfg=torchrun_cfg, use_torchrun=use_torchrun)
    cmd.extend(["-m", module])
    cmd.extend(_build_args(args_dict))
    if isinstance(extra_args, list):
        cmd.extend([str(x) for x in extra_args])

    print(f"[umm train] cwd: {cwd_path}")
    print(f"[umm train] running: {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd_path), env=env, check=True)
