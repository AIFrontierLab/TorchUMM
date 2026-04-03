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


def _resolve_cuda_visible_devices(gpus: int, dpg_cfg: dict[str, Any]) -> str:
    explicit = dpg_cfg.get("cuda_visible_devices")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    gpu_ids = dpg_cfg.get("gpu_ids")
    if isinstance(gpu_ids, list) and gpu_ids:
        return ",".join(str(int(i)) for i in gpu_ids)

    return ",".join(str(i) for i in range(gpus))


def _resolve_backbone(raw_cfg: dict[str, Any], repo_root: Path) -> str | None:
    inference_cfg = raw_cfg.get("inference")
    if not isinstance(inference_cfg, dict):
        return None

    backbone_value = inference_cfg.get("backbone")
    if isinstance(backbone_value, str) and backbone_value:
        return backbone_value

    infer_config_value = inference_cfg.get("infer_config")
    if isinstance(infer_config_value, str) and infer_config_value:
        infer_cfg_path = _resolve_config_path(infer_config_value, repo_root)
        nested_cfg = load_config(infer_cfg_path)
        nested_inf = nested_cfg.get("inference")
        if isinstance(nested_inf, dict):
            nested_backbone = nested_inf.get("backbone")
            if isinstance(nested_backbone, str) and nested_backbone:
                return nested_backbone
        nested_backbone = nested_cfg.get("backbone")
        if isinstance(nested_backbone, str) and nested_backbone:
            return nested_backbone
    return None


def run_eval_command(args: Any) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    config_path = _resolve_config_path(str(args.config), repo_root)
    raw_cfg = load_config(config_path)

    dpg_cfg = raw_cfg.get("dpg_bench", {})
    if not isinstance(dpg_cfg, dict):
        dpg_cfg = {}
    gpus = int(dpg_cfg.get("gpus", 1))
    if gpus < 1:
        raise ValueError("`dpg_bench.gpus` must be >= 1.")
    bagel_multiprocess = bool(dpg_cfg.get("bagel_multiprocess", False))
    backbone = _resolve_backbone(raw_cfg=raw_cfg, repo_root=repo_root)

    script_path = repo_root / "eval" / "generation" / "dpg_bench" / "run_generation_and_eval.py"
    if not script_path.exists():
        raise FileNotFoundError(f"DPG-Bench runner not found: {script_path}")

    use_torchrun = gpus > 1 and (backbone != "bagel" or bagel_multiprocess)
    if use_torchrun:
        cmd = [
            "torchrun",
            f"--nproc_per_node={gpus}",
            str(script_path),
            "--config",
            str(config_path),
        ]
    else:
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
    if backbone == "bagel" and gpus > 1 and not use_torchrun:
        env["CUDA_VISIBLE_DEVICES"] = _resolve_cuda_visible_devices(gpus=gpus, dpg_cfg=dpg_cfg)

    print(f"[umm eval] running DPG-Bench with {gpus} GPU(s)")
    if backbone == "bagel" and gpus > 1 and not use_torchrun:
        print("[umm eval] backbone=bagel -> using single process with model-parallel across visible GPUs")
        print(f"[umm eval] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    if backbone == "bagel" and use_torchrun:
        print("[umm eval] backbone=bagel -> using torchrun multiprocess mode (one process per GPU)")
    print(f"[umm eval] command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=repo_root, env=env)
    return int(result.returncode)
