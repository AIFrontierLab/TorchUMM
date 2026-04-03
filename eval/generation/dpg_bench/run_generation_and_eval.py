#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image
import torch
import torch.distributed as dist

from umm.core.config import load_config
from umm.inference import InferencePipeline


def _unwrap_inference_block(config: dict[str, Any]) -> dict[str, Any]:
    block = config.get("inference")
    if isinstance(block, dict):
        return block
    return config


def _unwrap_dpg_block(config: dict[str, Any]) -> dict[str, Any]:
    block = config.get("dpg_bench")
    if isinstance(block, dict):
        return block
    return config


def _resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _setup_distributed() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed run requested, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    return rank, world_size


def _teardown_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _sort_prompt_files(prompt_files: list[Path]) -> list[Path]:
    def _key(path: Path) -> tuple[int, str]:
        stem = path.stem
        if stem.isdigit():
            return (0, f"{int(stem):08d}")
        return (1, stem)

    return sorted(prompt_files, key=_key)


def _extract_images(result: Any) -> list[Image.Image]:
    def _safe_image_copy(value: Any) -> Image.Image | None:
        try:
            if isinstance(value, Image.Image):
                value.load()
                return value.convert("RGB").copy()
            if isinstance(value, str):
                with Image.open(value) as img:
                    return img.convert("RGB").copy()
        except Exception:
            return None
        return None

    if isinstance(result, Image.Image):
        image = _safe_image_copy(result)
        return [image] if image is not None else []

    if isinstance(result, list):
        images: list[Image.Image] = []
        for item in result:
            image = _safe_image_copy(item)
            if image is not None:
                images.append(image)
        return images

    if isinstance(result, dict):
        images: list[Image.Image] = []
        # support adapters that return file paths (strings) for images
        if isinstance(result.get("images"), list):
            for item in result["images"]:
                image = _safe_image_copy(item)
                if image is not None:
                    images.append(image)
        if isinstance(result.get("saved_paths"), list):
            for item in result["saved_paths"]:
                image = _safe_image_copy(item)
                if image is not None:
                    images.append(image)
        image = result.get("image")
        copied = _safe_image_copy(image)
        if copied is not None:
            images.append(copied)
        return images

    return []


def _make_2x2_grid(images: list[Image.Image]) -> Image.Image:
    if len(images) != 4:
        raise ValueError(f"Expected 4 images for grid, got {len(images)}.")

    width, height = images[0].size
    grid = Image.new("RGB", (width * 2, height * 2))
    grid.paste(images[0], (0, 0))
    grid.paste(images[1], (width, 0))
    grid.paste(images[2], (0, height))
    grid.paste(images[3], (width, height))
    return grid


def run_generation(
    infer_cfg: dict[str, Any],
    prompts_dir: Path,
    image_output_dir: Path,
    images_per_prompt: int = 4,
    overwrite: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    cfg = _unwrap_inference_block(infer_cfg)

    backbone = cfg.get("backbone")
    if not isinstance(backbone, str) or not backbone:
        raise ValueError("Inference config must define non-empty `backbone`.")
    backbone_cfg = cfg.get("backbone_cfg", {})
    if not isinstance(backbone_cfg, dict):
        raise ValueError("`backbone_cfg` must be a dict.")

    bagel_multiprocess_enabled = bool(backbone_cfg.get("distributed_single_gpu", False))
    if backbone == "bagel" and world_size > 1 and not bagel_multiprocess_enabled:
        raise ValueError(
            "Backbone `bagel` already uses model-parallel across GPUs internally. "
            "Set `dpg_bench.gpus: 1`, or enable multiprocess mode with "
            "`inference.backbone_cfg.distributed_single_gpu: true`."
        )

    request = cfg.get("request", {})
    if not isinstance(request, dict):
        raise ValueError("Inference config `request` must be a dict.")

    generation_params = dict(request.get("params", {}))

    pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)

    image_output_dir.mkdir(parents=True, exist_ok=True)
    all_prompt_files = _sort_prompt_files(list(prompts_dir.glob("*.txt")))
    if not all_prompt_files:
        raise FileNotFoundError(f"No prompt files found in {prompts_dir}")
    prompt_files = [p for i, p in enumerate(all_prompt_files) if i % world_size == rank]
    print(f"[rank {rank}] assigned {len(prompt_files)} / {len(all_prompt_files)} prompts")

    for idx, prompt_file in enumerate(prompt_files, start=1):
        prompt_id = prompt_file.stem
        output_path = image_output_dir / f"{prompt_id}.png"
        if output_path.exists() and not overwrite:
            print(f"[rank {rank}] [skip] {prompt_id} ({idx}/{len(prompt_files)}) exists")
            continue

        prompt = prompt_file.read_text(encoding="utf-8").strip()
        if not prompt:
            print(f"[rank {rank}] [skip] {prompt_id} ({idx}/{len(prompt_files)}) empty prompt")
            continue

        print(f"[rank {rank}] [gen ] {prompt_id} ({idx}/{len(prompt_files)})")
        collected: list[Image.Image] = []
        while len(collected) < images_per_prompt:
            payload = {
                "backbone": backbone,
                "task": "generation",
                "prompt": prompt,
                "params": generation_params,
            }
            result = pipeline.run(payload)
            extracted = _extract_images(result)
            collected.extend(extracted)
            if not extracted:
                if isinstance(result, dict):
                    returncode = result.get("returncode")
                    stderr = result.get("stderr")
                    stdout = result.get("stdout")
                    if returncode is not None and returncode != 0:
                        raise RuntimeError(
                            f"Backbone `{backbone}` generation failed for prompt `{prompt_id}` "
                            f"(returncode={returncode}). stderr={stderr!r}"
                        )
                    raise RuntimeError(
                        f"Backbone `{backbone}` generation returned no images for prompt `{prompt_id}`. "
                        f"stdout={stdout!r} stderr={stderr!r}"
                    )
                raise RuntimeError(
                    f"Backbone `{backbone}` generation result did not include PIL images for prompt `{prompt_id}`."
                )

        grid = _make_2x2_grid(collected[:images_per_prompt])
        grid.save(output_path)

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print(f"[done] generated grids at {image_output_dir}")


def run_eval(
    image_output_dir: Path,
    resolution: int,
    repo_root: Path,
    eval_port: int = 0,
    eval_python: str | None = None,
    eval_env_bin: str | None = None,
) -> None:
    eval_root = repo_root / "eval" / "generation"
    env = os.environ.copy()
    env["PORT"] = str(eval_port if eval_port > 0 else _pick_free_port())
    env["MAIN_PROCESS_IP"] = "127.0.0.1"

    if eval_env_bin:
        env_bin = str(Path(eval_env_bin).expanduser())
        env["PATH"] = f"{env_bin}:{env.get('PATH', '')}"
        print(f"[eval] prepended PATH with eval_env_bin={env_bin}")

    if eval_python:
        py_exec = str(Path(eval_python).expanduser())
        cmd = [
            py_exec,
            "dpg_bench/compute_dpg_bench.py",
            "--image-root-path",
            str(image_output_dir),
            "--resolution",
            str(resolution),
            "--pic-num",
            "4",
            "--vqa-model",
            "mplug",
        ]
    else:
        cmd = [
            "bash",
            "dpg_bench/dist_eval.sh",
            str(image_output_dir),
            str(resolution),
        ]

    print(f"[eval] running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=eval_root, env=env, check=True)


def _find_latest_result_file(image_output_dir: Path) -> Path | None:
    candidates = sorted(image_output_dir.glob("dpg-bench_*_results.txt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def _extract_final_score(result_file: Path) -> float | None:
    for line in result_file.read_text(encoding="utf-8").splitlines():
        prefix = "DPG-Bench score:"
        if line.startswith(prefix):
            raw = line[len(prefix) :].strip()
            try:
                return float(raw)
            except ValueError:
                return None
    return None


def save_score_summary(image_output_dir: Path, score_output_path: str | None = None) -> None:
    result_file = _find_latest_result_file(image_output_dir)
    if result_file is None:
        print(f"[eval] no DPG result file found under {image_output_dir}; skip summary save")
        return

    score = _extract_final_score(result_file)
    payload: dict[str, Any] = {
        "result_file": str(result_file),
        "final_score": score,
    }

    target = Path(score_output_path).expanduser() if score_output_path else (image_output_dir / "final_score.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[eval] saved final score summary to {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DPG-Bench 2x2 grids (4 images/prompt) with any UMM backbone and run evaluation."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Single YAML/JSON run config containing `inference` and `dpg_bench` blocks.",
    )
    parser.add_argument(
        "--infer-config",
        default=None,
        help="Path to UMM inference config YAML/JSON used for generation backbone and params.",
    )
    parser.add_argument(
        "--prompts-dir",
        default=None,
        help="Directory containing DPG-Bench prompt .txt files.",
    )
    parser.add_argument(
        "--image-output-dir",
        default=None,
        help="Directory to save generated grid images named <prompt_id>.png.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Per-image resolution used by dpg_bench evaluation crop logic.",
    )
    parser.add_argument(
        "--images-per-prompt",
        type=int,
        default=None,
        help="Number of images to collect per prompt before gridding (must be 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate grids even if output files already exist.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only generate images; skip running dist_eval.sh.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    rank, world_size = _setup_distributed()
    skip_eval = False
    image_output_dir: Path | None = None
    resolution = 512
    eval_port = 0
    eval_python: str | None = None
    eval_env_bin: str | None = None
    score_output_path: str | None = None

    try:
        if args.config:
            run_cfg_path = _resolve_path(args.config, repo_root)
            raw_run_cfg = load_config(run_cfg_path)
            run_cfg = _unwrap_dpg_block(raw_run_cfg)
        else:
            raw_run_cfg = {}
            run_cfg = {}

        prompts_dir_value = args.prompts_dir or run_cfg.get("prompts_dir") or "eval/generation/dpg_bench/prompts"
        image_output_dir_value = args.image_output_dir or run_cfg.get("image_output_dir")
        resolution = int(args.resolution or run_cfg.get("resolution", 512))
        eval_port = int(run_cfg.get("eval_port", 0))
        eval_python = run_cfg.get("eval_python")
        eval_env_bin = run_cfg.get("eval_env_bin")
        score_output_path = run_cfg.get("score_output_path")
        images_per_prompt = int(args.images_per_prompt or run_cfg.get("images_per_prompt", 4))
        overwrite = bool(args.overwrite or run_cfg.get("overwrite", False))
        skip_eval = bool(args.skip_eval or run_cfg.get("skip_eval", False))

        if images_per_prompt != 4:
            raise ValueError("This script currently supports only 4 images per prompt for 2x2 grids.")
        if not image_output_dir_value:
            raise ValueError("`image_output_dir` is required (via CLI or run config).")

        prompts_dir = _resolve_path(str(prompts_dir_value), repo_root)
        image_output_dir = _resolve_path(str(image_output_dir_value), repo_root)

        if "inference" in raw_run_cfg and isinstance(raw_run_cfg["inference"], dict):
            inf_block = raw_run_cfg["inference"]
            # support a short-form that references another inference config file:
            # inference:
            #   infer_config: configs/inference/show_o_generation.yaml
            if isinstance(inf_block.get("infer_config"), (str,)) and inf_block.get("infer_config"):
                infer_cfg_path = _resolve_path(str(inf_block.get("infer_config")), repo_root)
                infer_cfg = load_config(infer_cfg_path)
            else:
                infer_cfg = raw_run_cfg
        else:
            infer_config_value = args.infer_config or run_cfg.get("infer_config")
            if not infer_config_value:
                raise ValueError(
                    "Provide `inference` block in run config, or set `infer_config` (run config / --infer-config)."
                )
            infer_cfg_path = _resolve_path(str(infer_config_value), repo_root)
            infer_cfg = load_config(infer_cfg_path)

        run_generation(
            infer_cfg=infer_cfg,
            prompts_dir=prompts_dir,
            image_output_dir=image_output_dir,
            images_per_prompt=images_per_prompt,
            overwrite=overwrite,
            rank=rank,
            world_size=world_size,
        )
    finally:
        _teardown_distributed()

    if not skip_eval and rank == 0:
        if image_output_dir is None:
            raise RuntimeError("Internal error: image_output_dir was not resolved.")
        run_eval(
            image_output_dir=image_output_dir,
            resolution=resolution,
            repo_root=repo_root,
            eval_port=eval_port,
            eval_python=eval_python,
            eval_env_bin=eval_env_bin,
        )
        save_score_summary(image_output_dir=image_output_dir, score_output_path=score_output_path)


if __name__ == "__main__":
    main()
