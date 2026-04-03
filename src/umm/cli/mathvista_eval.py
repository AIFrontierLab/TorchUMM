from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image

from umm.core.config import load_config
from umm.inference import InferencePipeline


DS_COLLECTIONS = {
    "MathVista_testmini": {
        "root": "AI4Math/MathVista",
        "split": "testmini",
        "max_new_tokens": 4096,
    },
    "MathVista_test": {
        "root": "AI4Math/MathVista",
        "split": "test",
        "max_new_tokens": 4096,
    },
}


COT_INSTRUCTION = (
    "Your task is to answer the question below. "
    "Give step by step reasoning before you answer, and when you're ready to answer, "
    "please use the format \"Final answer: ..\""
    "\n\n"
    "Question:"
    "\n\n"
    "{question}"
)


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


def _extract_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("text", "answer", "response", "output", "generated_text"):
            value = output.get(key)
            if isinstance(value, str):
                return value
        results = output.get("results")
        if isinstance(results, dict):
            for key in ("text", "answer", "response", "output"):
                value = results.get(key)
                if isinstance(value, str):
                    return value
        if isinstance(results, list):
            for item in results:
                text = _extract_text(item)
                if text:
                    return text
    if isinstance(output, list):
        for item in output:
            text = _extract_text(item)
            if text:
                return text
    return ""


def _load_eval_cfg(config_path: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_cfg = load_config(config_path)
    eval_cfg = raw_cfg.get("eval", {}) if isinstance(raw_cfg.get("eval"), dict) else {}
    mathvista_cfg = raw_cfg.get("mathvista", {}) if isinstance(raw_cfg.get("mathvista"), dict) else {}
    inference_cfg = raw_cfg.get("inference", {}) if isinstance(raw_cfg.get("inference"), dict) else {}
    if not eval_cfg and "benchmark" in raw_cfg:
        eval_cfg = {"benchmark": raw_cfg.get("benchmark")}
    return eval_cfg, mathvista_cfg, inference_cfg


def run_mathvista_eval_command(args: Any) -> int:
    config_path = str(args.config)
    eval_cfg, mathvista_cfg, inference_cfg = _load_eval_cfg(config_path)
    benchmark = str(eval_cfg.get("benchmark", "")).strip().lower()
    if benchmark != "mathvista":
        raise ValueError(f"Expected `eval.benchmark: mathvista`, got: {benchmark or '<empty>'}")

    repo_root = Path(__file__).resolve().parents[3]

    backbone_raw = inference_cfg.get("backbone")
    if not isinstance(backbone_raw, str) or not backbone_raw:
        raise ValueError("`inference.backbone` is required for MathVista eval.")
    backbone = _normalize_backbone_name(backbone_raw)

    backbone_cfg = inference_cfg.get("backbone_cfg", {})
    if not isinstance(backbone_cfg, dict):
        raise ValueError("`inference.backbone_cfg` must be a dict when provided.")

    request_cfg = inference_cfg.get("request", {})
    request_params: dict[str, Any] = {}
    if isinstance(request_cfg, dict):
        params = request_cfg.get("params", {})
        if isinstance(params, dict):
            request_params = dict(params)

    datasets_value = mathvista_cfg.get("datasets", ["MathVista_testmini"])
    if isinstance(datasets_value, str):
        datasets = [name.strip() for name in datasets_value.split(",") if name.strip()]
    elif isinstance(datasets_value, list):
        datasets = [str(name).strip() for name in datasets_value if str(name).strip()]
    else:
        datasets = ["MathVista_testmini"]
    if not datasets:
        raise ValueError("`mathvista.datasets` must contain at least one dataset name.")

    out_dir = _resolve_path(str(mathvista_cfg.get("out_dir", f"output/mathvista/{backbone}")), repo_root)
    image_dir = _resolve_path(str(mathvista_cfg.get("image_dir", out_dir / "images")), repo_root)
    score_output_path = mathvista_cfg.get("score_output_path")
    cache_dir = mathvista_cfg.get("cache_dir")
    max_samples = int(mathvista_cfg.get("max_samples", 0) or 0)
    use_cot = bool(mathvista_cfg.get("cot", False))
    run_calculation = bool(mathvista_cfg.get("run_calculation", True))
    run_extract = bool(mathvista_cfg.get("run_extract", True))
    openai_api_key = mathvista_cfg.get("openai_api_key")
    gt_file = mathvista_cfg.get("gt_file")
    resume = bool(mathvista_cfg.get("resume", False))

    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)

    summary: dict[str, Any] = {
        "benchmark": "mathvista",
        "backbone": backbone,
        "out_dir": str(out_dir),
        "datasets": datasets,
        "cot": use_cot,
    }

    for ds_name in datasets:
        entry = DS_COLLECTIONS.get(ds_name)
        if not entry:
            raise ValueError(f"Unknown MathVista dataset: {ds_name}")

        dataset_root = str(mathvista_cfg.get("root", entry["root"]))
        split = str(mathvista_cfg.get("split", entry["split"]))
        dataset = load_dataset(
            dataset_root,
            cache_dir=str(_resolve_path(cache_dir, repo_root)) if cache_dir else None,
        )
        data = dataset[split]

        results_file: Path | None = None
        if resume:
            candidates = sorted(out_dir.glob(f"{ds_name}_*.json"))
            if candidates:
                results_file = max(candidates, key=lambda p: p.stat().st_mtime)
                print(f"[umm eval] resume enabled: using {results_file}")

        if results_file is None:
            results: dict[str, Any] = {}
            for idx, data_item in enumerate(data, start=1):
                image = data_item.get("decoded_image")
                if image is None:
                    raise ValueError("MathVista sample missing `decoded_image`.")
                if not isinstance(image, Image.Image):
                    raise ValueError("Expected `decoded_image` to be a PIL image.")

                pid = data_item.get("pid")
                if pid is None:
                    raise ValueError("MathVista sample missing `pid`.")
                image_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / f"{pid}.png"
                image.save(image_path, format="PNG")

                question = data_item.get("query")
                if question is None:
                    raise ValueError("MathVista sample missing `query`.")
                if use_cot:
                    prompt = COT_INSTRUCTION.format(question=question)
                else:
                    prompt = question

                payload = {
                    "backbone": backbone,
                    "task": "understanding",
                    "prompt": prompt,
                    "images": [str(image_path)],
                    "params": request_params,
                    "metadata": {"pid": pid, "dataset": ds_name},
                }
                response = _extract_text(pipeline.run(payload))

                item = dict(data_item)
                item.pop("decoded_image", None)
                item["response"] = response
                results[str(pid)] = item

                if max_samples > 0 and idx >= max_samples:
                    break

            time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
            results_file = out_dir / f"{ds_name}_{time_prefix}.json"
            results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

        summary[f"{ds_name}_output_path"] = str(results_file)

        if run_extract:
            cmd = [
                sys.executable,
                "src/umm/eval/internvl_chat/eval/mathvista/extract_answer.py",
                "--output_file",
                results_file.name,
                "--output_dir",
                str(out_dir),
            ]
            if use_cot:
                cmd.append("--quick_extract")
            env = None
            if isinstance(openai_api_key, str) and openai_api_key.strip():
                env = dict(os.environ)
                env["OPENAI_API_KEY"] = openai_api_key.strip()
            proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, env=env)
            print(proc.stdout)
            if proc.returncode != 0:
                if proc.stderr:
                    print(proc.stderr, file=sys.stderr)
                raise RuntimeError(f"MathVista extract_answer failed with return code {proc.returncode}")
            summary[f"{ds_name}_extract_stdout"] = proc.stdout

        if run_calculation:
            score_file = results_file.with_name(f"{results_file.stem}_score.json")
            cmd = [
                sys.executable,
                "src/umm/eval/internvl_chat/eval/mathvista/calculate_score.py",
                "--output_file",
                results_file.name,
                "--output_dir",
                str(out_dir),
                "--score_file",
                score_file.name,
            ]
            if isinstance(gt_file, str) and gt_file.strip():
                cmd.extend(["--gt_file", gt_file.strip()])
            env = None
            if isinstance(openai_api_key, str) and openai_api_key.strip():
                env = dict(os.environ)
                env["OPENAI_API_KEY"] = openai_api_key.strip()
            proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, env=env)
            print(proc.stdout)
            if proc.returncode != 0:
                if proc.stderr:
                    print(proc.stderr, file=sys.stderr)
                raise RuntimeError(f"MathVista calculate_score failed with return code {proc.returncode}")
            summary[f"{ds_name}_score_file"] = str(score_file)
            summary[f"{ds_name}_score_stdout"] = proc.stdout

    if isinstance(score_output_path, str) and score_output_path:
        score_path = _resolve_path(score_output_path, repo_root)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[umm eval] wrote MathVista summary to {score_path}")

    print(f"[umm eval] completed MathVista for backbone={backbone}, outputs={out_dir}")
    return 0
