from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from umm.core.config import load_config
from umm.inference import InferencePipeline


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


def _post_process(response: str) -> str:
    response = response.replace("\n", "").replace("不是", "No").replace("是", "Yes").replace("否", "No")
    response = response.lower().replace("true", "yes").replace("false", "no")
    response = re.sub(re.compile(r"[\u4e00-\u9fa5]"), "", response)
    return response


def _load_eval_cfg(config_path: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_cfg = load_config(config_path)
    eval_cfg = raw_cfg.get("eval", {}) if isinstance(raw_cfg.get("eval"), dict) else {}
    mme_cfg = raw_cfg.get("mme", {}) if isinstance(raw_cfg.get("mme"), dict) else {}
    inference_cfg = raw_cfg.get("inference", {}) if isinstance(raw_cfg.get("inference"), dict) else {}
    if not eval_cfg and "benchmark" in raw_cfg:
        eval_cfg = {"benchmark": raw_cfg.get("benchmark")}
    return eval_cfg, mme_cfg, inference_cfg


def run_mme_eval_command(args: Any) -> int:
    config_path = str(args.config)
    eval_cfg, mme_cfg, inference_cfg = _load_eval_cfg(config_path)
    benchmark = str(eval_cfg.get("benchmark", "")).strip().lower()
    if benchmark != "mme":
        raise ValueError(f"Expected `eval.benchmark: mme`, got: {benchmark or '<empty>'}")

    repo_root = Path(__file__).resolve().parents[3]

    backbone_raw = inference_cfg.get("backbone")
    if not isinstance(backbone_raw, str) or not backbone_raw:
        raise ValueError("`inference.backbone` is required for MME eval.")
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

    dataset_root = _resolve_path(
        str(mme_cfg.get("root", "src/umm/eval/internvl_chat/eval/mme/Your_Results")),
        repo_root,
    )
    image_root = _resolve_path(
        str(mme_cfg.get("image_root", "data/mme/MME_Benchmark_release_version")),
        repo_root,
    )
    out_dir = _resolve_path(str(mme_cfg.get("out_dir", f"output/mme/{backbone}")), repo_root)
    prompt_suffix = str(mme_cfg.get("prompt_suffix", "Answer the question using a single word or phrase."))
    run_calculation = bool(mme_cfg.get("run_calculation", True))
    score_output_path = mme_cfg.get("score_output_path")
    calculation_script = _resolve_path(
        str(mme_cfg.get("calculation_script", "src/umm/eval/internvl_chat/eval/mme/calculation.py")),
        repo_root,
    )
    max_samples = int(mme_cfg.get("max_samples", 0) or 0)

    if not dataset_root.exists():
        raise FileNotFoundError(f"MME question root not found: {dataset_root}")
    if not image_root.exists():
        raise FileNotFoundError(f"MME image root not found: {image_root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)

    txt_files = sorted([p for p in dataset_root.iterdir() if p.suffix == ".txt"])
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in MME root: {dataset_root}")

    total_written = 0
    missing_images = 0
    skipped_rows = 0
    for task_txt in txt_files:
        task_name = task_txt.stem
        out_file = out_dir / task_txt.name
        with task_txt.open("r", encoding="utf-8") as fin, out_file.open("w", encoding="utf-8") as fout:
            for idx, line in enumerate(fin, start=1):
                row = line.strip().split("\t")
                if len(row) != 3:
                    skipped_rows += 1
                    continue
                img, question, gt = row
                prompt = f"{question} {prompt_suffix}".strip()

                img_path = image_root / task_name / img
                if not img_path.exists():
                    img_path = image_root / task_name / "images" / img
                if not img_path.exists():
                    missing_images += 1
                    continue

                payload = {
                    "backbone": backbone,
                    "task": "understanding",
                    "prompt": prompt,
                    "images": [str(img_path)],
                    "params": request_params,
                }
                output = pipeline.run(payload)
                response = _post_process(_extract_text(output))
                print(img, prompt, gt, response, sep="\t", file=fout)
                total_written += 1
                if max_samples > 0 and idx >= max_samples:
                    break

    summary: dict[str, Any] = {
        "benchmark": "mme",
        "backbone": backbone,
        "out_dir": str(out_dir),
        "samples_written": total_written,
    }

    if total_written == 0:
        print(
            "[umm eval] warning: no MME samples were written. "
            "Skipping calculation. Check `mme.root` and `mme.image_root`."
        )
        run_calculation = False

    if run_calculation:
        cmd = [sys.executable, str(calculation_script), "--results_dir", str(out_dir)]
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"MME calculation failed with return code {proc.returncode}")
        summary["calculation_stdout"] = proc.stdout

    if isinstance(score_output_path, str) and score_output_path:
        score_path = _resolve_path(score_output_path, repo_root)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[umm eval] wrote MME summary to {score_path}")

    summary["missing_images"] = missing_images
    summary["skipped_rows"] = skipped_rows
    print(
        f"[umm eval] completed MME for backbone={backbone}, outputs={out_dir}, "
        f"samples_written={total_written}, missing_images={missing_images}, skipped_rows={skipped_rows}"
    )
    return 0
