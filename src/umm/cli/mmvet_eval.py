from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from umm.core.config import load_config
from umm.inference import InferencePipeline


DS_COLLECTIONS = {
    "mmvet": {
        "max_new_tokens": 1000,
    }
}


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
    mmvet_cfg = raw_cfg.get("mmvet", {}) if isinstance(raw_cfg.get("mmvet"), dict) else {}
    inference_cfg = raw_cfg.get("inference", {}) if isinstance(raw_cfg.get("inference"), dict) else {}
    if not eval_cfg and "benchmark" in raw_cfg:
        eval_cfg = {"benchmark": raw_cfg.get("benchmark")}
    return eval_cfg, mmvet_cfg, inference_cfg


def run_mmvet_eval_command(args: Any) -> int:
    config_path = str(args.config)
    eval_cfg, mmvet_cfg, inference_cfg = _load_eval_cfg(config_path)
    benchmark = str(eval_cfg.get("benchmark", "")).strip().lower()
    if benchmark != "mmvet":
        raise ValueError(f"Expected `eval.benchmark: mmvet`, got: {benchmark or '<empty>'}")

    repo_root = Path(__file__).resolve().parents[3]

    backbone_raw = inference_cfg.get("backbone")
    if not isinstance(backbone_raw, str) or not backbone_raw:
        raise ValueError("`inference.backbone` is required for MM-Vet eval.")
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

    datasets_value = mmvet_cfg.get("datasets", ["mmvet"])
    if isinstance(datasets_value, str):
        datasets = [name.strip() for name in datasets_value.split(",") if name.strip()]
    elif isinstance(datasets_value, list):
        datasets = [str(name).strip() for name in datasets_value if str(name).strip()]
    else:
        datasets = ["mmvet"]
    if not datasets:
        raise ValueError("`mmvet.datasets` must contain at least one dataset name.")

    out_dir = _resolve_path(str(mmvet_cfg.get("out_dir", f"output/mmvet/{backbone}")), repo_root)
    score_output_path = mmvet_cfg.get("score_output_path")
    max_samples = int(mmvet_cfg.get("max_samples", 0) or 0)

    dataset_paths = mmvet_cfg.get("dataset_paths", {})
    if not isinstance(dataset_paths, dict):
        dataset_paths = {}

    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)

    summary: dict[str, Any] = {
        "benchmark": "mmvet",
        "backbone": backbone,
        "out_dir": str(out_dir),
        "datasets": datasets,
    }

    for ds_name in datasets:
        entry = DS_COLLECTIONS.get(ds_name)
        if not entry and ds_name not in dataset_paths:
            raise ValueError(f"Unknown MM-Vet dataset: {ds_name}")
        image_root_value = dataset_paths.get("image_root")
        question_value = dataset_paths.get("question")
        if not image_root_value or not question_value:
            raise ValueError(
                "MM-Vet requires `mmvet.dataset_paths.image_root` and "
                "`mmvet.dataset_paths.question` to be set in the YAML config."
            )
        image_root = _resolve_path(str(image_root_value), repo_root)
        question_path = _resolve_path(str(question_value), repo_root)
        if not image_root.exists():
            raise FileNotFoundError(f"MM-Vet image root not found: {image_root}")
        if not question_path.exists():
            raise FileNotFoundError(f"MM-Vet question file not found: {question_path}")

        outputs = {}
        with question_path.open("r", encoding="utf-8") as reader:
            for idx, line in enumerate(reader, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                image_name = row["image"]
                question = row["text"]
                question_id = row["question_id"]

                image_path = image_root / image_name
                if not image_path.exists():
                    raise FileNotFoundError(f"MM-Vet image not found: {image_path}")

                # Ensure image is readable early (helps catch corrupt files)
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                except Exception as exc:
                    raise RuntimeError(f"Failed to open image {image_path}: {exc}") from exc

                payload = {
                    "backbone": backbone,
                    "task": "understanding",
                    "prompt": question,
                    "images": [str(image_path)],
                    "params": request_params,
                    "metadata": {"question_id": question_id, "dataset": ds_name},
                }
                response = _extract_text(pipeline.run(payload))
                outputs[f"v1_{question_id}"] = response

                if max_samples > 0 and idx >= max_samples:
                    break

        time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
        results_file = out_dir / f"{ds_name}_{time_prefix}.json"
        results_file.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
        summary[f"{ds_name}_output_path"] = str(results_file)

    if isinstance(score_output_path, str) and score_output_path:
        score_path = _resolve_path(score_output_path, repo_root)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[umm eval] wrote MM-Vet summary to {score_path}")

    print(f"[umm eval] completed MM-Vet for backbone={backbone}, outputs={out_dir}")
    return 0
