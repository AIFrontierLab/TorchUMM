from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from umm.core.config import load_config
from umm.inference import InferencePipeline


def _unwrap_inference_block(config: dict[str, Any]) -> dict[str, Any]:
    block = config.get("inference")
    if isinstance(block, dict):
        return block
    return config


def _resolve_requests(config: dict[str, Any]) -> list[dict[str, Any]]:
    if "requests" in config:
        requests = config["requests"]
        if not isinstance(requests, list):
            raise ValueError("`requests` must be a list.")
        return [dict(item) for item in requests]

    if "request" in config:
        req = config["request"]
        if not isinstance(req, dict):
            raise ValueError("`request` must be a dict.")
        return [dict(req)]

    task = config.get("task")
    if task is None:
        raise ValueError("Config must include `request`, `requests`, or top-level `task` fields.")

    req = {
        "task": task,
        "prompt": config.get("prompt"),
        "images": config.get("images", []),
        "videos": config.get("videos", []),
        "params": config.get("params", {}),
        "metadata": config.get("metadata", {}),
        "output_path": config.get("output_path"),
    }
    return [req]


def _serialize_results(results: list[Any]) -> list[dict[str, Any]]:
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, list):
            return [_to_jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(k): _to_jsonable(v) for k, v in value.items()}
        if value.__class__.__name__ == "Image":
            return "<PIL.Image>"
        return str(value)

    serializable: list[dict[str, Any]] = []
    for item in results:
        if isinstance(item, dict):
            entry: dict[str, Any] = {}
            for k, v in item.items():
                entry[k] = _to_jsonable(v)
            serializable.append(entry)
        else:
            serializable.append({"result": _to_jsonable(item)})
    return serializable


def _extract_tasks_from_config(config_path: str) -> list[str]:
    raw_cfg = load_config(config_path)
    cfg = _unwrap_inference_block(raw_cfg)
    requests = _resolve_requests(cfg)
    tasks: list[str] = []
    for req in requests:
        task = req.get("task")
        if isinstance(task, str):
            tasks.append(task)
    return tasks


def _extract_output_json_from_config(config_path: str) -> str | None:
    raw_cfg = load_config(config_path)
    cfg = _unwrap_inference_block(raw_cfg)
    value = cfg.get("output_json") or cfg.get("output_json_path")
    if isinstance(value, str) and value:
        return value
    return None


def run_infer(config_path: str) -> list[Any]:
    raw_cfg = load_config(config_path)
    cfg = _unwrap_inference_block(raw_cfg)

    backbone_name = cfg.get("backbone")
    if not isinstance(backbone_name, str) or not backbone_name:
        raise ValueError("Inference config requires non-empty `backbone`.")

    backbone_cfg = cfg.get("backbone_cfg", {})
    if not isinstance(backbone_cfg, dict):
        raise ValueError("`backbone_cfg` must be a dict if provided.")

    requests = _resolve_requests(cfg)
    pipeline = InferencePipeline(backbone_name=backbone_name, backbone_cfg=backbone_cfg)

    normalized_payloads: list[dict[str, Any]] = []
    for req in requests:
        payload = dict(req)
        payload["backbone"] = backbone_name
        normalized_payloads.append(payload)

    if len(normalized_payloads) == 1:
        return [pipeline.run(normalized_payloads[0])]

    batch_size = int(cfg.get("batch_size", 1))
    return pipeline.run_many(normalized_payloads, batch_size=batch_size)


def run_infer_command(args: Any) -> int:
    results = run_infer(args.config)
    print(f"[umm infer] completed {len(results)} request(s) from {Path(args.config)}")

    output_json = getattr(args, "output_json", None)
    if not output_json:
        output_json = _extract_output_json_from_config(args.config)
    if not output_json:
        tasks = _extract_tasks_from_config(args.config)
        if "understanding" in tasks:
            output_json = str(Path(args.config).with_suffix(".results.json"))

    if output_json:
        serializable = _serialize_results(results)
        with Path(output_json).open("w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        print(f"[umm infer] wrote summary JSON to {output_json}")
    return 0
