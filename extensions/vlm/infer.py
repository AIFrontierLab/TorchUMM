"""Dedicated inference entry point for the opt-in vision-language extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from umm.cli.infer import _resolve_requests, _serialize_results, _unwrap_inference_block
from umm.core.config import load_config
from umm.inference import InferencePipeline

from . import register


def run_infer(config_path: str) -> list[Any]:
    """Run an extension config and always release the loaded pipeline."""

    cfg = _unwrap_inference_block(load_config(config_path))
    backbone_name = cfg.get("backbone")
    if not isinstance(backbone_name, str) or not backbone_name:
        raise ValueError("Inference config requires non-empty `backbone`.")
    backbone_cfg = cfg.get("backbone_cfg", {})
    if not isinstance(backbone_cfg, dict):
        raise ValueError("`backbone_cfg` must be a dict if provided.")

    register()
    pipeline = InferencePipeline(backbone_name, backbone_cfg)
    try:
        payloads = []
        for request in _resolve_requests(cfg):
            payload = dict(request)
            payload["backbone"] = backbone_name
            payloads.append(payload)
        if len(payloads) == 1:
            return [pipeline.run(payloads[0])]
        return pipeline.run_many(payloads, batch_size=int(cfg.get("batch_size", 1)))
    finally:
        close = getattr(pipeline.backbone, "close", None)
        if callable(close):
            close()
        else:
            unload = getattr(pipeline.backbone, "unload", None)
            if callable(unload):
                unload()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    results = run_infer(args.config)
    output_json = args.output_json
    if output_json:
        with Path(output_json).open("w", encoding="utf-8") as handle:
            json.dump(_serialize_results(results), handle, indent=2)
    print(f"[vlm] completed {len(results)} request(s) from {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
