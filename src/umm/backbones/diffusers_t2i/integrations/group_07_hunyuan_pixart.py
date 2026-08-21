from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from umm.backbones.diffusers_t2i import (
    DiffusersTextToImageBackbone,
    TextToImageModelSpec,
    factories_from_specs,
)


HUNYUAN_SPEC = TextToImageModelSpec(
    name="hunyuan_image_2_1",
    model_id="hunyuanimage-v2.1-distilled",
    pipeline_class="HunyuanImagePipeline",
    default_generation_cfg={
        "height": 2048,
        "width": 2048,
        "num_inference_steps": 8,
        "guidance_scale": 3.25,
        "shift": 4,
        "use_reprompt": False,
        "use_refiner": False,
    },
    default_load_cfg={"use_fp8": True},
)


class HunyuanImage21Backbone(DiffusersTextToImageBackbone):
    def __init__(self) -> None:
        super().__init__(HUNYUAN_SPEC)

    def _build_diffusers_pipeline(self) -> Any:
        source_root = self.load_cfg.get("source_root")
        if not source_root:
            raise ValueError("Set backbone_cfg.source_root to the official HunyuanImage-2.1 checkout.")
        source_path = str(Path(source_root).expanduser().resolve())
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        try:
            module = importlib.import_module("hyimage.diffusion.pipelines.hunyuanimage_pipeline")
        except ImportError as exc:
            raise ImportError("Could not import the official HunyuanImage-2.1 runtime.") from exc
        pipeline = module.HunyuanImagePipeline.from_pretrained(
            model_name=self.model_path,
            use_fp8=bool(self.load_cfg.get("use_fp8", True)),
        )
        return pipeline.to(self.device)

    def _call_pipeline(self, prompt: str, generation_cfg: dict[str, Any]) -> Any:
        assert self.pipeline is not None
        call_cfg = dict(self.spec.default_generation_cfg)
        call_cfg.update(generation_cfg)
        return self.pipeline(prompt=prompt, seed=int(call_cfg.pop("seed", 42)), **call_cfg)


PIXART_SPECS = {
    "pixart_sigma": TextToImageModelSpec(
        name="pixart_sigma",
        model_id="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        pipeline_class="PixArtSigmaPipeline",
        default_generation_cfg={"height": 1024, "width": 1024, "num_inference_steps": 20, "guidance_scale": 4.5},
    )
}

BACKBONES = {"hunyuan_image_2_1": HunyuanImage21Backbone, **factories_from_specs(PIXART_SPECS)}
