from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from extensions.text2image import DiffusersTextToImageBackbone, TextToImageModelSpec


SPEC = TextToImageModelSpec(
    name="hidream_o1",
    model_id="HiDream-ai/HiDream-O1-Image-Dev-2604",
    pipeline_class="HiDreamO1Pipeline",
    default_generation_cfg={
        "height": 2048,
        "width": 2048,
        "num_inference_steps": 28,
        "guidance_scale": 0.0,
        "shift": 1.0,
    },
    default_load_cfg={"model_type": "dev"},
)


class HiDreamO1Backbone(DiffusersTextToImageBackbone):
    def __init__(self) -> None:
        super().__init__(SPEC)

    def _build_diffusers_pipeline(self) -> Any:
        source_root = self.load_cfg.get("source_root")
        if not source_root:
            raise ValueError("Set backbone_cfg.source_root to the official HiDream-O1-Image checkout.")
        source_path = str(Path(source_root).expanduser().resolve())
        if source_path not in sys.path:
            sys.path.insert(0, source_path)

        from transformers import AutoProcessor

        model_module = importlib.import_module("models.qwen3_vl_transformers")
        pipeline_module = importlib.import_module("models.pipeline")
        cache_dir = self._ensure_cache_dir()
        load_kwargs: dict[str, Any] = {"local_files_only": self.local_files_only}
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        processor = AutoProcessor.from_pretrained(self.model_path, **load_kwargs)
        model = model_module.Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self._resolve_dtype(self.torch_dtype),
            device_map=self.device,
            **load_kwargs,
        ).eval()
        return _HiDreamO1Runtime(
            model=model,
            processor=processor,
            generate_image=pipeline_module.generate_image,
            default_timesteps=pipeline_module.DEFAULT_TIMESTEPS,
            model_type=str(self.load_cfg.get("model_type", "dev")),
        )

    def _call_pipeline(self, prompt: str, generation_cfg: dict[str, Any]) -> Any:
        assert self.pipeline is not None
        call_cfg = dict(self.spec.default_generation_cfg)
        call_cfg.update(generation_cfg)
        return self.pipeline(prompt=prompt, seed=int(call_cfg.pop("seed", 42)), **call_cfg)


class _HiDreamO1Runtime:
    def __init__(
        self,
        model: Any,
        processor: Any,
        generate_image: Any,
        default_timesteps: Any,
        model_type: str,
    ) -> None:
        self.model = model
        self.processor = processor
        self.generate_image = generate_image
        self.default_timesteps = default_timesteps
        self.model_type = model_type

    def __call__(self, prompt: str, seed: int, **kwargs: Any) -> Image.Image:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        for attr, token in {
            "boi_token": "<|boi_token|>",
            "bor_token": "<|bor_token|>",
            "eor_token": "<|eor_token|>",
            "bot_token": "<|bot_token|>",
            "tms_token": "<|tms_token|>",
        }.items():
            setattr(tokenizer, attr, token)
        num_steps = int(kwargs.pop("num_inference_steps", 28))
        guidance_scale = float(kwargs.pop("guidance_scale", 0.0))
        shift = float(kwargs.pop("shift", 1.0))
        return self.generate_image(
            model=self.model,
            processor=self.processor,
            prompt=prompt,
            ref_image_paths=[],
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            timesteps_list=self.default_timesteps if self.model_type == "dev" else None,
            scheduler_name="flash" if self.model_type == "dev" else "default",
            seed=seed,
            **kwargs,
        )


BACKBONES = {SPEC.name: HiDreamO1Backbone}
