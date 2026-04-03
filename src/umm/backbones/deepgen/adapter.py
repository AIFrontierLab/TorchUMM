"""DeepGen backbone adapter for text-to-image generation and image editing.

Uses the diffusers-compatible pipeline (deepgenteam/DeepGen-1.0-diffusers)
which provides a self-contained DiffusionPipeline with no external repo needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image


class DeepGenBackbone:
    """DeepGen 1.0 backbone adapter (5B: 3B VLM + 2B DiT).

    Supports text-to-image generation and image editing via the diffusers
    format pipeline. Does NOT support image understanding.
    """

    name = "deepgen"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        **kwargs: Any,
    ) -> None:
        self.model_path = model_path or "deepgenteam/DeepGen-1.0-diffusers"
        self.device = device
        self.torch_dtype = torch_dtype
        self.seed = kwargs.get("seed", 42)
        self.pipeline = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_dtype(name: str) -> torch.dtype:
        return {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(name, torch.bfloat16)

    def _ensure_pipeline(self) -> None:
        """Lazy-load the diffusers pipeline."""
        if self.pipeline is not None:
            return
        from diffusers import DiffusionPipeline

        dtype = self._resolve_dtype(self.torch_dtype)
        print(f"Loading DeepGen model from {self.model_path} ...")
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.pipeline.to(self.device)
        print("DeepGen model loaded.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, cfg: dict[str, Any]) -> None:
        """Load or update configuration from YAML backbone_cfg."""
        changed = False
        for key in ("model_path", "device", "torch_dtype"):
            if key in cfg and cfg[key] != getattr(self, key):
                setattr(self, key, cfg[key])
                changed = True
        if "seed" in cfg:
            self.seed = int(cfg["seed"])
        # Reset pipeline if model config changed
        if changed:
            self.pipeline = None

    def generation(
        self,
        prompt: str | None,
        output_path: str | None,
        generation_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        """Text-to-image generation.

        Returns dict with 'images' (list[PIL.Image]) and 'image_paths' (list[str]).
        """
        if not prompt:
            raise ValueError("Generation requires a non-empty prompt.")

        self._ensure_pipeline()

        height = generation_cfg.get("height", 512)
        width = generation_cfg.get("width", 512)
        num_steps = generation_cfg.get("num_inference_steps", generation_cfg.get("num_steps", 50))
        guidance_scale = generation_cfg.get("guidance_scale", generation_cfg.get("cfg_scale", 4.0))
        seed = generation_cfg.get("seed", self.seed)

        result = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        imgs = result.images if hasattr(result, "images") else result
        image_paths: list[str] = []

        if output_path:
            save_path = Path(output_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(imgs, list) and imgs:
                imgs[0].save(str(save_path), format="PNG")
                image_paths.append(str(save_path))
                print(f"  Saved: {save_path}")
        else:
            for i, img in enumerate(imgs if isinstance(imgs, list) else [imgs]):
                if isinstance(img, Image.Image):
                    image_paths.append(f"deepgen_generated_{i}.png")

        return {"images": list(imgs) if isinstance(imgs, list) else [imgs], "image_paths": image_paths}

    def editing(
        self,
        prompt: str | None,
        images: list[str] | None,
        output_path: str | None,
        editing_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        """Image editing (image-to-image).

        Returns dict with 'images' (list[PIL.Image]) and 'image_paths' (list[str]).
        """
        if not prompt:
            raise ValueError("Editing requires a non-empty prompt.")
        if not images:
            raise ValueError("Editing requires at least one source image.")

        self._ensure_pipeline()

        if isinstance(images[0], Image.Image):
            src_img = images[0].convert("RGB")
        else:
            src_img = Image.open(images[0]).convert("RGB")

        height = editing_cfg.get("height", 512)
        width = editing_cfg.get("width", 512)
        num_steps = editing_cfg.get("num_inference_steps", editing_cfg.get("num_steps", 50))
        guidance_scale = editing_cfg.get("guidance_scale", editing_cfg.get("cfg_scale", 4.0))
        negative_prompt = editing_cfg.get(
            "negative_prompt",
            "blurry, low quality, low resolution, distorted, deformed, "
            "broken content, missing parts, damaged details, artifacts, "
            "glitch, noise, pixelated, grainy, compression artifacts, "
            "bad composition, wrong proportion, incomplete editing, "
            "unfinished, unedited areas.",
        )
        seed = editing_cfg.get("seed", self.seed)

        result = self.pipeline(
            prompt=prompt,
            image=src_img,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        imgs = result.images if hasattr(result, "images") else result
        image_paths: list[str] = []

        if output_path:
            save_path = Path(output_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(imgs, list) and imgs:
                imgs[0].save(str(save_path), format="PNG")
                image_paths.append(str(save_path))
                print(f"  Saved: {save_path}")

        return {"images": list(imgs) if isinstance(imgs, list) else [imgs], "image_paths": image_paths}
