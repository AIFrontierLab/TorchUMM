from __future__ import annotations

import gc
import importlib
import tempfile
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .specs import TextToImageModelSpec


PipelineFactory = Callable[[TextToImageModelSpec, dict[str, Any]], Any]


class DiffusersTextToImageBackbone:
    """Shared lazy adapter for open-weight text-to-image pipelines.

    Checkpoints are never downloaded during ``load``.  They are loaded
    lazily on the first generation call.  Tests can inject ``pipeline_factory``
    to exercise the complete framework path without touching the network.
    """

    def __init__(
        self,
        spec: TextToImageModelSpec,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self.spec = spec
        self.name = spec.name
        self.model_path = spec.model_id
        self.device = "cuda:0"
        self.torch_dtype = "bfloat16"
        self.pipeline: Any | None = None
        self.pipeline_factory = pipeline_factory
        self.load_cfg: dict[str, Any] = dict(spec.default_load_cfg)
        self.ephemeral_cache = False
        self.local_files_only = False
        self._cache_context: tempfile.TemporaryDirectory[str] | None = None

    def load(self, cfg: dict[str, Any]) -> None:
        """Store configuration without loading model weights."""

        reset_keys = {
            "model_path",
            "device",
            "torch_dtype",
            "ephemeral_cache",
            "local_files_only",
            "revision",
            "variant",
        }
        if any(key in cfg and cfg[key] != getattr(self, key, self.load_cfg.get(key)) for key in reset_keys):
            self.unload()

        self.model_path = str(cfg.get("model_path", self.model_path))
        self.device = str(cfg.get("device", self.device))
        self.torch_dtype = str(cfg.get("torch_dtype", self.torch_dtype))
        self.ephemeral_cache = bool(cfg.get("ephemeral_cache", self.ephemeral_cache))
        self.local_files_only = bool(cfg.get("local_files_only", self.local_files_only))

        reserved = {
            "model_path",
            "device",
            "torch_dtype",
            "ephemeral_cache",
            "local_files_only",
            "enable_cpu_offload",
            "enable_sequential_cpu_offload",
        }
        self.load_cfg.update({key: value for key, value in cfg.items() if key not in reserved})
        self.load_cfg["enable_cpu_offload"] = bool(cfg.get("enable_cpu_offload", False))
        self.load_cfg["enable_sequential_cpu_offload"] = bool(
            cfg.get("enable_sequential_cpu_offload", False)
        )

    @staticmethod
    def _resolve_dtype(name: str) -> Any:
        import torch

        normalized = str(name).lower()
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported torch dtype: {name}")
        return mapping[normalized]

    def _ensure_cache_dir(self) -> str | None:
        if not self.ephemeral_cache:
            cache_dir = self.load_cfg.get("cache_dir")
            return str(cache_dir) if cache_dir else None
        if self._cache_context is None:
            self._cache_context = tempfile.TemporaryDirectory(prefix=f"umm-{self.name}-")
        return self._cache_context.name

    def _diffusers_load_kwargs(self) -> dict[str, Any]:
        ignored = {
            "enable_cpu_offload",
            "enable_sequential_cpu_offload",
            "cache_dir",
        }
        kwargs = {key: value for key, value in self.load_cfg.items() if key not in ignored}
        kwargs["torch_dtype"] = self._resolve_dtype(self.torch_dtype)
        kwargs["local_files_only"] = self.local_files_only
        cache_dir = self._ensure_cache_dir()
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        return kwargs

    def _build_diffusers_pipeline(self) -> Any:
        diffusers = importlib.import_module("diffusers")
        try:
            pipeline_cls = getattr(diffusers, self.spec.pipeline_class)
        except AttributeError as exc:
            raise ImportError(
                f"Installed diffusers does not provide {self.spec.pipeline_class}. "
                "Upgrade diffusers or install the model's documented revision."
            ) from exc
        pipeline = pipeline_cls.from_pretrained(self.model_path, **self._diffusers_load_kwargs())
        if self.load_cfg.get("enable_sequential_cpu_offload"):
            pipeline.enable_sequential_cpu_offload()
        elif self.load_cfg.get("enable_cpu_offload"):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(self.device)
        return pipeline

    def _ensure_pipeline(self) -> None:
        if self.pipeline is not None:
            return
        if self.pipeline_factory is not None:
            self.pipeline = self.pipeline_factory(self.spec, dict(self.load_cfg))
        else:
            self.pipeline = self._build_diffusers_pipeline()

    @staticmethod
    def _make_generator(seed: int) -> Any:
        import torch

        return torch.Generator(device="cpu").manual_seed(seed)

    def _call_pipeline(self, prompt: str, generation_cfg: dict[str, Any]) -> Any:
        assert self.pipeline is not None
        call_cfg = dict(self.spec.default_generation_cfg)
        call_cfg.update(generation_cfg)
        if "num_steps" in call_cfg and "num_inference_steps" not in generation_cfg:
            call_cfg["num_inference_steps"] = call_cfg.pop("num_steps")
        if "cfg_scale" in call_cfg and "guidance_scale" not in generation_cfg:
            call_cfg["guidance_scale"] = call_cfg.pop("cfg_scale")
        seed = int(call_cfg.pop("seed", 42))

        call_cfg.setdefault("generator", self._make_generator(seed))
        return self.pipeline(prompt=prompt, **call_cfg)

    @staticmethod
    def _extract_images(result: Any) -> list[Image.Image]:
        if isinstance(result, Image.Image):
            return [result]
        images = getattr(result, "images", result)
        if isinstance(images, Image.Image):
            return [images]
        if isinstance(images, (list, tuple)) and all(isinstance(item, Image.Image) for item in images):
            return list(images)
        raise TypeError(
            "Text-to-image pipeline must return a PIL image, a sequence of PIL images, "
            "or an object with an `.images` sequence."
        )

    @staticmethod
    def _save_images(images: list[Image.Image], output_path: str | None) -> list[str]:
        if not output_path:
            return []
        base = Path(output_path).expanduser()
        base.parent.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for index, image in enumerate(images):
            path = base if index == 0 else base.with_name(f"{base.stem}_{index}{base.suffix or '.png'}")
            image.save(path)
            paths.append(str(path))
        return paths

    def generation(
        self,
        prompt: str | None,
        output_path: str | None,
        generation_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        if not prompt or not str(prompt).strip():
            raise ValueError("Generation requires a non-empty prompt.")
        self._ensure_pipeline()
        result = self._call_pipeline(str(prompt), generation_cfg)
        images = self._extract_images(result)
        return {
            "images": images,
            "image_paths": self._save_images(images, output_path),
            "model": self.name,
            "model_path": self.model_path,
        }

    def unload(self) -> None:
        """Release model references, CUDA allocations, and owned temp cache."""

        self.pipeline = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        if self._cache_context is not None:
            self._cache_context.cleanup()
            self._cache_context = None

    close = unload
