from __future__ import annotations

import gc
import importlib
from typing import Any, Callable

from .specs import VLMModelSpec


PipelineFactory = Callable[[VLMModelSpec, dict[str, Any]], Any]


class TransformersVLMBackbone:
    """Lazy adapter for Transformers ``image-text-to-text`` pipelines.

    The model cards for the supported integrations expose the same pipeline
    task, which lets the framework share loading, prompt construction, and
    result normalization while preserving per-model defaults in ``VLMModelSpec``.
    """

    def __init__(
        self,
        spec: VLMModelSpec,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self.spec = spec
        self.name = spec.name
        self.model_path = spec.model_id
        self.device = "cuda:0"
        self.torch_dtype = "bfloat16"
        self.device_map: str | dict[str, Any] | None = None
        self.trust_remote_code = spec.trust_remote_code
        self.local_files_only = False
        self.pipeline: Any | None = None
        self.pipeline_factory = pipeline_factory
        self.load_cfg: dict[str, Any] = dict(spec.default_load_cfg)

    def load(self, cfg: dict[str, Any]) -> None:
        """Store configuration without downloading or loading model weights."""

        reset_keys = {
            "model_path",
            "device",
            "torch_dtype",
            "device_map",
            "trust_remote_code",
            "local_files_only",
            "revision",
            "token",
            "cache_dir",
            "model_kwargs",
        }
        if any(
            key in cfg and cfg[key] != getattr(self, key, self.load_cfg.get(key))
            for key in reset_keys
        ):
            self.unload()

        self.model_path = str(cfg.get("model_path", self.model_path))
        self.device = str(cfg.get("device", self.device))
        self.torch_dtype = str(cfg.get("torch_dtype", self.torch_dtype))
        self.device_map = cfg.get("device_map", self.device_map)
        self.trust_remote_code = bool(cfg.get("trust_remote_code", self.trust_remote_code))
        self.local_files_only = bool(cfg.get("local_files_only", self.local_files_only))
        self.load_cfg.update(cfg)

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
            "auto": "auto",
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported torch dtype: {name}")
        return mapping[normalized]

    def _pipeline_kwargs(self) -> dict[str, Any]:
        from_pretrained_cfg = dict(self.load_cfg.get("model_kwargs", {}))
        for key in ("cache_dir", "revision", "token"):
            value = self.load_cfg.get(key)
            if value is not None:
                from_pretrained_cfg[key] = value
        from_pretrained_cfg["local_files_only"] = self.local_files_only
        # Keep dtype inside ``model_kwargs`` for compatibility with both the
        # older ``torch_dtype`` and newer ``dtype`` Transformers APIs.
        from_pretrained_cfg["torch_dtype"] = self._resolve_dtype(self.torch_dtype)

        kwargs: dict[str, Any] = {
            "model": self.model_path,
            "trust_remote_code": self.trust_remote_code,
            "model_kwargs": from_pretrained_cfg,
        }
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        else:
            kwargs["device"] = self.device
        return kwargs

    def _build_pipeline(self) -> Any:
        transformers = importlib.import_module("transformers")
        pipeline = getattr(transformers, "pipeline")
        return pipeline("image-text-to-text", **self._pipeline_kwargs())

    def _ensure_pipeline(self) -> None:
        if self.pipeline is not None:
            return
        if self.pipeline_factory is not None:
            self.pipeline = self.pipeline_factory(self.spec, dict(self.load_cfg))
        else:
            self.pipeline = self._build_pipeline()

    def _format_prompt(self, prompt: str) -> str:
        if self.spec.prompt_style != "paligemma":
            return prompt

        # PaliGemma 2 mix checkpoints use task prefixes. Preserve an explicit
        # prefix so callers can request OCR, detection, or another task.
        prefixes = (
            "cap ",
            "caption ",
            "describe ",
            "ocr",
            "answer ",
            "question ",
            "detect ",
            "segment ",
        )
        if prompt.lower().startswith(prefixes):
            return prompt
        return f"answer en {prompt}"

    def _messages(self, prompt: str | None, images: list[str]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "image", "url": image_path}
            for image_path in images
        ]
        content.append(
            {"type": "text", "text": self._format_prompt(prompt or "Describe the image.")}
        )
        return [{"role": "user", "content": content}]

    @staticmethod
    def _extract_text(result: Any) -> str:
        """Normalize common Transformers pipeline response shapes."""

        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            for key in ("generated_text", "text", "answer", "content"):
                if key in result:
                    return TransformersVLMBackbone._extract_text(result[key])
            return ""
        if isinstance(result, (list, tuple)):
            if not result:
                return ""
            # Chat pipelines can return the complete conversation. Prefer the
            # final assistant message over the original user prompt.
            if all(isinstance(item, dict) for item in result):
                assistant_messages = [
                    item for item in result if item.get("role") == "assistant"
                ]
                if assistant_messages:
                    return TransformersVLMBackbone._extract_text(assistant_messages[-1])
            if len(result) == 1:
                return TransformersVLMBackbone._extract_text(result[0])
            return TransformersVLMBackbone._extract_text(result[-1])
        return str(result).strip()

    def understanding(
        self,
        prompt: str | None,
        images: list[str],
        videos: list[str],
        understanding_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        if videos:
            raise NotImplementedError(
                f"{self.name} currently supports image understanding only; video inputs are not enabled."
            )
        if not prompt and not images:
            raise ValueError("Understanding requires at least one prompt or image.")

        self._ensure_pipeline()
        assert self.pipeline is not None
        generation_cfg = dict(self.spec.default_generation_cfg)
        generation_cfg.update(understanding_cfg)
        result = self.pipeline(text=self._messages(prompt, images), **generation_cfg)
        return {
            "text": self._extract_text(result),
            "model": self.name,
            "model_path": self.model_path,
        }

    def unload(self) -> None:
        self.pipeline = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    close = unload
