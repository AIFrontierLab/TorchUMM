"""Batched Qwen2.5-VL inference through vLLM for local benchmark judges."""

from __future__ import annotations

import os
from typing import Any


def ensure_flash_attn_compatibility() -> None:
    """Bridge flash-attn 2.8's optional wrapper on the Torch 2.5 runtime."""
    import torch

    if not hasattr(torch.library, "wrap_triton"):
        torch.library.wrap_triton = lambda kernel: kernel  # type: ignore[attr-defined]


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style image_url parts to Qwen's multimodal message form."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if role != "user":
            if isinstance(content, list):
                content = " ".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            normalized.append({"role": role, "content": str(content)})
            continue

        parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "image_url":
                    image_url = part.get("image_url", {})
                    if isinstance(image_url, dict) and image_url.get("url"):
                        parts.append({"type": "image", "image": image_url["url"]})
                elif part_type in {"image", "text"}:
                    parts.append(dict(part))
        normalized.append({"role": "user", "content": parts})
    return normalized


class Qwen25VLVLLM:
    """One persistent tensor-parallel vLLM engine with batched multimodal calls."""

    def __init__(
        self,
        model_path: str,
        *,
        tensor_parallel_size: int = 4,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.92,
        max_num_seqs: int = 4,
        max_image_pixels: int | None = None,
    ) -> None:
        ensure_flash_attn_compatibility()
        from transformers import AutoProcessor
        from vllm import LLM

        external_launcher = int(os.environ.get("WORLD_SIZE", "1")) > 1
        self.processor = AutoProcessor.from_pretrained(model_path)
        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "limit_mm_per_prompt": {"image": 1, "video": 0},
            "disable_custom_all_reduce": True,
            "disable_log_stats": True,
            "distributed_executor_backend": ("external_launcher" if external_launcher else None),
        }
        if max_image_pixels is not None and max_image_pixels > 0:
            llm_kwargs["mm_processor_kwargs"] = {"max_pixels": max_image_pixels}
        self.llm = LLM(**llm_kwargs)

    def generate(
        self,
        messages_batch: list[list[dict[str, Any]]],
        *,
        max_tokens: int,
    ) -> list[str]:
        from qwen_vl_utils import process_vision_info
        from vllm import SamplingParams

        requests: list[dict[str, Any]] = []
        for messages in messages_batch:
            normalized = normalize_messages(messages)
            prompt = self.processor.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(normalized)
            if video_inputs:
                raise ValueError("The local judge accepts images only")
            if not image_inputs or len(image_inputs) != 1:
                raise ValueError("Each local judge request must contain exactly one image")
            requests.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_inputs[0]},
                }
            )

        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
        )
        outputs = self.llm.generate(requests, sampling_params=sampling, use_tqdm=False)
        return [output.outputs[0].text.strip() for output in outputs]
