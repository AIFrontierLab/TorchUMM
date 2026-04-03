from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


InferenceTask = Literal["generation", "editing", "understanding"]


@dataclass(slots=True)
class InferenceRequest:
    backbone: str
    task: InferenceTask
    prompt: str | None = None
    images: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    output_path: str | None = None

    def to_batch(self) -> dict[str, Any]:
        batch: dict[str, Any] = {"prompt": self.prompt, "images": self.images, "videos": self.videos}
        if self.metadata:
            batch["metadata"] = self.metadata
        return batch


def normalize_request(payload: InferenceRequest | dict[str, Any]) -> InferenceRequest:
    if isinstance(payload, InferenceRequest):
        request = payload
    else:
        request = InferenceRequest(
            backbone=str(payload["backbone"]),
            task=payload["task"],
            prompt=payload.get("prompt"),
            images=list(payload.get("images", [])),
            videos=list(payload.get("videos", [])),
            params=dict(payload.get("params", {})),
            metadata=dict(payload.get("metadata", {})),
            output_path=payload.get("output_path"),
        )
    validate_request(request)
    return request


def validate_request(request: InferenceRequest) -> None:
    if request.task not in {"generation", "editing", "understanding"}:
        raise ValueError(f"Unsupported task: {request.task}")
    if not request.backbone:
        raise ValueError("`backbone` must be provided.")
    if request.task == "generation" and not request.prompt:
        raise ValueError("`prompt` is required for task `generation`.")
    if request.task == "editing":
        has_text = bool(request.prompt) or bool(request.params.get("instruction"))
        has_image = bool(request.images) or bool(request.params.get("image_path"))
        if not has_text:
            raise ValueError("Editing requires `prompt` or `params.instruction`.")
        if not has_image:
            raise ValueError("Editing requires `images` or `params.image_path`.")
