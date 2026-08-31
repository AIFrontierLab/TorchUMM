from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VLMModelSpec:
    """Model declaration consumed by the shared Transformers VLM adapter."""

    name: str
    model_id: str
    trust_remote_code: bool = False
    prompt_style: str = "chat"
    default_generation_cfg: dict[str, Any] = field(default_factory=dict)
    default_load_cfg: dict[str, Any] = field(default_factory=dict)
