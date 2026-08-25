from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TextToImageModelSpec:
    """Model-agnostic declaration consumed by the shared Diffusers adapter.

    Concrete integrations should define their own spec next to their backbone
    registration. Keeping model IDs and defaults out of this module lets model
    integrations be reviewed and submitted independently.
    """

    name: str
    model_id: str
    pipeline_class: str = "DiffusionPipeline"
    default_generation_cfg: dict[str, Any] = field(default_factory=dict)
    default_load_cfg: dict[str, Any] = field(default_factory=dict)
