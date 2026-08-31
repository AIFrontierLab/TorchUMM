"""Backbone exports with lazy imports for model-specific dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BagelBackbone": ("umm.backbones.bagel", "BagelBackbone"),
    "JanusProBackbone": ("umm.backbones.janus_pro", "JanusProBackbone"),
    "ShowOBackbone": ("umm.backbones.show_o", "ShowOBackbone"),
    "Emu3Backbone": ("umm.backbones.emu3", "Emu3Backbone"),
    "Emu3dot5Backbone": ("umm.backbones.emu3_5", "Emu3dot5Backbone"),
    "JanusFlowBackbone": ("umm.backbones.janus_flow", "JanusFlowBackbone"),
    "MMaDABackbone": ("umm.backbones.mmada", "MMaDABackbone"),
    "OvisU1Backbone": ("umm.backbones.ovis_u1", "OvisU1Backbone"),
    "TransformersVLMBackbone": (
        "umm.backbones.transformers_vlm",
        "TransformersVLMBackbone",
    ),
    "VLMModelSpec": ("umm.backbones.transformers_vlm", "VLMModelSpec"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
