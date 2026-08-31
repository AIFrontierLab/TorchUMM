"""Optional Transformers-backed vision-language model integrations."""

from umm.backbones.transformers_vlm.adapter import TransformersVLMBackbone
from umm.backbones.transformers_vlm.registration import (
    factories_from_specs,
    register_discovered_backbones,
)
from umm.backbones.transformers_vlm.specs import VLMModelSpec

__all__ = [
    "TransformersVLMBackbone",
    "VLMModelSpec",
    "factories_from_specs",
    "register_discovered_backbones",
]
