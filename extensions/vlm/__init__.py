"""Opt-in Transformers vision-language extensions for TorchUMM."""

from .adapter import TransformersVLMBackbone
from .registration import factories_from_specs, register_discovered_backbones
from .specs import VLMModelSpec


def register() -> None:
    """Register the installed vision-language backbones with TorchUMM."""

    register_discovered_backbones()


__all__ = [
    "TransformersVLMBackbone",
    "VLMModelSpec",
    "factories_from_specs",
    "register",
    "register_discovered_backbones",
]
