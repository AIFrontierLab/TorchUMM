"""Opt-in text-to-image extensions for TorchUMM."""

from .adapter import DiffusersTextToImageBackbone
from .registration import factories_from_specs, register_discovered_backbones
from .specs import TextToImageModelSpec


def register() -> None:
    """Register the installed text-to-image backbones with TorchUMM."""

    register_discovered_backbones()


__all__ = [
    "DiffusersTextToImageBackbone",
    "TextToImageModelSpec",
    "factories_from_specs",
    "register",
    "register_discovered_backbones",
]
