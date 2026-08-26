from __future__ import annotations

import importlib
import pkgutil
from functools import partial
from typing import Any, Callable, Mapping

from .adapter import DiffusersTextToImageBackbone
from .specs import TextToImageModelSpec
from umm.core import registry


BackboneFactory = Callable[[], Any]


def factories_from_specs(
    specs: Mapping[str, TextToImageModelSpec],
) -> dict[str, BackboneFactory]:
    """Turn model declarations into lazy generic-backbone factories."""

    factories: dict[str, BackboneFactory] = {}
    for name, spec in specs.items():
        if name != spec.name:
            raise ValueError(f"Integration key `{name}` does not match spec name `{spec.name}`.")
        factories[name] = partial(DiffusersTextToImageBackbone, spec)
    return factories


def discover_backbone_factories() -> dict[str, BackboneFactory]:
    """Discover drop-in modules below ``extensions.text2image.integrations``.

    An integration module exports a ``BACKBONES`` mapping. Most modules build
    that mapping with :func:`factories_from_specs`; non-Diffusers runtimes may
    provide their own zero-argument backbone factories while keeping the same
    TorchUMM interface.
    """

    package = importlib.import_module("extensions.text2image.integrations")
    factories: dict[str, BackboneFactory] = {}
    for module_info in pkgutil.iter_modules(package.__path__, f"{package.__name__}."):
        module = importlib.import_module(module_info.name)
        module_factories = getattr(module, "BACKBONES", {})
        if not isinstance(module_factories, Mapping):
            raise TypeError(f"{module_info.name}.BACKBONES must be a mapping.")
        duplicates = factories.keys() & module_factories.keys()
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise ValueError(f"Duplicate text-to-image backbone names: {names}")
        factories.update(module_factories)
    return factories


def register_discovered_backbones() -> None:
    """Register every installed text-to-image integration lazily."""

    registered = set(registry.list_registered("backbone"))
    for name, factory in discover_backbone_factories().items():
        if name not in registered:
            registry.register("backbone", name, factory)
