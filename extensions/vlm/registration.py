from __future__ import annotations

import importlib
import pkgutil
from functools import partial
from typing import Any, Callable, Mapping

from .adapter import TransformersVLMBackbone
from .specs import VLMModelSpec
from umm.core import registry


BackboneFactory = Callable[[], Any]


def factories_from_specs(
    specs: Mapping[str, VLMModelSpec],
) -> dict[str, BackboneFactory]:
    factories: dict[str, BackboneFactory] = {}
    for name, spec in specs.items():
        if name != spec.name:
            raise ValueError(f"Integration key `{name}` does not match spec name `{spec.name}`.")
        factories[name] = partial(TransformersVLMBackbone, spec)
    return factories


def discover_backbone_factories() -> dict[str, BackboneFactory]:
    package = importlib.import_module("extensions.vlm.integrations")
    factories: dict[str, BackboneFactory] = {}
    for module_info in pkgutil.iter_modules(package.__path__, f"{package.__name__}."):
        module = importlib.import_module(module_info.name)
        module_factories = getattr(module, "BACKBONES", {})
        if not isinstance(module_factories, Mapping):
            raise TypeError(f"{module_info.name}.BACKBONES must be a mapping.")
        duplicates = factories.keys() & module_factories.keys()
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise ValueError(f"Duplicate Transformers VLM backbone names: {names}")
        factories.update(module_factories)
    return factories


def register_discovered_backbones() -> None:
    registered = set(registry.list_registered("backbone"))
    for name, factory in discover_backbone_factories().items():
        if name not in registered:
            registry.register("backbone", name, factory)
