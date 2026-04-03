from __future__ import annotations

from collections.abc import Callable
from typing import Any


_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {
    "backbone": {},
    "evaluator": {},
    "post_trainer": {},
}


def register(kind: str, name: str, factory: Callable[..., Any]) -> None:
    if kind not in _REGISTRY:
        raise KeyError(f"Unknown registry kind: {kind}")
    _REGISTRY[kind][name] = factory


def get(kind: str, name: str) -> Callable[..., Any]:
    return _REGISTRY[kind][name]


def list_registered(kind: str) -> list[str]:
    return sorted(_REGISTRY[kind])
