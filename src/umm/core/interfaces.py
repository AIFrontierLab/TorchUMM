from __future__ import annotations

from typing import Any, Protocol


class BackboneAdapter(Protocol):
    name: str

    def load(self, cfg: dict[str, Any]) -> None:
        ...

    def encode(self, batch: dict[str, Any]) -> Any:
        ...

    def generate(self, batch: dict[str, Any], gen_cfg: dict[str, Any]) -> Any:
        ...


class Evaluator(Protocol):
    task_name: str

    def run(self, model: Any, dataloader: Any, cfg: dict[str, Any]) -> dict[str, Any]:
        ...


class PostTrainer(Protocol):
    method_name: str

    def fit(self, model: Any, train_dl: Any, val_dl: Any, cfg: dict[str, Any]) -> dict[str, Any]:
        ...
