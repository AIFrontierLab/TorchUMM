from __future__ import annotations

from typing import Any

from umm.core import registry
from umm.inference.batcher import batch_iter
from umm.inference.generation import run_editing, run_generation, run_understanding
from umm.inference.multimodal_inputs import InferenceRequest, normalize_request


def register_builtin_backbones() -> None:
    # Register builtins lazily to avoid heavy imports when unused.
    if "bagel" not in registry.list_registered("backbone"):
        from umm.backbones.bagel import BagelBackbone

        registry.register("backbone", "bagel", BagelBackbone)
    if "janus_pro" not in registry.list_registered("backbone"):
        from umm.backbones.janus_pro import JanusProBackbone

        registry.register("backbone", "janus_pro", JanusProBackbone)
    if "show_o" not in registry.list_registered("backbone"):
        from umm.backbones.show_o import ShowOBackbone

        registry.register("backbone", "show_o", ShowOBackbone)
    if "show_o2" not in registry.list_registered("backbone"):
        from umm.backbones.show_o import ShowOBackbone

        registry.register("backbone", "show_o2", ShowOBackbone)
    if "emu3" not in registry.list_registered("backbone"):
        from umm.backbones.emu3 import Emu3Backbone

        registry.register("backbone", "emu3", Emu3Backbone)
    if "omnigen2" not in registry.list_registered("backbone"):
        from umm.backbones.omnigen2 import OmniGen2Backbone

        registry.register("backbone", "omnigen2", OmniGen2Backbone)
    if "blip3o" not in registry.list_registered("backbone"):
        from umm.backbones.blip3o import Blip3oBackbone

        registry.register("backbone", "blip3o", Blip3oBackbone)
    if "tokenflow" not in registry.list_registered("backbone"):
        from umm.backbones.tokenflow import TokenFlowBackbone

        registry.register("backbone", "tokenflow", TokenFlowBackbone)
    if "deepgen" not in registry.list_registered("backbone"):
        from umm.backbones.deepgen import DeepGenBackbone

        registry.register("backbone", "deepgen", DeepGenBackbone)
    if "emu3_5" not in registry.list_registered("backbone"):
        from umm.backbones.emu3_5 import Emu3dot5Backbone

        registry.register("backbone", "emu3_5", Emu3dot5Backbone)
    if "janus_flow" not in registry.list_registered("backbone"):
        from umm.backbones.janus_flow import JanusFlowBackbone

        registry.register("backbone", "janus_flow", JanusFlowBackbone)
    if "mmada" not in registry.list_registered("backbone"):
        from umm.backbones.mmada import MMaDABackbone

        registry.register("backbone", "mmada", MMaDABackbone)
    if "ovis_u1" not in registry.list_registered("backbone"):
        from umm.backbones.ovis_u1 import OvisU1Backbone

        registry.register("backbone", "ovis_u1", OvisU1Backbone)


class InferencePipeline:
    def __init__(self, backbone_name: str, backbone_cfg: dict[str, Any] | None = None) -> None:
        register_builtin_backbones()
        self.backbone_name = backbone_name
        self.backbone_cfg = backbone_cfg or {}
        self.backbone = self._build_backbone(backbone_name)
        self._load_backbone()

    def _build_backbone(self, backbone_name: str) -> Any:
        try:
            factory = registry.get("backbone", backbone_name)
        except KeyError as exc:
            available = registry.list_registered("backbone")
            raise KeyError(
                f"Unknown backbone `{backbone_name}`. Registered backbones: {available}"
            ) from exc
        return factory()

    def _load_backbone(self) -> None:
        if hasattr(self.backbone, "load"):
            self.backbone.load(self.backbone_cfg)

    def run(self, payload: InferenceRequest | dict[str, Any]) -> Any:
        request = normalize_request(payload)
        if request.backbone != self.backbone_name:
            raise ValueError(
                f"Pipeline backbone is `{self.backbone_name}` but request asks for `{request.backbone}`."
            )

        batch = request.to_batch()
        if request.output_path:
            batch["output_path"] = request.output_path

        if request.task == "generation":
            return run_generation(self.backbone, batch, request.params)
        if request.task == "editing":
            return run_editing(self.backbone, batch, request.params)
        if request.task == "understanding":
            return run_understanding(self.backbone, batch, request.params)
        raise ValueError(f"Unsupported task: {request.task}")

    def run_many(
        self,
        payloads: list[InferenceRequest | dict[str, Any]],
        batch_size: int = 1,
    ) -> list[Any]:
        results: list[Any] = []
        for group in batch_iter(payloads, batch_size=batch_size):
            for payload in group:
                results.append(self.run(payload))
        return results
