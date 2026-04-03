from __future__ import annotations

from typing import Any


def run_generation(backbone: Any, batch: dict[str, Any], cfg: dict[str, Any]) -> Any:
    if hasattr(backbone, "generation"):
        return backbone.generation(
            prompt=batch.get("prompt"),
            output_path=batch.get("output_path"),
            generation_cfg=cfg,
        )
    if hasattr(backbone, "generate"):
        return backbone.generate(batch=batch, gen_cfg=cfg)
    raise NotImplementedError("Backbone does not implement generation (`generation` or `generate`).")


def run_editing(backbone: Any, batch: dict[str, Any], cfg: dict[str, Any]) -> Any:
    if hasattr(backbone, "editing"):
        return backbone.editing(
            prompt=batch.get("prompt"),
            images=batch.get("images", []),
            output_path=batch.get("output_path"),
            editing_cfg=cfg,
        )
    if hasattr(backbone, "edit"):
        return backbone.edit(batch=batch, edit_cfg=cfg)
    raise NotImplementedError("Backbone does not implement editing (`editing` or `edit`).")


def run_understanding(backbone: Any, batch: dict[str, Any], cfg: dict[str, Any]) -> Any:
    if hasattr(backbone, "understanding"):
        return backbone.understanding(
            prompt=batch.get("prompt"),
            images=batch.get("images", []),
            videos=batch.get("videos", []),
            understanding_cfg=cfg,
        )
    if hasattr(backbone, "understand"):
        return backbone.understand(batch=batch, understanding_cfg=cfg)
    if hasattr(backbone, "encode"):
        return backbone.encode(batch=batch)
    raise NotImplementedError(
        "Backbone does not implement understanding (`understanding`, `understand`, or `encode`)."
    )
