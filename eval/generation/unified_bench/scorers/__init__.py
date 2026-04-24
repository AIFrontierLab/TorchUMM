"""
Similarity scorers for Unified-Bench.

Each scorer is a class with:
  - __init__(model_path: str, device: str = "cuda")
  - score_pair(ref_image, gen_image) -> float
  - encode_image(image) -> torch.Tensor  (optional, enables batch scoring)

All scorers return cosine similarity in [-1, 1], higher is better.

Ported from model/UAE/Unified-Bench/{CLIP,DINO_v2,DINO_v3,LongCLIP}.py with
hardcoded paths removed (paths now come from the eval config).
"""
from __future__ import annotations

from typing import Any, Dict


def load_scorer(name: str, model_path: str, device: str = "cuda"):
    """Factory: instantiate a scorer by name. Raises ImportError if deps missing."""
    name_lower = name.strip().lower()
    if name_lower == "clip":
        from .clip_scorer import CLIPScorer
        return CLIPScorer(model_path=model_path, device=device)
    if name_lower in ("dinov2", "dino_v2", "dino-v2"):
        from .dinov2_scorer import DINOv2Scorer
        return DINOv2Scorer(model_path=model_path, device=device)
    if name_lower in ("dinov3", "dino_v3", "dino-v3"):
        from .dinov3_scorer import DINOv3Scorer
        return DINOv3Scorer(model_path=model_path, device=device)
    if name_lower in ("longclip", "long_clip", "long-clip"):
        from .longclip_scorer import LongCLIPScorer
        return LongCLIPScorer(model_path=model_path, device=device)
    raise ValueError(f"Unknown scorer '{name}'. Supported: clip, dinov2, dinov3, longclip")
