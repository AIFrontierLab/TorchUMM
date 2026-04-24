"""LongCLIP similarity (ported from model/UAE/Unified-Bench/LongCLIP.py).

LongCLIP weights are a .pt file from BeichenZhang/Long-CLIP. The repo is NOT
a pip package, so it's cloned into /opt/longclip at image build time (see
modal/images.py:_unified_bench_image). We add /opt to sys.path and import
`from longclip.model import longclip`.

If the import fails at runtime, this scorer raises ImportError and the
scoring script will skip it with a warning.
"""
from __future__ import annotations

import os
import sys

import torch
from torch.nn import functional as F
from PIL import Image


class LongCLIPScorer:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model_path = model_path
        print(f"[longclip] loading from {model_path}")

        # Allow overriding the clone location via env var, default /opt.
        longclip_root = os.environ.get("LONGCLIP_ROOT", "/opt")
        if longclip_root not in sys.path:
            sys.path.insert(0, longclip_root)

        try:
            from longclip.model import longclip
        except ImportError as exc:
            raise ImportError(
                f"LongCLIP module not found at {longclip_root}/longclip. "
                "Expected repo cloned from https://github.com/beichenzbc/Long-CLIP "
                "(done in modal/images.py:_unified_bench_image)."
            ) from exc

        self._longclip = longclip
        self.model, self.tform = longclip.load(model_path, device=self.device)
        self.model.eval()

    def _to_pil(self, image):
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB") if image.mode != "RGB" else image
        raise ValueError(f"LongCLIPScorer expects PIL.Image or path, got {type(image)}")

    @torch.no_grad()
    def encode_image(self, image):
        pil = self._to_pil(image)
        tensor = self.tform(pil).unsqueeze(0).to(device=self.device, dtype=self.model.dtype)
        features = self.model.encode_image(tensor)
        return F.normalize(features, p=2, dim=1)

    def score_pair(self, ref_image, gen_image) -> float:
        f1 = self.encode_image(ref_image)
        f2 = self.encode_image(gen_image)
        return torch.cosine_similarity(f1, f2, dim=1).item()
