"""DINOv2 CLS-token similarity (ported from model/UAE/Unified-Bench/DINO_v2.py)."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DINOv2Scorer:
    def __init__(self, model_path: str = "facebook/dinov2-large", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model_path = model_path
        print(f"[dinov2] loading from {model_path}")
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def _to_pil(self, image):
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB") if image.mode != "RGB" else image
        raise ValueError(f"DINOv2Scorer expects PIL.Image or path, got {type(image)}")

    @torch.no_grad()
    def encode_image(self, image):
        pil = self._to_pil(image)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # CLS token
        features = outputs.last_hidden_state[:, 0, :]
        return F.normalize(features, p=2, dim=1)

    def score_pair(self, ref_image, gen_image) -> float:
        f1 = self.encode_image(ref_image)
        f2 = self.encode_image(gen_image)
        return torch.cosine_similarity(f1, f2, dim=1).item()
