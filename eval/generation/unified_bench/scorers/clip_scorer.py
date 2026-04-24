"""CLIP vision-encoder similarity (ported from model/UAE/Unified-Bench/CLIP.py).

Uses CLIPImageProcessor directly (not CLIPProcessor) and explicitly moves
pixel_values to the target device. This avoids issues with BatchFeature.to()
on newer transformers versions where passing only images through the combined
CLIPProcessor can emit deprecation warnings / silently break.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel


class CLIPScorer:
    def __init__(self, model_path: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model_path = model_path
        print(f"[clip] loading from {model_path}")
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(model_path)
        self.model_dtype = next(self.model.parameters()).dtype

    def _to_pil(self, image):
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB") if image.mode != "RGB" else image
        raise ValueError(f"CLIPScorer expects PIL.Image or path, got {type(image)}")

    @torch.no_grad()
    def encode_image(self, image):
        pil = self._to_pil(image)
        processed = self.image_processor(images=pil, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device=self.device, dtype=self.model_dtype)
        # Bypass CLIPModel.get_image_features — in recent transformers versions
        # it can return the full BaseModelOutputWithPooling dataclass instead
        # of the projected features tensor. Call vision_model + visual_projection
        # directly to stay stable across versions.
        vision_out = self.model.vision_model(pixel_values=pixel_values)
        pooled = vision_out.pooler_output  # (batch, hidden_size)
        features = self.model.visual_projection(pooled)  # (batch, projection_dim)
        return F.normalize(features, p=2, dim=1)

    def score_pair(self, ref_image, gen_image) -> float:
        f1 = self.encode_image(ref_image)
        f2 = self.encode_image(gen_image)
        return torch.cosine_similarity(f1, f2, dim=1).item()
