from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Stage1Config:
    text_dim: int
    image_dim: int
    latent_dim: int
    hidden_dim: int = 1024
    output_dim: int = 0
    dropout: float = 0.1
    transformer_layers: int = 2
    transformer_heads: int = 8


class DualModalAligner(nn.Module):
    """
    Phi encoder for text/image Gemini embeddings.

    The latent is produced by a shared Transformer backbone. Single-modality
    calls encode text or image into the shared latent space. Paired calls fuse
    both modality tokens into the unified z used by downstream task heads.
    """

    def __init__(self, cfg: Stage1Config) -> None:
        super().__init__()
        self.text_proj = nn.Linear(cfg.text_dim, cfg.hidden_dim)
        self.image_proj = nn.Linear(cfg.image_dim, cfg.hidden_dim)
        self.modality_embed = nn.Parameter(torch.zeros(2, cfg.hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.single_to_latent = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.fuse = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def _encode_sequence(
        self,
        text_emb: Optional[torch.Tensor] = None,
        image_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = []
        if text_emb is not None:
            tokens.append(self.text_proj(text_emb) + self.modality_embed[0])
        if image_emb is not None:
            tokens.append(self.image_proj(image_emb) + self.modality_embed[1])
        if not tokens:
            raise ValueError("At least one modality embedding must be provided")
        sequence = torch.stack(tokens, dim=1)
        return self.norm(self.backbone(sequence))

    def encode_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        hidden = self._encode_sequence(text_emb=text_emb)[:, 0]
        return self.single_to_latent(hidden)

    def encode_image(self, image_emb: torch.Tensor) -> torch.Tensor:
        hidden = self._encode_sequence(image_emb=image_emb)[:, 0]
        return self.single_to_latent(hidden)

    def encode_embedding(self, embedding: torch.Tensor, modality: str = "image") -> torch.Tensor:
        if modality == "text":
            return self.encode_text(embedding)
        if modality == "image":
            return self.encode_image(embedding)
        raise ValueError(f"Unsupported modality: {modality}")

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        hidden = self._encode_sequence(text_emb=text_emb, image_emb=image_emb)
        return self.fuse(torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1))


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Stage1SharedLatentModel(nn.Module):
    """
    Stage 1 dual-alignment model.

    z_text = phi(text_embedding)
    z_image = phi(image_embedding)
    L_x-modal = ||z_text - z_image||_2^2

    z -> generator -> generated Gemini-space embedding -> re-encoder -> z_hat
    L_x-task = ||z - z_hat||_2^2
    """

    def __init__(self, cfg: Stage1Config) -> None:
        super().__init__()
        output_dim = cfg.output_dim if cfg.output_dim > 0 else cfg.text_dim
        cfg.output_dim = output_dim
        self.cfg = cfg
        self.aligner = DualModalAligner(cfg)
        self.generator = MLPHead(cfg.latent_dim, cfg.hidden_dim, output_dim, cfg.dropout)
        self.embedding_model = MLPHead(output_dim, cfg.hidden_dim, cfg.latent_dim, cfg.dropout)

    def generate_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def embed_output(self, output: torch.Tensor, modality: str = "image") -> torch.Tensor:
        expected_dim = self.cfg.image_dim if modality == "image" else self.cfg.text_dim
        if output.shape[-1] == expected_dim:
            return self.aligner.encode_embedding(output, modality=modality)
        return self.embedding_model(output)

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_text = self.aligner.encode_text(text_emb)
        z_image = self.aligner.encode_image(image_emb)
        z_student = self.aligner(text_emb, image_emb)
        z_teacher = 0.5 * (z_text.detach() + z_image.detach())

        generated_from_text = self.generate_from_latent(z_text)
        generated_from_fused = self.generate_from_latent(z_student)
        z_hat_text = self.embed_output(generated_from_text, modality="image")
        z_hat_fused = self.embed_output(generated_from_fused, modality="image")

        return {
            "z_text": z_text,
            "z_image": z_image,
            "z_student": z_student,
            "z_teacher": z_teacher,
            "generated_from_text": generated_from_text,
            "generated_from_fused": generated_from_fused,
            "z_hat_text": z_hat_text,
            "z_hat_fused": z_hat_fused,
            "text_image_cosine": F.cosine_similarity(z_text, z_image, dim=-1),
        }
