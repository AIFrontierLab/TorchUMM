from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


PATHS = ["direct", "l0", "l1", "l2", "l3"]
LOW_COST_ORDER = PATHS


def slice_bagel_features(features: torch.Tensor, mode: str) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature tensor, got shape={tuple(features.shape)}")
    if features.shape[1] % 3 != 0:
        raise ValueError(f"Expected [text_last, text_mean, image_mean], got dim={features.shape[1]}")
    hidden = features.shape[1] // 3
    text_last = features[:, :hidden]
    text_mean = features[:, hidden : 2 * hidden]
    image_mean = features[:, 2 * hidden :]
    if mode == "full":
        return features
    if mode == "joint_text_mean_image":
        return torch.cat([text_mean, image_mean], dim=1)
    if mode == "joint_text_mean_last_image":
        return torch.cat([text_mean, text_last, image_mean], dim=1)
    raise ValueError(f"Unknown bagel feature mode: {mode}")


def prepare_feature_tensor(payload: dict[str, Any], mode: str) -> torch.Tensor:
    raw_features = payload["features"].float()
    feature_layout = str(payload.get("feature_layout") or "")
    if feature_layout in {"all_abs", "direct_plus_delta"}:
        return raw_features
    return slice_bagel_features(raw_features, mode)


class PlannerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 5) -> None:
        super().__init__()
        self.planner_type = "mlp"
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BPRPathPlanner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 5) -> None:
        super().__init__()
        self.planner_type = "bpr"
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.path_embeddings = nn.Parameter(torch.empty(output_dim, hidden_dim))
        self.path_bias = nn.Parameter(torch.zeros(output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.path_embeddings)
        nn.init.zeros_(self.path_bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.proj(self.input_norm(x)), dim=-1)

    def forward_scores(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        path_matrix = nn.functional.normalize(self.path_embeddings, dim=-1)
        return z @ path_matrix.T + self.path_bias

    def forward_probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward_scores(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_scores(x)


class GateBPRPlanner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 5) -> None:
        super().__init__()
        if output_dim != 5:
            raise ValueError(f"gate_bpr planner expects output_dim=5, got {output_dim}")
        self.planner_type = "gate_bpr"
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.direct_head = nn.Linear(hidden_dim, 1)
        self.nondirect_embeddings = nn.Parameter(torch.empty(4, hidden_dim))
        self.nondirect_bias = nn.Parameter(torch.zeros(4))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.nondirect_embeddings)
        nn.init.zeros_(self.direct_head.weight)
        nn.init.zeros_(self.direct_head.bias)
        nn.init.zeros_(self.nondirect_bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.proj(self.input_norm(x)), dim=-1)

    def forward_components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        direct_logits = self.direct_head(z).squeeze(-1)
        nondirect_matrix = nn.functional.normalize(self.nondirect_embeddings, dim=-1)
        nondirect_scores = z @ nondirect_matrix.T + self.nondirect_bias
        return direct_logits, nondirect_scores

    def forward_probs(self, x: torch.Tensor) -> torch.Tensor:
        direct_logits, nondirect_scores = self.forward_components(x)
        p_direct = torch.sigmoid(direct_logits)
        p_levels = torch.softmax(nondirect_scores, dim=-1)
        return torch.cat([p_direct.unsqueeze(-1), (1.0 - p_direct).unsqueeze(-1) * p_levels], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logit(self.forward_probs(x).clamp(1e-6, 1.0 - 1e-6))


class TwoStagePlanner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.planner_type = "two_stage"
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.direct_head = nn.Linear(hidden_dim, 1)
        self.level_head = nn.Linear(hidden_dim, 4)

    def forward_components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        return self.direct_head(hidden).squeeze(-1), self.level_head(hidden)

    def forward_probs(self, x: torch.Tensor) -> torch.Tensor:
        direct_logits, level_logits = self.forward_components(x)
        p_direct = torch.sigmoid(direct_logits)
        p_levels = torch.softmax(level_logits, dim=-1)
        return torch.cat([p_direct.unsqueeze(-1), (1.0 - p_direct).unsqueeze(-1) * p_levels], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logit(self.forward_probs(x).clamp(1e-6, 1.0 - 1e-6))


def build_planner_model(planner_type: str, input_dim: int, hidden_dim: int, output_dim: int = 5) -> nn.Module:
    if planner_type == "mlp":
        return PlannerMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    if planner_type == "bpr":
        return BPRPathPlanner(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    if planner_type == "gate_bpr":
        return GateBPRPlanner(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    if planner_type == "two_stage":
        if output_dim != 5:
            raise ValueError(f"two_stage planner expects output_dim=5, got {output_dim}")
        return TwoStagePlanner(input_dim=input_dim, hidden_dim=hidden_dim)
    raise ValueError(f"Unknown planner_type: {planner_type}")


def planner_type_of(model: nn.Module) -> str:
    planner_type = getattr(model, "planner_type", "")
    return str(planner_type) if planner_type else "mlp"


def planner_probs(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_probs"):
        return model.forward_probs(x)
    return torch.sigmoid(model(x))


def choose_path_for_model(model: nn.Module, scores: torch.Tensor, margin: float) -> int:
    best = float(scores.max().item())
    candidates = [idx for idx, score in enumerate(scores.tolist()) if score >= best - margin]
    order = {path_name: idx for idx, path_name in enumerate(LOW_COST_ORDER)}
    return min(candidates, key=lambda idx: order[PATHS[idx]])


def load_planner(path: str | Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu")
    paths = list(payload.get("paths") or PATHS)
    if paths != PATHS:
        raise ValueError(f"Planner paths mismatch: expected {PATHS}, got {paths}")
    model = build_planner_model(
        str(payload.get("planner_type") or "mlp"),
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        output_dim=len(PATHS),
    )
    state = payload.get("model_state") or payload
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, payload
