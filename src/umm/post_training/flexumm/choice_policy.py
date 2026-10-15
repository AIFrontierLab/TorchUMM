"""Benchmark-agnostic consensus routing for explicit-choice prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ChoiceConsensusConfig:
    """One global late-depth policy shared by every explicit-choice input."""

    candidate_depths: tuple[int, ...] = (24, 26, 28)
    depth_weights: tuple[float, ...] = (0.15, 0.30, 0.55)
    early_exit_margin: float = 1.25
    early_exit_max_entropy: float = 0.55
    require_depth_agreement: bool = True

    @classmethod
    def from_dict(cls, value: dict | None) -> "ChoiceConsensusConfig":
        if not value:
            return cls()
        candidate_depths = tuple(
            int(item)
            for item in value.get("candidate_depths", cls.candidate_depths)
        )
        depth_weights = tuple(
            float(item)
            for item in value.get("depth_weights", cls.depth_weights)
        )
        config = cls(
            candidate_depths=candidate_depths,
            depth_weights=depth_weights,
            early_exit_margin=float(
                value.get("early_exit_margin", cls.early_exit_margin)
            ),
            early_exit_max_entropy=float(
                value.get(
                    "early_exit_max_entropy",
                    cls.early_exit_max_entropy,
                )
            ),
            require_depth_agreement=bool(
                value.get(
                    "require_depth_agreement",
                    cls.require_depth_agreement,
                )
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.candidate_depths:
            raise ValueError("candidate_depths must not be empty")
        if any(depth <= 0 for depth in self.candidate_depths):
            raise ValueError("candidate_depths must be positive")
        if tuple(sorted(set(self.candidate_depths))) != self.candidate_depths:
            raise ValueError(
                "candidate_depths must be strictly increasing and unique"
            )
        if len(self.depth_weights) != len(self.candidate_depths):
            raise ValueError(
                "depth_weights must have one entry per candidate depth"
            )
        if any(weight < 0 for weight in self.depth_weights):
            raise ValueError("depth_weights must be non-negative")
        if sum(self.depth_weights) <= 0:
            raise ValueError("at least one depth weight must be positive")
        if self.early_exit_margin < 0:
            raise ValueError("early_exit_margin must be non-negative")
        if not 0 <= self.early_exit_max_entropy <= 1:
            raise ValueError("early_exit_max_entropy must be in [0, 1]")


def aggregate_choice_log_probs(
    log_probs: Sequence[torch.Tensor],
    weights: Sequence[float],
) -> torch.Tensor:
    """Return a normalized weighted log-probability ensemble."""

    if not log_probs:
        raise ValueError("log_probs must not be empty")
    if len(log_probs) != len(weights):
        raise ValueError("weights must match log_probs")
    reference_shape = log_probs[0].shape
    if any(value.shape != reference_shape for value in log_probs):
        raise ValueError("all choice log-probability tensors must match")
    weight_tensor = torch.as_tensor(
        weights,
        device=log_probs[0].device,
        dtype=torch.float32,
    )
    if torch.any(weight_tensor < 0) or weight_tensor.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    weight_tensor = weight_tensor / weight_tensor.sum()
    stacked = torch.stack(
        [value.float() for value in log_probs],
        dim=0,
    )
    combined = (stacked * weight_tensor[:, None]).sum(dim=0)
    return combined - torch.logsumexp(combined, dim=-1)


def choice_confidence(log_probs: torch.Tensor) -> tuple[float, float]:
    """Return top-two log margin and entropy normalized to [0, 1]."""

    if log_probs.ndim != 1 or log_probs.numel() < 2:
        raise ValueError("choice log_probs must be a vector with >=2 entries")
    normalized = log_probs.float() - torch.logsumexp(log_probs.float(), dim=-1)
    top_two = torch.topk(normalized, k=2).values
    margin = float((top_two[0] - top_two[1]).item())
    probabilities = normalized.exp()
    entropy = -(probabilities * normalized).sum()
    max_entropy = torch.log(
        torch.tensor(
            float(normalized.numel()),
            device=normalized.device,
        )
    )
    normalized_entropy = float((entropy / max_entropy).item())
    return margin, normalized_entropy


def should_exit_choice(
    per_depth_log_probs: Sequence[torch.Tensor],
    aggregate_log_probs: torch.Tensor,
    *,
    margin_threshold: float,
    max_entropy: float,
    require_agreement: bool,
) -> bool:
    """Conservatively stop when late-depth evidence agrees and is confident."""

    if not per_depth_log_probs:
        return False
    if require_agreement:
        if len(per_depth_log_probs) < 2:
            return False
        if per_depth_log_probs[-1].argmax().item() != per_depth_log_probs[
            -2
        ].argmax().item():
            return False
    margin, entropy = choice_confidence(aggregate_log_probs)
    return margin >= margin_threshold and entropy <= max_entropy
