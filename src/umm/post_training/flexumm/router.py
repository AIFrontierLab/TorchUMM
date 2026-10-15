"""Small input-adaptive halting router for candidate transformer exits."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


class CandidateDepthRouter(nn.Module):
    """Predict a halting logit from an intermediate shared-backbone state.

    The router is intentionally small relative to BAGEL. It shares one MLP
    across all exits. The benchmark-agnostic mode conditions on representation
    state, change since the preceding exit, candidate depth, route kind, and
    diffusion timestep, but not a task or dataset name. Understanding uses
    learned halting; generation can continue to use the deterministic schedule.
    """

    TASK_ALIASES: dict[str, str] = {
        "mmbench_test_en_v11": "mmbench_en",
        "mmbench_test_cn_v11": "mmbench_cn",
        "mmbench-test-en-v11": "mmbench_en",
        "mmbench-test-cn-v11": "mmbench_cn",
        "mmmu_val": "mmmu",
        "mmmu-val": "mmmu",
        "mathvista_testmini": "mathvista",
        "mathvista-testmini": "mathvista",
    }

    def __init__(
        self,
        *,
        backbone_hidden_size: int,
        router_hidden_size: int,
        num_layers: int,
        task_names: Sequence[str],
        initial_bias: float = -2.0,
        condition_on_task: bool = True,
        use_state_drift: bool = False,
        use_last_token: bool = False,
        feature_mode: str = "full",
        compute_in_float32: bool = False,
    ):
        super().__init__()
        if feature_mode not in {
            "full",
            "hidden_state",
            "representation_change",
        }:
            raise ValueError(
                "feature_mode must be full, hidden_state, or "
                "representation_change"
            )
        canonical_tasks = []
        for task in ("unknown", *task_names):
            normalized = self.normalize_task(task)
            if normalized not in canonical_tasks:
                canonical_tasks.append(normalized)
        self.task_names = tuple(canonical_tasks)
        self.task_to_id = {
            task: index for index, task in enumerate(self.task_names)
        }
        self.num_layers = int(num_layers)
        self.initial_bias = float(initial_bias)
        self.condition_on_task = bool(condition_on_task)
        self.use_state_drift = bool(use_state_drift)
        self.use_last_token = bool(use_last_token)
        self.feature_mode = str(feature_mode)
        self.compute_in_float32 = bool(compute_in_float32)

        self.state_norm = nn.LayerNorm(backbone_hidden_size)
        self.state_proj = nn.Linear(backbone_hidden_size, router_hidden_size)
        self.last_state_proj = (
            nn.Linear(backbone_hidden_size, router_hidden_size)
            if self.use_last_token
            else None
        )
        self.task_embedding = (
            nn.Embedding(len(self.task_names), router_hidden_size)
            if self.condition_on_task
            else None
        )
        self.drift_proj = (
            nn.Linear(backbone_hidden_size, router_hidden_size)
            if self.use_state_drift
            else None
        )
        self.state_statistics_proj = (
            nn.Linear(4, router_hidden_size)
            if self.use_state_drift
            else None
        )
        self.depth_embedding = nn.Embedding(num_layers + 1, router_hidden_size)
        self.kind_embedding = nn.Embedding(2, router_hidden_size)
        self.timestep_proj = nn.Linear(3, router_hidden_size)
        self.output = nn.Sequential(
            nn.SiLU(),
            nn.Linear(router_hidden_size, router_hidden_size),
            nn.SiLU(),
            nn.Linear(router_hidden_size, 1),
        )
        # Predict harmful regret relative to the full-depth candidate and its
        # uncertainty scale. At inference an upper confidence bound provides
        # a benchmark-independent, quality-first guard on early exit.
        self.risk_output = nn.Sequential(
            nn.SiLU(),
            nn.Linear(router_hidden_size, router_hidden_size),
            nn.SiLU(),
            nn.Linear(router_hidden_size, 2),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize a router while preserving its configured structure."""

        self.state_norm.reset_parameters()
        self.state_proj.reset_parameters()
        if self.last_state_proj is not None:
            self.last_state_proj.reset_parameters()
        if self.task_embedding is not None:
            self.task_embedding.reset_parameters()
        if self.drift_proj is not None:
            self.drift_proj.reset_parameters()
        if self.state_statistics_proj is not None:
            self.state_statistics_proj.reset_parameters()
        self.depth_embedding.reset_parameters()
        self.kind_embedding.reset_parameters()
        self.timestep_proj.reset_parameters()
        self.output[1].reset_parameters()
        self.output[3].reset_parameters()
        self.risk_output[1].reset_parameters()
        self.risk_output[3].reset_parameters()
        nn.init.zeros_(self.output[-1].weight)
        nn.init.constant_(self.output[-1].bias, self.initial_bias)
        nn.init.zeros_(self.risk_output[-1].weight)
        with torch.no_grad():
            self.risk_output[-1].bias.copy_(
                self.risk_output[-1].bias.new_tensor([0.0, -2.0])
            )

    @staticmethod
    def normalize_task(task: str | None) -> str:
        normalized = str(task or "unknown").strip().lower().replace("-", "_")
        return CandidateDepthRouter.TASK_ALIASES.get(normalized, normalized)

    def task_id(self, task: str | None) -> int:
        normalized = self.normalize_task(task)
        if normalized in self.task_to_id:
            return self.task_to_id[normalized]
        if normalized in {"mmbench_en", "mmbench_cn"} and "mmbench" in self.task_to_id:
            return self.task_to_id["mmbench"]
        return self.task_to_id["unknown"]

    @staticmethod
    def _pool(
        hidden_states: torch.Tensor,
        *,
        token_indexes: torch.Tensor | None,
        sample_lens: Sequence[int] | None,
        active_sample_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        per_sample = sample_lens is not None
        if per_sample:
            assert sample_lens is not None
            packed_length = hidden_states.shape[0]
            sample_length_sum = sum(int(length) for length in sample_lens)
            if sample_length_sum < packed_length:
                raise ValueError(
                    "sample_lens cannot be shorter than the packed sequence length"
                )
            if token_indexes is None or token_indexes.numel() == 0:
                positions = torch.arange(
                    hidden_states.shape[0], device=hidden_states.device
                )
            elif token_indexes.dtype == torch.bool:
                positions = token_indexes.nonzero(as_tuple=False).flatten()
            else:
                positions = token_indexes.to(
                    device=hidden_states.device, dtype=torch.long
                )
            if active_sample_ids is None:
                active_sample_ids = torch.arange(
                    len(sample_lens), device=hidden_states.device
                )
            else:
                active_sample_ids = active_sample_ids.to(
                    device=hidden_states.device, dtype=torch.long
                )

            offsets = [0]
            for length in sample_lens:
                offsets.append(offsets[-1] + int(length))
            pooled_states = []
            last_states = []
            for sample_id in active_sample_ids.detach().cpu().tolist():
                start = offsets[sample_id]
                end = min(offsets[sample_id + 1], packed_length)
                sample_positions = positions[
                    (positions >= start) & (positions < end)
                ]
                sample_states = (
                    hidden_states[sample_positions]
                    if sample_positions.numel()
                    else hidden_states[start:end]
                )
                if sample_states.numel() == 0:
                    raise ValueError(f"Cannot route empty packed sample {sample_id}")
                pooled_states.append(sample_states.float().mean(dim=0))
                last_states.append(sample_states[-1].float())
            pooled = torch.stack(pooled_states, dim=0)
            last = torch.stack(last_states, dim=0)
        else:
            if token_indexes is not None and token_indexes.numel() > 0:
                hidden_states = hidden_states[token_indexes]
            if hidden_states.numel() == 0:
                raise ValueError("Cannot route an empty hidden-state tensor")
            pooled = hidden_states.float().mean(dim=0, keepdim=True)
            last = hidden_states[-1:].float()

        if pooled.numel() == 0:
            raise ValueError("Cannot route an empty hidden-state tensor")
        return pooled, last, per_sample

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        task: str | Sequence[str] | None,
        depth: int,
        kind: str,
        timestep: float | None,
        token_indexes: torch.Tensor | None = None,
        detach_features: bool = True,
        sample_lens: Sequence[int] | None = None,
        active_sample_ids: torch.Tensor | None = None,
        previous_hidden_states: torch.Tensor | None = None,
        return_risk: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled, last, per_sample = self._pool(
            hidden_states,
            token_indexes=token_indexes,
            sample_lens=sample_lens,
            active_sample_ids=active_sample_ids,
        )
        previous_pooled = None
        previous_last = None
        drift_active = self.use_state_drift and self.feature_mode in {
            "full",
            "representation_change",
        }
        if previous_hidden_states is not None and drift_active:
            previous_pooled, previous_last, previous_per_sample = self._pool(
                previous_hidden_states,
                token_indexes=token_indexes,
                sample_lens=sample_lens,
                active_sample_ids=active_sample_ids,
            )
            if previous_per_sample != per_sample:
                raise ValueError("Current and previous router states must align")

        if detach_features:
            pooled = pooled.detach()
            last = last.detach()
            if previous_pooled is not None:
                previous_pooled = previous_pooled.detach()
            if previous_last is not None:
                previous_last = previous_last.detach()
        parameter = self.state_proj.weight
        compute_dtype = (
            torch.float32 if self.compute_in_float32 else parameter.dtype
        )
        pooled = pooled.to(device=parameter.device, dtype=compute_dtype)
        last = last.to(device=parameter.device, dtype=compute_dtype)
        if previous_pooled is not None:
            previous_pooled = previous_pooled.to(
                device=parameter.device, dtype=compute_dtype
            )
        if previous_last is not None:
            previous_last = previous_last.to(
                device=parameter.device, dtype=compute_dtype
            )
        batch_size = pooled.shape[0]

        depth_ids = torch.tensor(
            [int(depth)], device=parameter.device, dtype=torch.long
        ).expand(batch_size)
        kind_id = 1 if kind == "generation" else 0
        kind_ids = torch.tensor(
            [kind_id], device=parameter.device, dtype=torch.long
        ).expand(batch_size)
        if timestep is None:
            t = 0.0
        elif isinstance(timestep, torch.Tensor):
            # Diffusion supplies packed, batched timesteps. Keep the router's
            # scalar conditioning contract by matching the controller's min
            # reduction used for dynamic-depth plans.
            values = timestep.detach().float()
            t = 0.0 if values.numel() == 0 else float(values.min().item())
            t = min(max(t, 0.0), 1.0)
        else:
            t = min(max(float(timestep), 0.0), 1.0)
        timestep_features = pooled.new_tensor(
            [[t, math.sin(math.pi * t), math.cos(math.pi * t)]]
        ).expand(batch_size, -1)

        def linear(module: nn.Linear, values: torch.Tensor) -> torch.Tensor:
            if not self.compute_in_float32:
                return module(values)
            return torch.nn.functional.linear(
                values.float(),
                module.weight.float(),
                None if module.bias is None else module.bias.float(),
            )

        def embedding(module: nn.Embedding, indexes: torch.Tensor) -> torch.Tensor:
            if not self.compute_in_float32:
                return module(indexes)
            return torch.nn.functional.embedding(indexes, module.weight.float())

        def layer_norm(module: nn.LayerNorm, values: torch.Tensor) -> torch.Tensor:
            if not self.compute_in_float32:
                return module(values)
            return torch.nn.functional.layer_norm(
                values.float(),
                module.normalized_shape,
                module.weight.float(),
                module.bias.float(),
                module.eps,
            )

        features = (
            embedding(self.depth_embedding, depth_ids)
            + embedding(self.kind_embedding, kind_ids)
            + linear(self.timestep_proj, timestep_features)
        )
        # The ablation masks contributions while retaining the original
        # checkpoint-compatible module shapes. Fixed metadata remains present
        # in every learned-router variant so only the requested signal changes.
        if self.feature_mode in {"full", "hidden_state"}:
            features = features + linear(
                self.state_proj, layer_norm(self.state_norm, pooled)
            )
        if self.last_state_proj is not None and self.feature_mode in {
            "full",
            "hidden_state",
        }:
            features = features + linear(
                self.last_state_proj, layer_norm(self.state_norm, last)
            )
        if self.task_embedding is not None:
            if isinstance(task, str) or task is None:
                task_ids = torch.tensor(
                    [self.task_id(task)], device=parameter.device, dtype=torch.long
                ).expand(batch_size)
            else:
                tasks = list(task)
                if len(tasks) != batch_size:
                    raise ValueError(
                        "per-sample router tasks must match the pooled batch size"
                    )
                task_ids = torch.tensor(
                    [self.task_id(item) for item in tasks],
                    device=parameter.device,
                    dtype=torch.long,
                )
            features = features + embedding(self.task_embedding, task_ids)
        if drift_active:
            assert self.drift_proj is not None
            assert self.state_statistics_proj is not None
            normalized = layer_norm(self.state_norm, pooled)
            if previous_pooled is None:
                previous_normalized = torch.zeros_like(normalized)
                has_previous = torch.zeros(
                    batch_size, device=parameter.device, dtype=compute_dtype
                )
            else:
                previous_normalized = layer_norm(
                    self.state_norm, previous_pooled
                )
                has_previous = torch.ones(
                    batch_size, device=parameter.device, dtype=compute_dtype
                )
            previous_mask = has_previous.unsqueeze(-1)
            # A first candidate has no layer-to-layer comparison. Treating
            # its current state as a drift would leak the hidden-state signal
            # into the representation-change-only condition.
            drift = (normalized - previous_normalized) * previous_mask
            if previous_pooled is None:
                # There is no preceding candidate at the first exit.  Do not
                # evaluate cosine similarity against a zero vector: although
                # its forward value is finite, its backward derivative can be
                # undefined and poison the router's LayerNorm gradients.
                cosine_change = torch.zeros(
                    batch_size,
                    device=parameter.device,
                    dtype=compute_dtype,
                )
            else:
                cosine_change = 1.0 - torch.nn.functional.cosine_similarity(
                    normalized.float(),
                    previous_normalized.float(),
                    dim=-1,
                    eps=1e-6,
                ).to(dtype=compute_dtype)
            statistics = torch.stack(
                [
                    normalized.float().pow(2).mean(dim=-1).sqrt().to(compute_dtype),
                    normalized.float().std(dim=-1).to(compute_dtype),
                    # The first candidate has an all-zero drift vector.  A
                    # bare sqrt(0) has an infinite derivative, which can
                    # become 0*inf when the output head is initialized with
                    # zero weights.  Keep the statistic numerically smooth
                    # at that valid first-exit state.
                    drift.float()
                    .pow(2)
                    .mean(dim=-1)
                    .clamp_min(1e-12)
                    .sqrt()
                    .to(compute_dtype),
                    cosine_change * has_previous,
                ],
                dim=-1,
            )
            features = features + linear(self.drift_proj, drift) + linear(
                self.state_statistics_proj, statistics
            )
        output_hidden = torch.nn.functional.silu(features)
        output_hidden = linear(self.output[1], output_hidden)
        output_hidden = torch.nn.functional.silu(output_hidden)
        logits = linear(self.output[3], output_hidden).squeeze(-1)
        risk_hidden = torch.nn.functional.silu(features)
        risk_hidden = linear(self.risk_output[1], risk_hidden)
        risk_hidden = torch.nn.functional.silu(risk_hidden)
        risk_parameters = linear(self.risk_output[3], risk_hidden)
        risk_mean = torch.sigmoid(risk_parameters[..., 0])
        risk_scale = torch.nn.functional.softplus(risk_parameters[..., 1]) + 1e-4
        if not per_sample:
            logits = logits.squeeze(0)
            risk_mean = risk_mean.squeeze(0)
            risk_scale = risk_scale.squeeze(0)
        if return_risk:
            return logits, risk_mean, risk_scale
        return logits

    @staticmethod
    def quality_prediction_loss(
        predicted_mean: torch.Tensor,
        predicted_scale: torch.Tensor,
        target_risk: torch.Tensor,
        *,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict harmful-regret risk and calibrate an absolute-error scale."""

        if not (
            predicted_mean.shape
            == predicted_scale.shape
            == target_risk.shape
        ):
            raise ValueError("quality prediction tensors must have the same shape")
        if sample_weights is not None:
            if sample_weights.shape != predicted_mean.shape[:1]:
                raise ValueError("sample_weights must have shape [B]")
            if torch.any(sample_weights < 0):
                raise ValueError("sample_weights cannot contain negatives")
        if torch.any((target_risk < 0) | (target_risk > 1)):
            raise ValueError("quality risk targets must be in [0, 1]")
        predicted_probability = predicted_mean.float().clamp(
            min=1e-6, max=1.0 - 1e-6
        )
        probability_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.logit(predicted_probability),
            target_risk.float(),
            reduction="none",
        )
        calibration_target = (
            (
                predicted_probability.detach()
                - target_risk.float()
            ).abs()
            + 0.02
        )
        scale_loss = torch.nn.functional.smooth_l1_loss(
            predicted_scale.float(),
            calibration_target,
            beta=0.05,
            reduction="none",
        )
        if sample_weights is not None:
            normalized_weights = (
                sample_weights.float()
                / sample_weights.float().mean().clamp_min(1e-6)
            ).to(device=probability_loss.device)
            probability_loss = probability_loss * normalized_weights.unsqueeze(-1)
            scale_loss = scale_loss * normalized_weights.unsqueeze(-1)
        return probability_loss.mean() + 0.25 * scale_loss.mean()

    @staticmethod
    def risk_targets(
        per_depth_cost: torch.Tensor,
        *,
        strategy: str,
        quality_margin: float,
        harmful_regret: torch.Tensor,
        answer_regret: torch.Tensor,
        sequence_regret_weight: float = 0.0,
        target_indexes: torch.Tensor | None = None,
        acceptable_mask: torch.Tensor | None = None,
        label_correct: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build risk-head targets aligned with the configured gate objective."""

        if per_depth_cost.ndim != 2:
            raise ValueError("per_depth_cost must have shape [B, K]")
        if harmful_regret.shape != per_depth_cost.shape:
            raise ValueError("harmful_regret must match per_depth_cost")
        if answer_regret.shape != per_depth_cost.shape:
            raise ValueError("answer_regret must match per_depth_cost")
        if quality_margin < 0:
            raise ValueError("quality_margin cannot be negative")
        if sequence_regret_weight < 0:
            raise ValueError("sequence_regret_weight cannot be negative")
        if strategy not in {
            "harmful_regret",
            "target_depth",
            "acceptable_binary",
            "label_error",
            "best_label_ce",
        }:
            raise ValueError(
                "strategy must be one of: harmful_regret, target_depth, "
                "acceptable_binary, label_error, best_label_ce"
            )

        if strategy == "harmful_regret":
            return (
                harmful_regret.float()
                + sequence_regret_weight * answer_regret.float()
            ).clamp(max=1.0)

        if strategy == "target_depth":
            if target_indexes is None:
                raise ValueError(
                    "target_indexes are required for target_depth risk targets"
                )
            if target_indexes.shape != per_depth_cost.shape[:1]:
                raise ValueError("target_indexes must have shape [B]")
            if torch.any(
                (target_indexes < 0)
                | (target_indexes >= per_depth_cost.shape[-1])
            ):
                raise ValueError("target_indexes contains an invalid candidate")
            indexes = torch.arange(
                per_depth_cost.shape[-1],
                device=per_depth_cost.device,
            ).unsqueeze(0)
            return indexes.lt(
                target_indexes.to(device=per_depth_cost.device).unsqueeze(1)
            ).float()

        if strategy == "label_error":
            if label_correct is None:
                raise ValueError(
                    "label_correct is required for label_error risk targets"
                )
            if label_correct.shape != per_depth_cost.shape:
                raise ValueError("label_correct must match per_depth_cost")
            unsafe = ~label_correct.to(
                device=per_depth_cost.device,
                dtype=torch.bool,
            )
            if acceptable_mask is not None:
                if acceptable_mask.shape != per_depth_cost.shape:
                    raise ValueError("acceptable_mask must match per_depth_cost")
                unsafe = unsafe | ~acceptable_mask.to(
                    device=per_depth_cost.device,
                    dtype=torch.bool,
                )
            unsafe[:, -1] = False
            return unsafe.float()

        if strategy == "best_label_ce":
            best_cost = per_depth_cost.min(dim=-1, keepdim=True).values
            unsafe = per_depth_cost > (best_cost + quality_margin)
            if acceptable_mask is not None:
                if acceptable_mask.shape != per_depth_cost.shape:
                    raise ValueError("acceptable_mask must match per_depth_cost")
                unsafe = unsafe | ~acceptable_mask.to(
                    device=per_depth_cost.device,
                    dtype=torch.bool,
                )
            unsafe[:, -1] = False
            return unsafe.float()

        unsafe = per_depth_cost > (per_depth_cost[:, -1:] + quality_margin)
        if acceptable_mask is not None:
            if acceptable_mask.shape != per_depth_cost.shape:
                raise ValueError("acceptable_mask must match per_depth_cost")
            unsafe = unsafe | ~acceptable_mask.to(
                device=per_depth_cost.device,
                dtype=torch.bool,
            )
        unsafe[:, -1] = False
        return unsafe.float()

    @staticmethod
    def harmful_teacher_regret(
        candidate_predictions: torch.Tensor,
        teacher_predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Mark tokens an exit gets wrong that the full-depth teacher gets right.

        Unlike symmetric teacher disagreement, this does not penalize a
        shallower exit for correcting a full-depth mistake.
        """

        if candidate_predictions.shape != labels.shape:
            raise ValueError("candidate_predictions and labels must align")
        if teacher_predictions.shape != labels.shape:
            raise ValueError("teacher_predictions and labels must align")
        return (
            candidate_predictions.ne(labels)
            & teacher_predictions.eq(labels)
        ).float()

    @staticmethod
    def sample_sequence_risk(
        token_risk: torch.Tensor,
        sample_ids: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Collapse token risk to whether any answer token is unsafe."""

        if token_risk.ndim != 2:
            raise ValueError("token_risk must have shape [K, T]")
        if sample_ids.ndim != 1 or sample_ids.shape[0] != token_risk.shape[1]:
            raise ValueError("sample_ids must have shape [T]")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        risks = []
        for sample_index in range(num_samples):
            sample_mask = sample_ids == sample_index
            if not torch.any(sample_mask):
                raise ValueError(f"sample {sample_index} has no answer tokens")
            risks.append(token_risk[:, sample_mask].amax(dim=1))
        return torch.stack(risks, dim=0)

    @staticmethod
    def sample_teacher_exact_match(
        candidate_predictions: torch.Tensor,
        teacher_predictions: torch.Tensor,
        sample_ids: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Return per-sample exact match to the full-depth teacher answer."""

        if candidate_predictions.shape != teacher_predictions.shape:
            raise ValueError(
                "candidate_predictions and teacher_predictions must align"
            )
        if candidate_predictions.ndim != 2:
            raise ValueError("predictions must have shape [K, T]")
        if (
            sample_ids.ndim != 1
            or sample_ids.shape[0] != candidate_predictions.shape[1]
        ):
            raise ValueError("sample_ids must have shape [T]")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        matches = []
        token_match = candidate_predictions.eq(teacher_predictions)
        for sample_index in range(num_samples):
            sample_mask = sample_ids == sample_index
            if not torch.any(sample_mask):
                raise ValueError(f"sample {sample_index} has no answer tokens")
            matches.append(token_match[:, sample_mask].all(dim=1).float())
        return torch.stack(matches, dim=0)

    @staticmethod
    def sample_label_exact_match(
        candidate_predictions: torch.Tensor,
        labels: torch.Tensor,
        sample_ids: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Return per-sample exact match to supervised answer labels.

        This is a gain-seeking target: it can select a shallow exit that is
        correct even when the full-depth teacher is wrong, while still falling
        back to full depth when no candidate exactly matches the label.
        """

        if candidate_predictions.shape != labels.shape:
            raise ValueError("candidate_predictions and labels must align")
        if candidate_predictions.ndim != 2:
            raise ValueError("predictions must have shape [K, T]")
        if sample_ids.ndim != 1 or sample_ids.shape[0] != candidate_predictions.shape[1]:
            raise ValueError("sample_ids must have shape [T]")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        matches = []
        token_match = candidate_predictions.eq(labels)
        for sample_index in range(num_samples):
            sample_mask = sample_ids == sample_index
            if not torch.any(sample_mask):
                raise ValueError(f"sample {sample_index} has no answer tokens")
            matches.append(token_match[:, sample_mask].all(dim=1).float())
        return torch.stack(matches, dim=0)

    @staticmethod
    def halting_distribution(
        logits: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """Convert sequential halt logits into an exit distribution.

        The last candidate receives all remaining probability, so the
        distribution always sums to one.
        """

        if logits.ndim < 1 or logits.shape[-1] < 1:
            raise ValueError("logits must have a non-empty candidate dimension")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if logits.shape[-1] == 1:
            return torch.ones_like(logits)

        hazards = torch.sigmoid(logits[..., :-1] / temperature)
        survival = torch.ones(
            logits.shape[:-1], device=logits.device, dtype=logits.dtype
        )
        probabilities = []
        for hazard in hazards.unbind(dim=-1):
            probabilities.append(survival * hazard)
            survival = survival * (1.0 - hazard)
        probabilities.append(survival)
        return torch.stack(probabilities, dim=-1)

    @staticmethod
    def supervised_halting_loss(
        logits: torch.Tensor,
        per_depth_cost: torch.Tensor,
        *,
        quality_margin: float,
        temperature: float = 1.0,
        acceptable_mask: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
        target_strategy: str = "earliest_safe",
        min_exit_gain: float = 0.0,
        positive_weight: float = 1.0,
        label_correct: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Train sequential hazards to choose a quality-preserving exit.

        An exit is acceptable when its per-sample token CE is within
        ``quality_margin`` nats of the final candidate.  The final exit is
        therefore always acceptable. ``earliest_safe`` chooses the first
        acceptable candidate. ``lowest_cost_safe`` chooses the lowest-cost
        acceptable candidate, with ``min_exit_gain`` requiring an early exit to
        beat the final candidate by that many nats. Hazards before the target
        learn "continue", the target hazard learns "halt", and later hazards
        are masked because inference would never reach them.
        """

        if logits.ndim != 2 or per_depth_cost.shape != logits.shape:
            raise ValueError("logits and per_depth_cost must have shape [B, K]")
        if logits.shape[-1] < 1:
            raise ValueError("candidate dimension cannot be empty")
        if quality_margin < 0:
            raise ValueError("quality_margin cannot be negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if target_strategy not in {
            "earliest_safe",
            "lowest_cost_safe",
            "label_correct",
            "best_label_ce",
        }:
            raise ValueError(
                "target_strategy must be one of: earliest_safe, "
                "lowest_cost_safe, label_correct, best_label_ce"
            )
        if min_exit_gain < 0:
            raise ValueError("min_exit_gain cannot be negative")
        if positive_weight <= 0:
            raise ValueError("positive_weight must be positive")
        if sample_weights is not None:
            if sample_weights.shape != logits.shape[:1]:
                raise ValueError("sample_weights must have shape [B]")
            if torch.any(sample_weights < 0):
                raise ValueError("sample_weights cannot contain negatives")

        acceptable = per_depth_cost <= (
            per_depth_cost[:, -1:] + quality_margin
        )
        if acceptable_mask is not None:
            if acceptable_mask.shape != acceptable.shape:
                raise ValueError(
                    "acceptable_mask must match logits and per_depth_cost"
                )
            acceptable = acceptable & acceptable_mask.to(
                device=acceptable.device, dtype=torch.bool
            )
        # The full-depth candidate is the teacher and always provides a safe
        # fallback, even if a caller supplied a malformed final mask entry.
        acceptable[:, -1] = True
        if target_strategy == "earliest_safe":
            target_indexes = acceptable.to(dtype=torch.int64).argmax(dim=-1)
        elif target_strategy == "lowest_cost_safe":
            target_eligible = acceptable.clone()
            if logits.shape[-1] > 1:
                improves_final = per_depth_cost <= (
                    per_depth_cost[:, -1:] - min_exit_gain
                )
                target_eligible[:, :-1] &= improves_final[:, :-1]
            target_eligible[:, -1] = True
            tie_breaker = torch.arange(
                logits.shape[-1],
                device=per_depth_cost.device,
                dtype=torch.float32,
            ).unsqueeze(0) * 1e-6
            masked_cost = (
                per_depth_cost.float() + tie_breaker
            ).masked_fill(~target_eligible, torch.inf)
            target_indexes = masked_cost.argmin(dim=-1)
        elif target_strategy == "label_correct":
            if label_correct is None:
                raise ValueError(
                    "label_correct is required for label_correct target strategy"
                )
            if label_correct.shape != logits.shape:
                raise ValueError("label_correct must match logits")
            target_eligible = (
                label_correct.to(device=acceptable.device, dtype=torch.bool)
                & acceptable
            )
            has_label_correct = target_eligible.any(dim=-1, keepdim=True)
            fallback = torch.zeros_like(target_eligible)
            fallback[:, -1] = True
            target_eligible = torch.where(
                has_label_correct, target_eligible, fallback
            )
            target_indexes = target_eligible.to(dtype=torch.int64).argmax(dim=-1)
        else:
            target_eligible = acceptable.clone()
            if logits.shape[-1] > 1 and min_exit_gain > 0:
                improves_final = per_depth_cost <= (
                    per_depth_cost[:, -1:] - min_exit_gain
                )
                target_eligible[:, :-1] &= improves_final[:, :-1]
            target_eligible[:, -1] = True
            tie_breaker = torch.arange(
                logits.shape[-1],
                device=per_depth_cost.device,
                dtype=torch.float32,
            ).unsqueeze(0) * 1e-6
            masked_cost = (
                per_depth_cost.float() + tie_breaker
            ).masked_fill(~target_eligible, torch.inf)
            target_indexes = masked_cost.argmin(dim=-1)
        if logits.shape[-1] == 1:
            return logits.sum() * 0.0, target_indexes

        candidate_indexes = torch.arange(
            logits.shape[-1] - 1, device=logits.device
        ).unsqueeze(0)
        supervised_mask = candidate_indexes <= target_indexes.unsqueeze(1)
        halt_targets = candidate_indexes == target_indexes.unsqueeze(1)
        per_hazard_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, :-1] / temperature,
            halt_targets.to(dtype=logits.dtype),
            reduction="none",
        )
        if positive_weight != 1.0:
            per_hazard_loss = per_hazard_loss * torch.where(
                halt_targets,
                per_hazard_loss.new_full((), float(positive_weight)),
                per_hazard_loss.new_ones(()),
            )
        per_sample_loss = (
            (per_hazard_loss * supervised_mask).sum(dim=-1)
            / supervised_mask.sum(dim=-1).clamp_min(1)
        )
        if sample_weights is not None:
            normalized_weights = (
                sample_weights.float()
                / sample_weights.float().mean().clamp_min(1e-6)
            )
            per_sample_loss = per_sample_loss * normalized_weights
        return per_sample_loss.mean(), target_indexes

    @staticmethod
    def exit_confidence_loss(
        logits: torch.Tensor,
        per_depth_cost: torch.Tensor,
        *,
        quality_margin: float,
        temperature: float = 1.0,
        acceptable_mask: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
        target_indexes: torch.Tensor | None = None,
        positive_weight: float = 1.0,
    ) -> torch.Tensor:
        """Calibrate early-exit hazards as stop probabilities.

        Without ``target_indexes`` each acceptable early exit is trained as
        safe to stop. With ``target_indexes`` the hazards are calibrated to the
        selected sequential stop target, so earlier/later hazards stay low.
        """

        if logits.ndim != 2 or per_depth_cost.shape != logits.shape:
            raise ValueError("logits and per_depth_cost must have shape [B, K]")
        if logits.shape[-1] < 1:
            raise ValueError("candidate dimension cannot be empty")
        if quality_margin < 0:
            raise ValueError("quality_margin cannot be negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if sample_weights is not None:
            if sample_weights.shape != logits.shape[:1]:
                raise ValueError("sample_weights must have shape [B]")
            if torch.any(sample_weights < 0):
                raise ValueError("sample_weights cannot contain negatives")
        if target_indexes is not None:
            if target_indexes.shape != logits.shape[:1]:
                raise ValueError("target_indexes must have shape [B]")
            if torch.any(
                (target_indexes < 0) | (target_indexes >= logits.shape[-1])
            ):
                raise ValueError("target_indexes contains an invalid candidate")
        if positive_weight <= 0:
            raise ValueError("positive_weight must be positive")
        if logits.shape[-1] == 1:
            return logits.sum() * 0.0

        if target_indexes is None:
            acceptable = per_depth_cost <= (
                per_depth_cost[:, -1:] + quality_margin
            )
            if acceptable_mask is not None:
                if acceptable_mask.shape != acceptable.shape:
                    raise ValueError(
                        "acceptable_mask must match logits and per_depth_cost"
                    )
                acceptable = acceptable & acceptable_mask.to(
                    device=acceptable.device, dtype=torch.bool
                )
            acceptable[:, -1] = True
            targets = acceptable[:, :-1].to(dtype=logits.dtype)
        else:
            early_indexes = torch.arange(
                logits.shape[-1] - 1,
                device=logits.device,
            ).unsqueeze(0)
            targets = early_indexes.eq(
                target_indexes.to(device=logits.device).unsqueeze(1)
            ).to(dtype=logits.dtype)
        per_hazard_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, :-1] / temperature,
            targets,
            reduction="none",
        )
        if positive_weight != 1.0:
            per_hazard_loss = per_hazard_loss * torch.where(
                targets.bool(),
                per_hazard_loss.new_full((), float(positive_weight)),
                per_hazard_loss.new_ones(()),
            )
        if sample_weights is not None:
            normalized_weights = (
                sample_weights.float()
                / sample_weights.float().mean().clamp_min(1e-6)
            )
            per_hazard_loss = per_hazard_loss * normalized_weights.unsqueeze(-1)
        return per_hazard_loss.mean()
