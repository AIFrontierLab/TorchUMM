"""Task- and timestep-conditioned depth policies."""

from __future__ import annotations

import fnmatch
import json
import math
import re
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch


@dataclass(frozen=True)
class DepthRequest:
    """Runtime information supplied by the BAGEL wrapper to the controller."""

    task: str | None = None
    sample_tasks: Sequence[str] | None = None
    sample_lens: Sequence[int] | None = None
    kind: str | None = None
    timestep: torch.Tensor | float | None = None
    depth_override: int | None = None
    has_understanding: bool | None = None
    has_generation: bool | None = None
    und_token_indexes: torch.Tensor | None = None
    gen_token_indexes: torch.Tensor | None = None
    ce_loss_indexes: torch.Tensor | None = None
    mse_loss_indexes: torch.Tensor | None = None


_CURRENT_DEPTH_REQUEST: ContextVar[DepthRequest | None] = ContextVar(
    "flexumm_depth_request", default=None
)


def current_depth_request() -> DepthRequest | None:
    return _CURRENT_DEPTH_REQUEST.get()


@dataclass(frozen=True)
class DepthPlan:
    """Resolved exits for one shared-backbone call.

    Depths count executed transformer blocks and are therefore one-based.
    """

    task: str
    kind: str
    understanding_depth: int | None
    generation_depth: int | None
    execution_depth: int
    timestep: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DynamicDepthConfig:
    """Configuration for dynamic shared-backbone depth.

    ``None`` means full depth.  Consequently, constructing this class with no
    arguments exactly preserves BAGEL's original 28-layer behavior.
    """

    understanding_depth: int | None = None
    generation_min_depth: int | None = None
    generation_max_depth: int | None = None
    generation_schedule: str = "fixed"
    depth_multiple: int = 1
    timestep_reduce: str = "min"
    understanding_task_depths: dict[str, int] = field(default_factory=dict)
    generation_task_depths: dict[str, int] = field(default_factory=dict)
    replay_deeper_kv: bool = True
    strict_cache_validation: bool = True
    router_enabled: bool = False
    router_candidates_understanding: list[int] = field(default_factory=list)
    router_candidates_generation: list[int] = field(default_factory=list)
    # Optional task-specific candidate lists.  This lets a quality-sensitive
    # task stay at the final checkpoint while another task uses the same
    # learned router for genuine early exits.
    router_task_candidates_understanding: dict[str, list[int]] = field(
        default_factory=dict
    )
    router_min_exit_depth: int | None = None
    router_hidden_size: int = 256
    router_condition_on_task: bool = True
    router_use_state_drift: bool = False
    router_use_last_token: bool = False
    # Which input-dependent signal the inference router may use. ``full`` is
    # the trained contract; the other learned modes are controlled feature
    # masks for conditioning ablations. ``prediction_entropy`` is an
    # evaluation-only heuristic and bypasses the learned router head.
    router_conditioning: str = "full"
    router_conditioning_generation: str | None = None
    router_prediction_entropy_threshold: float = 0.35
    # Generation can use a separately trained router while retaining the
    # understanding router's lighter feature contract and task vocabulary.
    # ``None`` preserves the historical shared settings for older configs.
    router_tasks_generation: list[str] | None = None
    router_use_state_drift_generation: bool | None = None
    router_use_last_token_generation: bool | None = None
    router_tasks: list[str] = field(
        default_factory=lambda: [
            "understanding",
            "mixed",
            "vqa",
        ]
    )
    router_threshold: float = 0.5
    router_task_thresholds: dict[str, float] = field(default_factory=dict)
    router_task_loss_weights: dict[str, float] = field(default_factory=dict)
    router_temperature: float = 1.0
    router_initial_bias: float = -2.0
    router_expected_loss_weight: float = 1.0
    router_compute_weight: float = 0.02
    router_entropy_weight: float = 0.01
    router_supervised_weight: float = 1.0
    router_supervised_positive_weight: float = 1.0
    router_exit_calibration_weight: float = 0.0
    router_target_strategy: str = "earliest_safe"
    router_min_exit_gain: float = 0.0
    router_risk_target_strategy: str = "harmful_regret"
    router_hard_sample_weight: float = 0.0
    router_hard_sample_temperature: float = 1.0
    router_quality_margin: float = 0.15
    router_quality_prediction_weight: float = 0.0
    router_teacher_agreement: float = 0.0
    router_answer_agreement: float = 0.0
    router_max_teacher_regret: float = 1.0
    router_max_answer_regret: float = 1.0
    router_sequence_regret_weight: float = 0.0
    router_risk_threshold: float = 0.15
    router_risk_confidence: float = 0.0
    router_detach_features: bool = True
    # The generation router is small enough to keep in FP32.  This avoids
    # BF16 backward instability through LayerNorm/cosine feature paths while
    # leaving the shared BAGEL backbone and understanding branch unchanged.
    generation_router_float32: bool = False
    # The understanding router may optionally run its feature path in FP32.
    # This is useful for newly initialized routers that use LayerNorm and
    # representation-drift features, while leaving the shared backbone and
    # all non-router modules unchanged.
    understanding_router_float32: bool = False
    collect_multi_exit_training: bool = True
    # Current FlexUMM method: shared attention/norm plus a constrained FFN
    # subset pool and a TAFE SHARE/DECOUPLE/EXIT controller.  The default is
    # disabled so a plain pretrained BAGEL checkpoint remains loadable until
    # the new FFN subsets and controller have been post-trained.
    tafe_enabled: bool = False
    # Optional routed attention residual banks.  These use the same low-rank
    # subset shape and controller settings as the FFN route, but are kept in
    # separate namespaces so an attention-only ablation does not activate the
    # FFN route.  Attention routing never controls layer-depth exits.
    tafe_attention_enabled: bool = False
    tafe_attention_adapter_rank: int = 16
    tafe_attention_adapter_scale: float = 1.0
    tafe_attention_parameter_budget: int | None = None
    # FlexUMM uses independent controllers and specialized FFN banks for the
    # two BAGEL objectives.  The transformer stack remains shared.
    tafe_task_names_understanding: list[str] = field(
        default_factory=lambda: ["understanding", "mixed", "vqa"]
    )
    tafe_task_names_generation: list[str] = field(
        default_factory=lambda: ["generation", "mixed"]
    )
    tafe_controller_hidden_size: int = 256
    tafe_num_specialized_subsets: int = 2
    # Number of specialized FFN banks selected by a hard route.  ``1`` is
    # the original top-1 controller; larger values implement top-k routing.
    tafe_routing_width: int = 1
    tafe_adapter_rank: int = 16
    tafe_adapter_scale: float = 1.0
    tafe_condition_on_task: bool = True
    # Ablation controls.  When false, the TAFE utility head receives no
    # sample hidden-state signal and therefore tests task-only routing.
    tafe_use_hidden_state: bool = True
    # "learned" is the normal FlexUMM policy.  Fixed policies are useful for
    # static shared/private controls while keeping the same checkpoint layout.
    tafe_action_policy: str = "learned"
    # Optional static sharing boundary for the tuned-static control. Layers
    # 1..k use SHARE and layers k+1..N use DECOUPLE.
    tafe_static_boundary_layer: int | None = None
    tafe_allow_exit: bool = True
    tafe_lambda_cost: float = 0.0
    tafe_action_costs: list[float] = field(
        default_factory=lambda: [1.0, 0.75, 0.0]
    )
    # Optional inference-time cost overrides for task-family-specific TAFE
    # policies. These affect only SHARE/DECOUPLE/EXIT selection; they do not
    # add checkpoint parameters or alter the shared backbone.
    tafe_task_action_costs: dict[str, list[float]] = field(default_factory=dict)
    tafe_initial_action_bias: list[float] = field(
        default_factory=lambda: [0.0, -0.1, -1.0]
    )
    # During post-training, use a differentiable path mixture so the TAFE
    # utilities and specialized banks receive task-loss gradients. Evaluation
    # normally uses discrete argmax actions, with an optional deterministic
    # probability mixture for quality-preserving inference.
    tafe_soft_routing_train: bool = False
    tafe_soft_routing_eval: bool = False
    tafe_soft_routing_temperature: float = 1.0
    # Maximizing training-time action entropy prevents the learned FFN gate
    # from collapsing to one hard action for every token.  Inference remains
    # discrete, so this regularizer only shapes the router parameters.
    tafe_entropy_weight: float = 0.0
    # Encourage a non-collapsed, token-dependent SHARE/DECOUPLE policy.  The
    # balance term controls the batch-level action prior; the diversity term
    # is a mutual-information-style regularizer that makes individual token
    # decisions more decisive while preserving that prior.  Both are
    # training-only and leave the shared backbone untouched.
    tafe_action_balance_weight: float = 0.0
    tafe_action_diversity_weight: float = 0.0
    tafe_action_target: list[float] = field(
        default_factory=lambda: [0.5, 0.5, 0.0]
    )
    tafe_parameter_budget: int | None = None
    tafe_hard_budget: float | None = None
    # Legacy fusion-sealing fields remain parseable only for old experiment
    # manifests; they are no longer part of the active FlexUMM path.
    fusion_sealing_enabled: bool = False
    fusion_sealing_router_enabled: bool = True
    fusion_sealing_candidates_understanding: list[int] = field(
        default_factory=list
    )
    fusion_sealing_candidates_generation: list[int] = field(
        default_factory=list
    )
    fusion_sealing_num_tokens: int = 8
    fusion_sealing_bottleneck_size: int = 64
    fusion_sealing_realizer_rank: int = 64
    fusion_sealing_realizer_scale: float = 1.0
    fusion_sealing_router_hidden_size: int = 256
    fusion_sealing_initial_seal_bias: float = -6.0
    fusion_sealing_threshold: float = 0.5
    # Optional task-pattern overrides for the early and final controller
    # decisions.  Empty mappings preserve the scalar thresholds above.
    fusion_sealing_thresholds: dict[str, float] = field(default_factory=dict)
    # Optional threshold for the final realization quality gate.  Keeping it
    # separate from the early-seal threshold lets inference calibrate the
    # residual realization without making unsafe shallow exits.
    fusion_sealing_final_threshold: float | None = None
    fusion_sealing_final_thresholds: dict[str, float] = field(
        default_factory=dict
    )
    # Apply the final candidate's lightweight realization even when the
    # controller continues through the full shared-backbone depth.  This
    # makes the compiler/realizer an accuracy path rather than only an early
    # exit path; disabled by default for legacy behavior.
    fusion_sealing_apply_final_realization: bool = False
    # Apply the final realization per sample only when the controller predicts
    # that it is no worse than the unmodified final hidden state.
    fusion_sealing_gate_final_realization: bool = False
    fusion_sealing_condition_on_task: bool = True
    fusion_sealing_condition_on_fine_task: bool = False
    # Optionally expose the proposed residual realization itself to the
    # release controller.  This gives the quality gate a direct signal about
    # the change it is deciding to apply, while keeping the extra path small.
    fusion_sealing_condition_on_realization_delta: bool = False
    # Optionally let the lightweight compiler use the same compact benchmark
    # family identity as the release controller.  Kept separate for backward
    # compatibility with checkpoints created before task-conditioned capsules.
    fusion_sealing_condition_compiler_on_task: bool = False
    fusion_sealing_task_names: list[str] = field(default_factory=list)
    fusion_sealing_detach_controller_features: bool = True
    fusion_sealing_quality_margin: float = 0.08
    fusion_sealing_compute_weight: float = 0.02
    # Distill the full-fusion realization into each candidate capsule so the
    # compiler/realizer receives gradients before useful seal labels exist.
    fusion_sealing_preservation_weight: float = 1.0
    # Supervise the final lightweight realization directly with the task
    # target.  This is needed when the backbone is frozen: preservation alone
    # can only copy the current checkpoint and cannot improve its answer.
    fusion_sealing_quality_weight: float = 0.0
    # Optional task-pattern weights for the direct realization objective.
    # These weight the shared loss; they do not create benchmark-specific
    # modules or checkpoints.
    fusion_sealing_task_loss_weights: dict[str, float] = field(
        default_factory=dict
    )
    # Optional baseline-relative hinge, combined with the absolute task loss.
    fusion_sealing_improvement_weight: float = 0.0
    fusion_sealing_improvement_margin: float = 0.0
    understanding_adapter_layers: list[int] = field(default_factory=list)
    understanding_adapter_rank: int = 0
    understanding_adapter_scale: float = 1.0
    understanding_attention_lora_layers: list[int] = field(
        default_factory=list
    )
    understanding_attention_lora_rank: int = 0
    understanding_attention_lora_scale: float = 1.0
    understanding_mlp_lora_layers: list[int] = field(
        default_factory=list
    )
    understanding_mlp_lora_rank: int = 0
    understanding_mlp_lora_scale: float = 1.0
    understanding_fusion_source_depth: int | None = None
    understanding_fusion_target_depth: int | None = None
    understanding_fusion_rank: int = 0
    understanding_fusion_scale: float = 1.0
    understanding_delta_task_patterns: list[str] = field(default_factory=list)
    understanding_extra_delta_task_patterns: list[str] = field(
        default_factory=list
    )
    understanding_extra_adapter_layers: list[int] = field(default_factory=list)
    understanding_extra_adapter_rank: int = 0
    understanding_extra_adapter_scale: float = 1.0
    understanding_extra_attention_lora_layers: list[int] = field(
        default_factory=list
    )
    understanding_extra_attention_lora_rank: int = 0
    understanding_extra_attention_lora_scale: float = 1.0
    understanding_extra_mlp_lora_layers: list[int] = field(default_factory=list)
    understanding_extra_mlp_lora_rank: int = 0
    understanding_extra_mlp_lora_scale: float = 1.0
    understanding_extra_fusion_source_depth: int | None = None
    understanding_extra_fusion_target_depth: int | None = None
    understanding_extra_fusion_rank: int = 0
    understanding_extra_fusion_scale: float = 1.0
    understanding_named_delta_banks: list[dict[str, Any]] = field(
        default_factory=list
    )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DynamicDepthConfig":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown dynamic-depth configuration keys: {unknown}")
        return cls(**dict(values))

    @classmethod
    def from_json(cls, path: str | Path) -> "DynamicDepthConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def validate(self, num_layers: int) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if self.depth_multiple < 1:
            raise ValueError("depth_multiple must be positive")
        if self.generation_schedule not in {"fixed", "late_detail", "linear"}:
            raise ValueError(
                "generation_schedule must be one of: fixed, late_detail, linear"
            )
        if self.timestep_reduce not in {"min", "mean", "max"}:
            raise ValueError("timestep_reduce must be one of: min, mean, max")
        if self.router_hidden_size < 1:
            raise ValueError("router_hidden_size must be positive")
        valid_router_conditioning = {
            "full",
            "hidden_state",
            "representation_change",
            "prediction_entropy",
        }
        if self.router_conditioning not in valid_router_conditioning:
            raise ValueError(
                "router_conditioning must be one of: full, hidden_state, "
                "representation_change, prediction_entropy"
            )
        if (
            self.router_conditioning_generation is not None
            and self.router_conditioning_generation not in valid_router_conditioning
        ):
            raise ValueError(
                "router_conditioning_generation must be one of: full, "
                "hidden_state, representation_change, prediction_entropy, "
                "or null"
            )
        if not 0.0 <= self.router_prediction_entropy_threshold <= 1.0:
            raise ValueError(
                "router_prediction_entropy_threshold must be in [0, 1]"
            )
        if self.tafe_controller_hidden_size < 1:
            raise ValueError("tafe_controller_hidden_size must be positive")
        if self.tafe_num_specialized_subsets < 1:
            raise ValueError("tafe_num_specialized_subsets must be positive")
        if not 1 <= self.tafe_routing_width <= self.tafe_num_specialized_subsets:
            raise ValueError(
                "tafe_routing_width must be within the specialized subset count"
            )
        if self.tafe_adapter_rank < 1:
            raise ValueError("tafe_adapter_rank must be positive")
        if self.tafe_adapter_scale < 0:
            raise ValueError("tafe_adapter_scale cannot be negative")
        if self.tafe_attention_adapter_rank < 1:
            raise ValueError("tafe_attention_adapter_rank must be positive")
        if self.tafe_attention_adapter_scale < 0:
            raise ValueError(
                "tafe_attention_adapter_scale cannot be negative"
            )
        if self.tafe_action_policy not in {
            "learned",
            "share",
            "decouple",
            "exit",
        }:
            raise ValueError(
                "tafe_action_policy must be one of: learned, share, decouple, exit"
            )
        if self.tafe_static_boundary_layer is not None and not (
            0 <= int(self.tafe_static_boundary_layer) <= num_layers
        ):
            raise ValueError(
                "tafe_static_boundary_layer must be within [0, num_layers]"
            )
        if self.tafe_lambda_cost < 0:
            raise ValueError("tafe_lambda_cost cannot be negative")
        if len(self.tafe_action_costs) != 3:
            raise ValueError("tafe_action_costs must contain SHARE, DECOUPLE, EXIT")
        if any(float(cost) < 0 for cost in self.tafe_action_costs):
            raise ValueError("tafe_action_costs must be non-negative")
        for task, costs in self.tafe_task_action_costs.items():
            if len(costs) != 3:
                raise ValueError(
                    f"tafe_task_action_costs[{task!r}] must contain SHARE, DECOUPLE, EXIT"
                )
            if any(float(cost) < 0 for cost in costs):
                raise ValueError(
                    f"tafe_task_action_costs[{task!r}] must be non-negative"
                )
        if len(self.tafe_initial_action_bias) != 3:
            raise ValueError(
                "tafe_initial_action_bias must contain SHARE, DECOUPLE, EXIT"
            )
        if self.tafe_soft_routing_temperature <= 0:
            raise ValueError("tafe_soft_routing_temperature must be positive")
        if self.tafe_parameter_budget is not None and self.tafe_parameter_budget < 1:
            raise ValueError("tafe_parameter_budget must be positive when set")
        if (
            self.tafe_attention_parameter_budget is not None
            and self.tafe_attention_parameter_budget < 1
        ):
            raise ValueError(
                "tafe_attention_parameter_budget must be positive when set"
            )
        if self.tafe_hard_budget is not None and self.tafe_hard_budget < 0:
            raise ValueError("tafe_hard_budget cannot be negative")
        if self.tafe_entropy_weight < 0:
            raise ValueError("tafe_entropy_weight cannot be negative")
        if self.tafe_action_balance_weight < 0:
            raise ValueError("tafe_action_balance_weight cannot be negative")
        if self.tafe_action_diversity_weight < 0:
            raise ValueError("tafe_action_diversity_weight cannot be negative")
        if len(self.tafe_action_target) != 3:
            raise ValueError("tafe_action_target must contain three values")
        if any(float(value) < 0 for value in self.tafe_action_target):
            raise ValueError("tafe_action_target must be non-negative")
        target_sum = sum(float(value) for value in self.tafe_action_target)
        if target_sum <= 0:
            raise ValueError("tafe_action_target must have positive mass")
        if not 0.0 < self.router_threshold <= 1.0:
            raise ValueError("router_threshold must be in (0, 1]")
        for task, threshold in self.router_task_thresholds.items():
            if not 0.0 < threshold <= 1.0:
                raise ValueError(
                    f"router_task_thresholds[{task!r}] must be in (0, 1]"
                )
        for task, weight in self.router_task_loss_weights.items():
            if weight <= 0:
                raise ValueError(
                    f"router_task_loss_weights[{task!r}] must be positive"
                )
        if self.router_temperature <= 0:
            raise ValueError("router_temperature must be positive")
        if self.router_compute_weight < 0:
            raise ValueError("router_compute_weight cannot be negative")
        if self.router_expected_loss_weight < 0:
            raise ValueError("router_expected_loss_weight cannot be negative")
        if self.router_entropy_weight < 0:
            raise ValueError("router_entropy_weight cannot be negative")
        if self.router_supervised_weight < 0:
            raise ValueError("router_supervised_weight cannot be negative")
        if self.router_supervised_positive_weight <= 0:
            raise ValueError(
                "router_supervised_positive_weight must be positive"
            )
        if self.router_exit_calibration_weight < 0:
            raise ValueError("router_exit_calibration_weight cannot be negative")
        if self.router_target_strategy not in {
            "earliest_safe",
            "lowest_cost_safe",
            "label_correct",
            "best_label_ce",
        }:
            raise ValueError(
                "router_target_strategy must be one of: "
                "earliest_safe, lowest_cost_safe, label_correct, best_label_ce"
            )
        if self.router_min_exit_gain < 0:
            raise ValueError("router_min_exit_gain cannot be negative")
        if self.router_risk_target_strategy not in {
            "harmful_regret",
            "target_depth",
            "acceptable_binary",
            "label_error",
            "best_label_ce",
        }:
            raise ValueError(
                "router_risk_target_strategy must be one of: "
                "harmful_regret, target_depth, acceptable_binary, "
                "label_error, best_label_ce"
            )
        if self.router_hard_sample_weight < 0:
            raise ValueError("router_hard_sample_weight cannot be negative")
        if self.router_hard_sample_temperature <= 0:
            raise ValueError("router_hard_sample_temperature must be positive")
        if self.router_quality_prediction_weight < 0:
            raise ValueError("router_quality_prediction_weight cannot be negative")
        if self.router_quality_margin < 0:
            raise ValueError("router_quality_margin cannot be negative")
        if not 0.0 <= self.router_teacher_agreement <= 1.0:
            raise ValueError("router_teacher_agreement must be in [0, 1]")
        if not 0.0 <= self.router_answer_agreement <= 1.0:
            raise ValueError("router_answer_agreement must be in [0, 1]")
        if not 0.0 <= self.router_max_teacher_regret <= 1.0:
            raise ValueError("router_max_teacher_regret must be in [0, 1]")
        if not 0.0 <= self.router_max_answer_regret <= 1.0:
            raise ValueError("router_max_answer_regret must be in [0, 1]")
        if self.router_sequence_regret_weight < 0:
            raise ValueError("router_sequence_regret_weight cannot be negative")
        if self.router_risk_threshold < 0:
            raise ValueError("router_risk_threshold cannot be negative")
        if self.router_risk_confidence < 0:
            raise ValueError("router_risk_confidence cannot be negative")
        if self.understanding_adapter_rank < 0:
            raise ValueError("understanding_adapter_rank cannot be negative")
        if self.understanding_adapter_scale < 0:
            raise ValueError("understanding_adapter_scale cannot be negative")
        if self.understanding_attention_lora_rank < 0:
            raise ValueError(
                "understanding_attention_lora_rank cannot be negative"
            )
        if self.understanding_attention_lora_scale < 0:
            raise ValueError(
                "understanding_attention_lora_scale cannot be negative"
            )
        if self.understanding_mlp_lora_rank < 0:
            raise ValueError("understanding_mlp_lora_rank cannot be negative")
        if self.understanding_mlp_lora_scale < 0:
            raise ValueError("understanding_mlp_lora_scale cannot be negative")
        if self.understanding_fusion_rank < 0:
            raise ValueError("understanding_fusion_rank cannot be negative")
        if self.understanding_fusion_scale < 0:
            raise ValueError("understanding_fusion_scale cannot be negative")
        if self.understanding_extra_adapter_rank < 0:
            raise ValueError(
                "understanding_extra_adapter_rank cannot be negative"
            )
        if self.understanding_extra_adapter_scale < 0:
            raise ValueError(
                "understanding_extra_adapter_scale cannot be negative"
            )
        if self.understanding_extra_attention_lora_rank < 0:
            raise ValueError(
                "understanding_extra_attention_lora_rank cannot be negative"
            )
        if self.understanding_extra_attention_lora_scale < 0:
            raise ValueError(
                "understanding_extra_attention_lora_scale cannot be negative"
            )
        if self.understanding_extra_mlp_lora_rank < 0:
            raise ValueError(
                "understanding_extra_mlp_lora_rank cannot be negative"
            )
        if self.understanding_extra_mlp_lora_scale < 0:
            raise ValueError(
                "understanding_extra_mlp_lora_scale cannot be negative"
            )
        if self.understanding_extra_fusion_rank < 0:
            raise ValueError(
                "understanding_extra_fusion_rank cannot be negative"
            )
        if self.understanding_extra_fusion_scale < 0:
            raise ValueError(
                "understanding_extra_fusion_scale cannot be negative"
            )
        for field_name, patterns in (
            (
                "understanding_delta_task_patterns",
                self.understanding_delta_task_patterns,
            ),
            (
                "understanding_extra_delta_task_patterns",
                self.understanding_extra_delta_task_patterns,
            ),
        ):
            for index, pattern in enumerate(patterns):
                if not isinstance(pattern, str) or not pattern.strip():
                    raise ValueError(
                        f"{field_name} must contain non-empty strings; "
                        f"item {index} is {pattern!r}"
                    )
        seen_named_banks: set[str] = set()
        bank_name_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        for bank_index, bank in enumerate(self.understanding_named_delta_banks):
            if not isinstance(bank, Mapping):
                raise ValueError(
                    "understanding_named_delta_banks must contain objects; "
                    f"item {bank_index} is {bank!r}"
                )
            name = bank.get("name")
            if not isinstance(name, str) or not bank_name_re.match(name):
                raise ValueError(
                    "understanding_named_delta_banks entries need a valid "
                    f"identifier name; item {bank_index} has {name!r}"
                )
            if name in seen_named_banks:
                raise ValueError(
                    f"Duplicate understanding named delta bank {name!r}"
                )
            seen_named_banks.add(name)
            patterns = bank.get("task_patterns", ())
            if not isinstance(patterns, Sequence) or isinstance(patterns, str):
                raise ValueError(
                    f"Named bank {name!r} task_patterns must be a list"
                )
            for pattern_index, pattern in enumerate(patterns):
                if not isinstance(pattern, str) or not pattern.strip():
                    raise ValueError(
                        f"Named bank {name!r} task_patterns must contain "
                        f"non-empty strings; item {pattern_index} is {pattern!r}"
                    )
            for rank_name in (
                "adapter_rank",
                "attention_lora_rank",
                "mlp_lora_rank",
                "fusion_rank",
            ):
                value = int(bank.get(rank_name, 0) or 0)
                if value < 0:
                    raise ValueError(
                        f"Named bank {name!r} {rank_name} cannot be negative"
                    )
            for scale_name in (
                "adapter_scale",
                "attention_lora_scale",
                "mlp_lora_scale",
                "fusion_scale",
            ):
                value = float(bank.get(scale_name, 1.0))
                if value < 0:
                    raise ValueError(
                        f"Named bank {name!r} {scale_name} cannot be negative"
                    )
            for layers_name, rank_name in (
                ("adapter_layers", "adapter_rank"),
                ("attention_lora_layers", "attention_lora_rank"),
                ("mlp_lora_layers", "mlp_lora_rank"),
            ):
                layers = list(bank.get(layers_name, ()) or ())
                if layers != sorted(set(layers)):
                    raise ValueError(
                        f"Named bank {name!r} {layers_name} must be sorted "
                        "and unique"
                    )
                if bool(layers) != bool(int(bank.get(rank_name, 0) or 0)):
                    raise ValueError(
                        f"Named bank {name!r} {layers_name} and {rank_name} "
                        "must either both enable the module or both be empty/zero"
                    )
            fusion_values = (
                bank.get("fusion_source_depth"),
                bank.get("fusion_target_depth"),
                int(bank.get("fusion_rank", 0) or 0) or None,
            )
            if any(value is not None for value in fusion_values) and not all(
                value is not None for value in fusion_values
            ):
                raise ValueError(
                    f"Named bank {name!r} fusion source, target, and rank "
                    "must either all enable fusion or all be unset/zero"
                )
        if self.understanding_adapter_layers != sorted(
            set(self.understanding_adapter_layers)
        ):
            raise ValueError(
                "understanding_adapter_layers must be sorted and unique"
            )
        if bool(self.understanding_adapter_layers) != bool(
            self.understanding_adapter_rank
        ):
            raise ValueError(
                "understanding_adapter_layers and understanding_adapter_rank "
                "must either both enable adapters or both be empty/zero"
            )
        if self.understanding_attention_lora_layers != sorted(
            set(self.understanding_attention_lora_layers)
        ):
            raise ValueError(
                "understanding_attention_lora_layers must be sorted and unique"
            )
        if bool(self.understanding_attention_lora_layers) != bool(
            self.understanding_attention_lora_rank
        ):
            raise ValueError(
                "understanding_attention_lora_layers and "
                "understanding_attention_lora_rank must either both enable "
                "attention LoRA or both be empty/zero"
            )
        if self.understanding_mlp_lora_layers != sorted(
            set(self.understanding_mlp_lora_layers)
        ):
            raise ValueError(
                "understanding_mlp_lora_layers must be sorted and unique"
            )
        if bool(self.understanding_mlp_lora_layers) != bool(
            self.understanding_mlp_lora_rank
        ):
            raise ValueError(
                "understanding_mlp_lora_layers and "
                "understanding_mlp_lora_rank must either both enable MLP "
                "LoRA or both be empty/zero"
            )
        if self.understanding_extra_adapter_layers != sorted(
            set(self.understanding_extra_adapter_layers)
        ):
            raise ValueError(
                "understanding_extra_adapter_layers must be sorted and unique"
            )
        if bool(self.understanding_extra_adapter_layers) != bool(
            self.understanding_extra_adapter_rank
        ):
            raise ValueError(
                "understanding_extra_adapter_layers and "
                "understanding_extra_adapter_rank must either both enable "
                "adapters or both be empty/zero"
            )
        if self.understanding_extra_attention_lora_layers != sorted(
            set(self.understanding_extra_attention_lora_layers)
        ):
            raise ValueError(
                "understanding_extra_attention_lora_layers must be sorted "
                "and unique"
            )
        if bool(self.understanding_extra_attention_lora_layers) != bool(
            self.understanding_extra_attention_lora_rank
        ):
            raise ValueError(
                "understanding_extra_attention_lora_layers and "
                "understanding_extra_attention_lora_rank must either both "
                "enable attention LoRA or both be empty/zero"
            )
        if self.understanding_extra_mlp_lora_layers != sorted(
            set(self.understanding_extra_mlp_lora_layers)
        ):
            raise ValueError(
                "understanding_extra_mlp_lora_layers must be sorted and unique"
            )
        if bool(self.understanding_extra_mlp_lora_layers) != bool(
            self.understanding_extra_mlp_lora_rank
        ):
            raise ValueError(
                "understanding_extra_mlp_lora_layers and "
                "understanding_extra_mlp_lora_rank must either both enable "
                "MLP LoRA or both be empty/zero"
            )
        fusion_values = (
            self.understanding_fusion_source_depth,
            self.understanding_fusion_target_depth,
            self.understanding_fusion_rank or None,
        )
        if any(value is not None for value in fusion_values) and not all(
            value is not None for value in fusion_values
        ):
            raise ValueError(
                "understanding fusion source, target, and rank must either "
                "all enable fusion or all be unset/zero"
            )
        if (
            self.understanding_fusion_source_depth is not None
            and self.understanding_fusion_target_depth is not None
            and self.understanding_fusion_source_depth
            >= self.understanding_fusion_target_depth
        ):
            raise ValueError(
                "understanding_fusion_source_depth must be less than "
                "understanding_fusion_target_depth"
            )
        extra_fusion_values = (
            self.understanding_extra_fusion_source_depth,
            self.understanding_extra_fusion_target_depth,
            self.understanding_extra_fusion_rank or None,
        )
        if any(value is not None for value in extra_fusion_values) and not all(
            value is not None for value in extra_fusion_values
        ):
            raise ValueError(
                "extra understanding fusion source, target, and rank must "
                "either all enable fusion or all be unset/zero"
            )
        if (
            self.understanding_extra_fusion_source_depth is not None
            and self.understanding_extra_fusion_target_depth is not None
            and self.understanding_extra_fusion_source_depth
            >= self.understanding_extra_fusion_target_depth
        ):
            raise ValueError(
                "understanding_extra_fusion_source_depth must be less than "
                "understanding_extra_fusion_target_depth"
            )
        if self.router_enabled and not (
            self.router_candidates_understanding
            or self.router_candidates_generation
        ):
            raise ValueError(
                "at least one router candidate list cannot be empty"
            )
        if self.router_candidates_understanding != sorted(
            set(self.router_candidates_understanding)
        ):
            raise ValueError(
                "router_candidates_understanding must be sorted and unique"
            )
        if self.router_candidates_generation != sorted(
            set(self.router_candidates_generation)
        ):
            raise ValueError(
                "router_candidates_generation must be sorted and unique"
            )
        for task, candidates in self.router_task_candidates_understanding.items():
            if not isinstance(task, str) or not task.strip():
                raise ValueError(
                    "router_task_candidates_understanding keys must be non-empty"
                )
            if candidates != sorted(set(candidates)) or not candidates:
                raise ValueError(
                    "router_task_candidates_understanding entries must be "
                    "non-empty, sorted, and unique"
                )
            for index, depth in enumerate(candidates):
                if not 1 <= depth <= num_layers:
                    raise ValueError(
                        "router_task_candidates_understanding["
                        f"{task!r}][{index}]={depth} is outside the valid "
                        f"range [1, {num_layers}]"
                    )
        if (
            self.router_min_exit_depth is not None
            and not 1 <= self.router_min_exit_depth <= num_layers
        ):
            raise ValueError(
                "router_min_exit_depth must be within the backbone"
            )
        if (
            self.router_enabled
            and self.router_min_exit_depth is not None
            and self.router_candidates_understanding
            and self.router_min_exit_depth
            > self.router_candidates_understanding[-1]
        ):
            raise ValueError(
                "router_min_exit_depth cannot exceed the final understanding "
                "candidate"
            )
        if (
            self.router_enabled
            and self.router_min_exit_depth is not None
            and self.router_candidates_generation
            and self.router_min_exit_depth
            > self.router_candidates_generation[-1]
        ):
            raise ValueError(
                "router_min_exit_depth cannot exceed the final generation "
                "candidate"
            )

        if self.fusion_sealing_num_tokens < 1:
            raise ValueError("fusion_sealing_num_tokens must be positive")
        if self.fusion_sealing_bottleneck_size < 1:
            raise ValueError(
                "fusion_sealing_bottleneck_size must be positive"
            )
        if self.fusion_sealing_realizer_rank < 1:
            raise ValueError("fusion_sealing_realizer_rank must be positive")
        if self.fusion_sealing_realizer_scale < 0:
            raise ValueError(
                "fusion_sealing_realizer_scale cannot be negative"
            )
        if self.fusion_sealing_router_hidden_size < 1:
            raise ValueError(
                "fusion_sealing_router_hidden_size must be positive"
            )
        if not 0.0 < self.fusion_sealing_threshold <= 1.0:
            raise ValueError("fusion_sealing_threshold must be in (0, 1]")
        if self.fusion_sealing_final_threshold is not None and not (
            0.0 < self.fusion_sealing_final_threshold <= 1.0
        ):
            raise ValueError(
                "fusion_sealing_final_threshold must be in (0, 1] when set"
            )
        for name, threshold in self.fusion_sealing_thresholds.items():
            if not 0.0 < threshold <= 1.0:
                raise ValueError(
                    f"fusion_sealing_thresholds[{name!r}] must be in (0, 1]"
                )
        for name, threshold in self.fusion_sealing_final_thresholds.items():
            if not 0.0 < threshold <= 1.0:
                raise ValueError(
                    "fusion_sealing_final_thresholds["
                    f"{name!r}] must be in (0, 1]"
                )
        if self.fusion_sealing_quality_margin < 0:
            raise ValueError(
                "fusion_sealing_quality_margin cannot be negative"
            )
        if self.fusion_sealing_compute_weight < 0:
            raise ValueError(
                "fusion_sealing_compute_weight cannot be negative"
            )
        if self.fusion_sealing_preservation_weight < 0:
            raise ValueError(
                "fusion_sealing_preservation_weight cannot be negative"
            )
        if self.fusion_sealing_quality_weight < 0:
            raise ValueError(
                "fusion_sealing_quality_weight cannot be negative"
            )
        for name, weight in self.fusion_sealing_task_loss_weights.items():
            if weight <= 0:
                raise ValueError(
                    f"fusion_sealing_task_loss_weights[{name!r}] must be positive"
                )
        if self.fusion_sealing_improvement_weight < 0:
            raise ValueError(
                "fusion_sealing_improvement_weight cannot be negative"
            )
        if self.fusion_sealing_improvement_margin < 0:
            raise ValueError(
                "fusion_sealing_improvement_margin cannot be negative"
            )
        for field_name, candidates in (
            (
                "fusion_sealing_candidates_understanding",
                self.fusion_sealing_candidates_understanding,
            ),
            (
                "fusion_sealing_candidates_generation",
                self.fusion_sealing_candidates_generation,
            ),
        ):
            if candidates != sorted(set(candidates)):
                raise ValueError(f"{field_name} must be sorted and unique")
            for index, depth in enumerate(candidates):
                if not 1 <= depth <= num_layers:
                    raise ValueError(
                        f"{field_name}[{index}]={depth} is outside the "
                        f"valid range [1, {num_layers}]"
                    )
        if self.fusion_sealing_enabled and not (
            self.fusion_sealing_candidates_understanding
            or self.fusion_sealing_candidates_generation
        ):
            raise ValueError(
                "fusion sealing requires at least one candidate depth"
            )

        named_depths: dict[str, int | None] = {
            "understanding_depth": self.understanding_depth,
            "generation_min_depth": self.generation_min_depth,
            "generation_max_depth": self.generation_max_depth,
            **{
                f"understanding_task_depths[{name!r}]": value
                for name, value in self.understanding_task_depths.items()
            },
            **{
                f"generation_task_depths[{name!r}]": value
                for name, value in self.generation_task_depths.items()
            },
            **(
                {
                    f"router_candidates_understanding[{index}]": value
                    for index, value in enumerate(
                        self.router_candidates_understanding
                    )
                }
                if self.router_enabled
                else {}
            ),
            **(
                {
                    f"router_candidates_generation[{index}]": value
                    for index, value in enumerate(
                        self.router_candidates_generation
                    )
                }
                if self.router_enabled
                else {}
            ),
            **{
                f"understanding_adapter_layers[{index}]": value
                for index, value in enumerate(
                    self.understanding_adapter_layers
                )
            },
            **{
                f"understanding_attention_lora_layers[{index}]": value
                for index, value in enumerate(
                    self.understanding_attention_lora_layers
                )
            },
            **{
                f"understanding_mlp_lora_layers[{index}]": value
                for index, value in enumerate(
                    self.understanding_mlp_lora_layers
                )
            },
            **{
                f"understanding_extra_adapter_layers[{index}]": value
                for index, value in enumerate(
                    self.understanding_extra_adapter_layers
                )
            },
            **{
                f"understanding_extra_attention_lora_layers[{index}]": value
                for index, value in enumerate(
                    self.understanding_extra_attention_lora_layers
                )
            },
            **{
                f"understanding_extra_mlp_lora_layers[{index}]": value
                for index, value in enumerate(
                    self.understanding_extra_mlp_lora_layers
                )
            },
            "understanding_fusion_source_depth": (
                self.understanding_fusion_source_depth
            ),
            "understanding_fusion_target_depth": (
                self.understanding_fusion_target_depth
            ),
            "understanding_extra_fusion_source_depth": (
                self.understanding_extra_fusion_source_depth
            ),
            "understanding_extra_fusion_target_depth": (
                self.understanding_extra_fusion_target_depth
            ),
        }
        for name, depth in named_depths.items():
            if depth is not None and not 1 <= depth <= num_layers:
                raise ValueError(
                    f"{name}={depth} is outside the valid range [1, {num_layers}]"
                )

        minimum = self.generation_min_depth or num_layers
        maximum = self.generation_max_depth or num_layers
        if minimum > maximum:
            raise ValueError(
                "generation_min_depth cannot exceed generation_max_depth"
            )


class DynamicDepthController:
    """Resolve task/timestep metadata into transformer exit depths."""

    def __init__(self, config: DynamicDepthConfig, num_layers: int):
        config.validate(num_layers)
        self.config = config
        self.num_layers = num_layers
        self.last_plan: DepthPlan | None = None
        self.depth_counts: Counter[tuple[str, str, int]] = Counter()

    @contextmanager
    def route(
        self,
        *,
        task: str | None = None,
        sample_tasks: Sequence[str] | None = None,
        sample_lens: Sequence[int] | None = None,
        kind: str | None = None,
        timestep: torch.Tensor | float | None = None,
        depth_override: int | None = None,
        has_understanding: bool | None = None,
        has_generation: bool | None = None,
        und_token_indexes: torch.Tensor | None = None,
        gen_token_indexes: torch.Tensor | None = None,
        ce_loss_indexes: torch.Tensor | None = None,
        mse_loss_indexes: torch.Tensor | None = None,
    ) -> Iterator[None]:
        request = DepthRequest(
            task=task,
            sample_tasks=tuple(sample_tasks) if sample_tasks is not None else None,
            sample_lens=(
                tuple(int(length) for length in sample_lens)
                if sample_lens is not None
                else None
            ),
            kind=kind,
            timestep=timestep,
            depth_override=depth_override,
            has_understanding=has_understanding,
            has_generation=has_generation,
            und_token_indexes=und_token_indexes,
            gen_token_indexes=gen_token_indexes,
            ce_loss_indexes=ce_loss_indexes,
            mse_loss_indexes=mse_loss_indexes,
        )
        token = _CURRENT_DEPTH_REQUEST.set(request)
        try:
            yield
        finally:
            _CURRENT_DEPTH_REQUEST.reset(token)

    def resolve(
        self,
        *,
        kind: str,
        task: str | None = None,
        timestep: torch.Tensor | float | None = None,
        depth_override: int | None = None,
        has_understanding: bool | None = None,
        has_generation: bool | None = None,
    ) -> DepthPlan:
        if kind not in {"understanding", "generation", "mixed"}:
            raise ValueError(f"Unsupported route kind: {kind!r}")

        if has_understanding is None:
            has_understanding = kind in {"understanding", "mixed"}
        if has_generation is None:
            has_generation = kind in {"generation", "mixed"}

        task_name = (task or kind).strip().lower()
        timestep_value = self._reduce_timestep(timestep)

        if depth_override is not None:
            self._validate_depth("depth_override", depth_override)
            understanding_depth = depth_override if has_understanding else None
            generation_depth = depth_override if has_generation else None
        else:
            understanding_depth = (
                self._understanding_depth(task_name) if has_understanding else None
            )
            generation_depth = (
                self._generation_depth(task_name, timestep_value)
                if has_generation
                else None
            )

        active_depths = [
            depth
            for depth in (understanding_depth, generation_depth)
            if depth is not None
        ]
        if not active_depths:
            # A conditioning-only call still needs a well-defined route.
            fallback_kind = "generation" if kind == "generation" else "understanding"
            fallback_depth = (
                self._generation_depth(task_name, timestep_value)
                if fallback_kind == "generation"
                else self._understanding_depth(task_name)
            )
            active_depths = [fallback_depth]

        plan = DepthPlan(
            task=task_name,
            kind=kind,
            understanding_depth=understanding_depth,
            generation_depth=generation_depth,
            execution_depth=max(active_depths),
            timestep=timestep_value,
        )
        self.last_plan = plan
        self.depth_counts[(plan.task, plan.kind, plan.execution_depth)] += 1
        return plan

    def stats(self) -> list[dict[str, Any]]:
        return [
            {"task": task, "kind": kind, "depth": depth, "calls": calls}
            for (task, kind, depth), calls in sorted(self.depth_counts.items())
        ]

    def router_threshold(self, task: str | None) -> float:
        """Return a task-specific halt threshold, with glob support."""

        task_name = (task or "understanding").strip().lower().replace("-", "_")
        mapping = self.config.router_task_thresholds
        if task_name in mapping:
            return float(mapping[task_name])
        for pattern, threshold in mapping.items():
            if fnmatch.fnmatch(task_name, pattern.lower().replace("-", "_")):
                return float(threshold)
        return float(self.config.router_threshold)

    def router_candidates(
        self, *, kind: str, task: str | None
    ) -> list[int]:
        """Return the candidate depths for this task, with glob support."""

        if kind != "understanding":
            return list(self.config.router_candidates_generation)
        task_name = (task or "understanding").strip().lower().replace("-", "_")
        mapping = self.config.router_task_candidates_understanding
        if task_name in mapping:
            return list(mapping[task_name])
        for pattern, candidates in mapping.items():
            normalized_pattern = pattern.lower().replace("-", "_")
            if fnmatch.fnmatch(task_name, normalized_pattern):
                return list(candidates)
        return list(self.config.router_candidates_understanding)

    def _understanding_depth(self, task: str) -> int:
        task_depth = self._match_task_depth(task, self.config.understanding_task_depths)
        depth = task_depth or self.config.understanding_depth or self.num_layers
        self._validate_depth("understanding depth", depth)
        return depth

    def _generation_depth(self, task: str, timestep: float | None) -> int:
        task_depth = self._match_task_depth(task, self.config.generation_task_depths)
        if task_depth is not None:
            self._validate_depth("generation task depth", task_depth)
            return task_depth

        minimum = self.config.generation_min_depth or self.num_layers
        maximum = self.config.generation_max_depth or self.num_layers
        if self.config.generation_schedule == "fixed" or timestep is None:
            depth = maximum
        else:
            # BAGEL integrates from t=1 (noise) to t=0 (clean/detail).
            progress = 1.0 - min(max(timestep, 0.0), 1.0)
            depth = math.ceil(minimum + progress * (maximum - minimum))
            multiple = self.config.depth_multiple
            depth = math.ceil(depth / multiple) * multiple
            depth = min(maximum, max(minimum, depth))

        self._validate_depth("generation depth", depth)
        return depth

    def _reduce_timestep(
        self, timestep: torch.Tensor | float | None
    ) -> float | None:
        if timestep is None:
            return None
        if isinstance(timestep, torch.Tensor):
            values = timestep.detach().float()
            if values.numel() == 0:
                return None
            if self.config.timestep_reduce == "min":
                value = values.min()
            elif self.config.timestep_reduce == "max":
                value = values.max()
            else:
                value = values.mean()
            return float(value.item())
        return float(timestep)

    @staticmethod
    def _match_task_depth(task: str, mapping: Mapping[str, int]) -> int | None:
        if task in mapping:
            return mapping[task]
        for pattern, depth in mapping.items():
            if fnmatch.fnmatch(task, pattern.lower()):
                return depth
        return None

    def _validate_depth(self, name: str, depth: int) -> None:
        if not 1 <= depth <= self.num_layers:
            raise ValueError(
                f"{name}={depth} is outside the valid range [1, {self.num_layers}]"
            )
