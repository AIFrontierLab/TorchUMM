"""TAFE tri-state routing and capacity-constrained FFN subsets.

This module implements the current FlexUMM method for the shared BAGEL/Qwen
backbone. Attention and normalization remain owned by the original decoder
layer. The routed FFN chooses SHARE, DECOUPLE, or EXIT after those shared
operations. Optional routed attention adds a low-rank residual after the
original attention output without wrapping its checkpointed projections.
Specialized banks are low-rank residual subsets by default, which keeps the
initial checkpoint compatible and avoids copying a full block per task.
"""

from __future__ import annotations

import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Sequence

import torch
from torch import nn


class TAFEAction(IntEnum):
    """Actions in the FlexUMM tri-state action space."""

    SHARE = 0
    DECOUPLE = 1
    EXIT = 2


ACTION_NAMES = tuple(action.name for action in TAFEAction)


@dataclass(frozen=True)
class TAFEOutput:
    """Action utilities and selected paths for one routed FFN call."""

    utilities: torch.Tensor
    actions: torch.Tensor
    subset_ids: torch.Tensor
    adjusted_utilities: torch.Tensor
    # Present only for the training-time differentiable mixture.  Discrete
    # actions remain the authoritative inference and diagnostics route.
    action_probabilities: torch.Tensor | None = None
    subset_probabilities: torch.Tensor | None = None


@dataclass
class TAFEExecutionState:
    """Per-forward diagnostics shared by routed FFNs."""

    task: str | None = None
    kind: str | None = None
    remaining_budget: float = 1.0
    action_trace: list[dict[str, Any]] | None = None
    should_stop: bool = False

    def __post_init__(self) -> None:
        if self.action_trace is None:
            self.action_trace = []

    def record(
        self,
        *,
        layer: int,
        kind: str,
        result: TAFEOutput,
    ) -> None:
        actions = result.actions.detach().flatten()
        counts = torch.bincount(
            actions.to(dtype=torch.long),
            minlength=len(TAFEAction),
        )
        self.should_stop = bool(
            actions.numel() > 0
            and int(counts[TAFEAction.EXIT].item()) == actions.numel()
        )
        self.action_trace.append(
            {
                "layer": int(layer),
                "kind": kind,
                "share": int(counts[TAFEAction.SHARE].item()),
                "decouple": int(counts[TAFEAction.DECOUPLE].item()),
                "exit": int(counts[TAFEAction.EXIT].item()),
                "mean_utility": result.utilities.detach()
                .float()
                .mean(dim=0)
                .cpu()
                .tolist(),
            }
        )


_CURRENT_TAFE_STATE: ContextVar[TAFEExecutionState | None] = ContextVar(
    "flexumm_tafe_state",
    default=None,
)


def current_tafe_state() -> TAFEExecutionState | None:
    """Return the active TAFE execution state, if one is installed."""

    return _CURRENT_TAFE_STATE.get()


@contextmanager
def tafe_execution(
    *,
    task: str | None = None,
    kind: str | None = None,
    remaining_budget: float = 1.0,
):
    """Install a TAFE state for a model forward."""

    state = TAFEExecutionState(
        task=task,
        kind=kind,
        remaining_budget=float(remaining_budget),
    )
    token = _CURRENT_TAFE_STATE.set(state)
    try:
        yield state
    finally:
        _CURRENT_TAFE_STATE.reset(token)


class FFNResidualSubset(nn.Module):
    """A small task/modal-specific residual FFN subset.

    The up projection starts at zero, so a newly added subset is behavior
    preserving. The down projection is initialized normally so it can train
    immediately once the subset is selected.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        *,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or rank < 1:
            raise ValueError("hidden_size and rank must be positive")
        if scale < 0:
            raise ValueError("scale must be non-negative")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.scale = float(scale)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = torch.nn.functional.rms_norm(
            hidden_states,
            (self.hidden_size,),
        )
        return self.up(torch.nn.functional.silu(self.down(normalized))) * self.scale


class AttentionResidualSubset(FFNResidualSubset):
    """A parameter-matched low-rank residual subset for attention output.

    Attention routing deliberately uses the same architecture and
    initialization as an FFN residual subset.  With equal rank and subset
    count, one routed attention bank has exactly the same parameter count as
    one routed FFN bank while leaving the pretrained attention projections
    untouched.
    """


def _packed_sample_ids_for_tokens(
    sample_tasks: Sequence[str] | None,
    sample_lens: Sequence[int] | None,
    token_indexes: torch.Tensor | None,
    token_count: int,
    *,
    device: torch.device,
) -> tuple[tuple[str, ...], torch.Tensor] | None:
    """Return validated sample labels and packed row-to-sample IDs."""

    if not sample_tasks or not sample_lens or token_count < 1:
        return None
    if len(sample_tasks) != len(sample_lens):
        return None

    if token_indexes is None:
        if sum(int(length) for length in sample_lens) != token_count:
            return None
        positions = torch.arange(token_count, device=device, dtype=torch.long)
    else:
        positions = token_indexes.to(device=device, dtype=torch.long).reshape(-1)
        if positions.numel() != token_count:
            return None

    lengths = torch.tensor(
        [int(length) for length in sample_lens],
        device=device,
        dtype=torch.long,
    )
    boundaries = torch.cumsum(lengths, dim=0)[:-1]
    # A position equal to a cumulative boundary belongs to the next sample.
    sample_ids = torch.bucketize(positions, boundaries, right=True)
    return tuple(str(task) for task in sample_tasks), sample_ids


def _sample_tasks_for_tokens(
    sample_tasks: Sequence[str] | None,
    sample_lens: Sequence[int] | None,
    token_indexes: torch.Tensor | None,
    token_count: int,
    *,
    device: torch.device,
) -> tuple[str, ...] | None:
    """Expand packed-sample task labels to the rows routed through one FFN.

    BAGEL packs several examples into one transformer sequence.  The depth
    controller already keeps the sample-level task labels and packed sample
    lengths, but a routed FFN sees only the rows selected for its modality.
    Convert those packed positions back to sample labels so TAFE task
    conditioning remains correct for mixed benchmark batches.
    """

    packed = _packed_sample_ids_for_tokens(
        sample_tasks,
        sample_lens,
        token_indexes,
        token_count,
        device=device,
    )
    if packed is None:
        return None
    sample_tasks, sample_ids = packed
    return tuple(
        sample_tasks[int(sample_id)]
        for sample_id in sample_ids.detach().cpu().tolist()
    )


def _sample_task_ids_for_tokens(
    sample_tasks: Sequence[str] | None,
    sample_lens: Sequence[int] | None,
    token_indexes: torch.Tensor | None,
    token_count: int,
    *,
    device: torch.device,
    task_id_lookup,
) -> torch.Tensor | None:
    """Expand packed samples to on-device TAFE task IDs without CPU sync."""

    packed = _packed_sample_ids_for_tokens(
        sample_tasks,
        sample_lens,
        token_indexes,
        token_count,
        device=device,
    )
    if packed is None:
        return None
    sample_tasks, sample_ids = packed
    sample_task_ids = torch.tensor(
        [int(task_id_lookup(task)) for task in sample_tasks],
        device=device,
        dtype=torch.long,
    )
    return sample_task_ids[sample_ids]


class FFNSubsetPool(nn.Module):
    """A default FFN plus a constrained pool of specialized FFN subsets.

    The default FFN is supplied by the base UMM. DECOUPLE adds one selected
    residual subset to that output; EXIT returns no FFN residual so the outer
    decoder residual preserves the current shared state.
    """

    def __init__(
        self,
        default_ffn: nn.Module,
        hidden_size: int,
        *,
        num_specialized_subsets: int = 2,
        adapter_rank: int = 16,
        adapter_scale: float = 1.0,
        parameter_budget: int | None = None,
    ) -> None:
        super().__init__()
        if num_specialized_subsets < 1:
            raise ValueError("num_specialized_subsets must be positive")
        self.default_ffn = default_ffn
        self.hidden_size = int(hidden_size)
        self.num_specialized_subsets = int(num_specialized_subsets)
        self.adapter_rank = int(adapter_rank)
        self.parameter_budget = parameter_budget
        self.specialized_banks = nn.ModuleList(
            [
                FFNResidualSubset(
                    hidden_size,
                    adapter_rank,
                    scale=adapter_scale,
                )
                for _ in range(num_specialized_subsets)
            ]
        )
        if parameter_budget is not None:
            self.validate_parameter_budget()

    def specialized_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for bank in self.specialized_banks
            for parameter in bank.parameters()
        )

    def validate_parameter_budget(self) -> None:
        if self.parameter_budget is None:
            return
        count = self.specialized_parameter_count()
        if count > self.parameter_budget:
            raise ValueError(
                "Specialized FFN pool exceeds parameter budget: "
                f"{count} > {self.parameter_budget}"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        actions: torch.Tensor,
        subset_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_actions = actions.to(device=flat_hidden.device).reshape(-1)
        if flat_actions.numel() != flat_hidden.shape[0]:
            raise ValueError("actions must have one entry per hidden-state row")
        if subset_ids is None:
            subset_ids = torch.zeros_like(flat_actions)
        flat_subsets = subset_ids.to(device=flat_hidden.device).reshape(-1)
        if flat_subsets.numel() != flat_hidden.shape[0]:
            raise ValueError(
                "subset_ids must have one entry per hidden-state row"
            )

        default_output = self.default_ffn(hidden_states)
        flat_output = default_output.reshape_as(flat_hidden).clone()
        decouple_positions = (
            flat_actions == int(TAFEAction.DECOUPLE)
        ).nonzero(as_tuple=False).flatten()
        for bank_id in flat_subsets[decouple_positions].unique(
            sorted=True
        ).tolist():
            if bank_id < 0 or bank_id >= self.num_specialized_subsets:
                raise ValueError(f"Invalid specialized FFN subset id {bank_id}")
            positions = decouple_positions[
                flat_subsets[decouple_positions] == bank_id
            ]
            residual = self.specialized_banks[bank_id](
                flat_hidden[positions]
            ).to(dtype=flat_output.dtype)
            flat_output[positions] = flat_output[positions] + residual

        exit_positions = (
            flat_actions == int(TAFEAction.EXIT)
        ).nonzero(as_tuple=False).flatten()
        if exit_positions.numel():
            flat_output[exit_positions] = 0
        return flat_output.reshape_as(default_output)


class TAFEGate(nn.Module):
    """Small task-utility and cost-aware tri-state controller."""

    def __init__(
        self,
        *,
        backbone_hidden_size: int,
        controller_hidden_size: int,
        num_layers: int,
        task_names: Sequence[str] = (),
        num_specialized_subsets: int = 2,
        routing_width: int = 1,
        condition_on_task: bool = True,
        use_hidden_state: bool = True,
        action_policy: str = "learned",
        static_boundary_layer: int | None = None,
        allow_exit: bool = True,
        initial_action_bias: Sequence[float] = (0.0, -0.1, -1.0),
        action_costs: Sequence[float] = (1.0, 0.75, 0.0),
        task_action_costs: Mapping[str, Sequence[float]] | None = None,
        lambda_cost: float = 0.0,
        soft_routing_train: bool = False,
        soft_routing_eval: bool = False,
        soft_routing_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if backbone_hidden_size < 1 or controller_hidden_size < 1:
            raise ValueError("hidden sizes must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if num_specialized_subsets < 1:
            raise ValueError("num_specialized_subsets must be positive")
        if not 1 <= routing_width <= num_specialized_subsets:
            raise ValueError(
                "routing_width must be within the specialized subset count"
            )
        if action_policy not in {"learned", "share", "decouple", "exit"}:
            raise ValueError(
                "action_policy must be learned, share, decouple, or exit"
            )
        if static_boundary_layer is not None and not (
            0 <= int(static_boundary_layer) <= num_layers
        ):
            raise ValueError(
                "static_boundary_layer must be within [0, num_layers]"
            )
        if len(initial_action_bias) != len(TAFEAction):
            raise ValueError("initial_action_bias must contain three values")
        if len(action_costs) != len(TAFEAction):
            raise ValueError("action_costs must contain three values")
        if any(float(value) < 0 for value in action_costs):
            raise ValueError("action_costs must be non-negative")
        if lambda_cost < 0:
            raise ValueError("lambda_cost must be non-negative")
        if soft_routing_temperature <= 0:
            raise ValueError("soft_routing_temperature must be positive")

        canonical_tasks: list[str] = []
        for task in ("unknown", *task_names):
            normalized = self.normalize_task(task)
            if normalized not in canonical_tasks:
                canonical_tasks.append(normalized)
        self.task_names = tuple(canonical_tasks)
        self.task_to_id = {
            task: index for index, task in enumerate(self.task_names)
        }
        self.backbone_hidden_size = int(backbone_hidden_size)
        self.controller_hidden_size = int(controller_hidden_size)
        self.num_layers = int(num_layers)
        self.condition_on_task = bool(condition_on_task)
        self.use_hidden_state = bool(use_hidden_state)
        self.action_policy = str(action_policy)
        self.static_boundary_layer = (
            int(static_boundary_layer)
            if static_boundary_layer is not None
            else None
        )
        self.allow_exit = bool(allow_exit)
        self.num_specialized_subsets = int(num_specialized_subsets)
        self.routing_width = int(routing_width)
        self.lambda_cost = float(lambda_cost)
        self.task_action_costs = {
            self.normalize_task(task): tuple(float(value) for value in costs)
            for task, costs in (task_action_costs or {}).items()
        }
        if any(len(costs) != len(TAFEAction) for costs in self.task_action_costs.values()):
            raise ValueError("task_action_costs must contain three values per task")
        if any(any(value < 0 for value in costs) for costs in self.task_action_costs.values()):
            raise ValueError("task_action_costs must be non-negative")
        self.soft_routing_train = bool(soft_routing_train)
        self.soft_routing_eval = bool(soft_routing_eval)
        self.soft_routing_temperature = float(soft_routing_temperature)

        # Runtime-only inference telemetry.  These Python counters are
        # intentionally not tensors/buffers, so they never enter a
        # checkpoint or affect optimization.  ``last_result`` alone is not
        # sufficient for auditing a generation run because it is overwritten
        # at every decoder layer.
        self._inference_action_counts = [0] * len(TAFEAction)
        self._inference_subset_counts = [0] * self.num_specialized_subsets
        self._inference_calls = 0
        self._inference_routed_tokens = 0
        self.last_training_action_entropy: torch.Tensor | None = None
        self.last_training_action_probabilities: torch.Tensor | None = None

        self.state_norm = nn.LayerNorm(backbone_hidden_size)
        self.state_proj = nn.Linear(
            backbone_hidden_size,
            controller_hidden_size,
        )
        self.layer_embedding = nn.Embedding(num_layers + 1, controller_hidden_size)
        self.kind_embedding = nn.Embedding(2, controller_hidden_size)
        self.budget_proj = nn.Linear(1, controller_hidden_size)
        self.task_embedding = (
            nn.Embedding(len(self.task_names), controller_hidden_size)
            if condition_on_task
            else None
        )
        self.utility_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(controller_hidden_size, controller_hidden_size),
            nn.SiLU(),
            nn.Linear(controller_hidden_size, len(TAFEAction)),
        )
        self.subset_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(controller_hidden_size, controller_hidden_size),
            nn.SiLU(),
            nn.Linear(controller_hidden_size, num_specialized_subsets),
        )
        self.register_buffer(
            "action_costs",
            torch.tensor(action_costs, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "initial_action_bias",
            torch.tensor(initial_action_bias, dtype=torch.float32),
            persistent=True,
        )
        self.reset_parameters()

    @staticmethod
    def normalize_task(task: str | None) -> str:
        return str(task or "unknown").strip().lower().replace("-", "_")

    def reset_parameters(self) -> None:
        self.state_norm.reset_parameters()
        self.state_proj.reset_parameters()
        self.layer_embedding.reset_parameters()
        self.kind_embedding.reset_parameters()
        self.budget_proj.reset_parameters()
        if self.task_embedding is not None:
            self.task_embedding.reset_parameters()
        for layer in (
            self.utility_head[1],
            self.utility_head[3],
            self.subset_head[1],
            self.subset_head[3],
        ):
            layer.reset_parameters()
        nn.init.zeros_(self.utility_head[3].weight)
        with torch.no_grad():
            self.utility_head[3].bias.copy_(self.initial_action_bias)
        nn.init.zeros_(self.subset_head[3].weight)
        nn.init.zeros_(self.subset_head[3].bias)

    def task_ids(
        self,
        task: str | Sequence[str] | torch.Tensor | None,
        *,
        count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(task, torch.Tensor):
            ids = task.to(device=device, dtype=torch.long).reshape(-1)
            if ids.numel() == 1:
                ids = ids.expand(count)
            if ids.numel() != count:
                raise ValueError("task tensor must match hidden-state rows")
            return ids
        if isinstance(task, str) or task is None:
            task_values = [task] * count
        else:
            task_values = list(task)
            if len(task_values) == 1:
                task_values = task_values * count
            if len(task_values) != count:
                raise ValueError("task sequence must match hidden-state rows")
        return torch.tensor(
            [
                self._task_id(value)
                for value in task_values
            ],
            device=device,
            dtype=torch.long,
        )

    def _task_id(self, task: str | None) -> int:
        """Resolve exact and benchmark-family task labels.

        Dataset metadata may carry fine-grained labels such as
        ``mmstar_l2_geometry`` or ``mathvista_a_okvqa_text`` while the gate
        manifest intentionally registers the benchmark-family labels.  Keep
        those examples on their family embedding instead of silently sending
        them all to ``unknown``.
        """

        normalized = self.normalize_task(task)
        exact = self.task_to_id.get(normalized)
        if exact is not None:
            return exact
        candidates = [
            (name, index)
            for name, index in self.task_to_id.items()
            if name != "unknown"
            and (
                normalized.startswith(f"{name}_")
                or normalized.startswith(f"{name}:")
            )
        ]
        if candidates:
            return max(candidates, key=lambda item: len(item[0]))[1]
        return 0

    def _kind_id(self, kind: str | int | None) -> int:
        if isinstance(kind, int):
            return max(0, min(1, kind))
        return 1 if str(kind or "").lower() == "generation" else 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        task: str | Sequence[str] | torch.Tensor | None = None,
        layer: int,
        kind: str | int | None = None,
        remaining_budget: float | torch.Tensor = 1.0,
        lambda_cost: float | None = None,
        hard_action_mask: torch.Tensor | None = None,
    ) -> TAFEOutput:
        if layer < 1 or layer > self.num_layers:
            raise ValueError(f"layer must be in [1, {self.num_layers}]")
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        if flat_hidden.shape[-1] != self.backbone_hidden_size:
            raise ValueError(
                "hidden state width does not match TAFE backbone width"
            )
        # The gate can be inside an FSDP activation-checkpointed module. In
        # that recomputation window its original parameters may be flattened
        # and ``next(self.parameters())`` is not guaranteed to exist. The
        # routed hidden states already carry the correct device/dtype for the
        # gathered gate parameters.
        features = flat_hidden
        if self.use_hidden_state:
            features = self.state_proj(self.state_norm(features))
        else:
            # Keep the projection and its parameters present for checkpoint
            # compatibility, but remove all input-dependent information.
            features = self.state_proj(torch.zeros_like(features))
        task_ids = self.task_ids(
            task,
            count=features.shape[0],
            device=features.device,
        )
        if self.task_embedding is not None:
            features = features + self.task_embedding(task_ids)
        layer_ids = torch.full(
            (features.shape[0],),
            int(layer),
            device=features.device,
            dtype=torch.long,
        )
        features = features + self.layer_embedding(layer_ids)
        kind_ids = torch.full(
            (features.shape[0],),
            self._kind_id(kind),
            device=features.device,
            dtype=torch.long,
        )
        features = features + self.kind_embedding(kind_ids)
        if isinstance(remaining_budget, torch.Tensor):
            budget = remaining_budget.to(
                device=features.device,
                dtype=features.dtype,
            ).reshape(-1)
            if budget.numel() == 1:
                budget = budget.expand(features.shape[0])
            if budget.numel() != features.shape[0]:
                raise ValueError("remaining_budget must match hidden rows")
        else:
            budget = torch.full(
                (features.shape[0],),
                float(remaining_budget),
                device=features.device,
                dtype=features.dtype,
            )
        features = features + self.budget_proj(budget[:, None])
        utilities = self.utility_head(features)
        subset_logits = self.subset_head(features)
        cost_weight = (
            self.lambda_cost if lambda_cost is None else float(lambda_cost)
        )
        costs = self.action_costs.to(
            device=utilities.device,
            dtype=utilities.dtype,
        )
        if self.task_action_costs:
            if isinstance(task, torch.Tensor):
                task_ids = task.to(
                    device=utilities.device,
                    dtype=torch.long,
                ).reshape(-1)
                if task_ids.numel() == 1:
                    task_ids = task_ids.expand(features.shape[0])
                if task_ids.numel() != features.shape[0]:
                    raise ValueError("task tensor must match hidden-state rows")
            else:
                task_ids = self.task_ids(
                    task,
                    count=features.shape[0],
                    device=utilities.device,
                )
            task_costs = costs.expand(features.shape[0], -1).clone()
            for task_name, task_values in self.task_action_costs.items():
                task_id = self._task_id(task_name)
                selected = task_ids == task_id
                if bool(selected.any().item()):
                    task_costs[selected] = utilities.new_tensor(task_values)
            costs = task_costs
        adjusted = utilities - cost_weight * costs
        if hard_action_mask is not None:
            mask = hard_action_mask.to(
                device=utilities.device,
                dtype=torch.bool,
            )
            if mask.shape != adjusted.shape:
                raise ValueError("hard_action_mask must match utility shape")
            adjusted = adjusted.masked_fill(~mask, -torch.inf)
        if not self.allow_exit:
            adjusted = adjusted.clone()
            adjusted[:, int(TAFEAction.EXIT)] = -torch.inf
        actions = adjusted.argmax(dim=-1)
        if self.routing_width == 1:
            subset_ids = subset_logits.argmax(dim=-1)
        else:
            subset_ids = subset_logits.topk(
                self.routing_width,
                dim=-1,
                largest=True,
                sorted=True,
            ).indices
        action_probabilities = None
        subset_probabilities = None
        self.last_training_action_entropy = None
        self.last_training_action_probabilities = None
        use_soft_routing = (
            self.training and self.soft_routing_train
        ) or (
            not self.training and self.soft_routing_eval
        )
        if use_soft_routing:
            temperature = self.soft_routing_temperature
            # Compute the auxiliary entropy in fp32 and via log_softmax.  A
            # direct ``p * log(p)`` in bf16 can produce NaNs for masked
            # actions because 0 * log(0) is evaluated as 0 * -inf.
            action_log_probabilities = torch.log_softmax(
                adjusted.float() / temperature,
                dim=-1,
            )
            action_probabilities = action_log_probabilities.exp()
            self.last_training_action_probabilities = action_probabilities
            if action_probabilities.numel() == 0:
                # Some modality branches legitimately contain no tokens.
                # Keep the auxiliary loss connected to the graph while
                # avoiding mean(empty) -> NaN.
                self.last_training_action_entropy = action_probabilities.sum()
            else:
                # Replace masked ``-inf`` values before multiplication.  A
                # torch.where around ``p * log(p)`` is insufficient because
                # autograd can still differentiate through the unselected
                # ``0 * -inf`` branch and create NaN gradients.
                finite_log_probabilities = torch.where(
                    torch.isfinite(action_log_probabilities),
                    action_log_probabilities,
                    torch.zeros_like(action_log_probabilities),
                )
                entropy_terms = (
                    action_probabilities * finite_log_probabilities
                )
                self.last_training_action_entropy = (
                    -entropy_terms.sum(dim=-1).mean()
                )
            subset_probabilities = torch.softmax(
                subset_logits / temperature,
                dim=-1,
            )
            if self.routing_width < self.num_specialized_subsets:
                topk_ids = subset_logits.topk(
                    self.routing_width,
                    dim=-1,
                    largest=True,
                    sorted=False,
                ).indices
                topk_mask = torch.zeros_like(subset_probabilities).scatter(
                    -1,
                    topk_ids,
                    1.0,
                )
                subset_probabilities = subset_probabilities * topk_mask
                subset_probabilities = subset_probabilities / subset_probabilities.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(torch.finfo(subset_probabilities.dtype).eps)
        if self.static_boundary_layer is not None:
            # Tuned-static boundary control: preserve shared computation
            # through k, then use the trained specialized residual bank for
            # every later layer. Utilities remain available for telemetry but
            # do not affect the fixed execution policy.
            forced_action = (
                TAFEAction.SHARE
                if layer <= self.static_boundary_layer
                else TAFEAction.DECOUPLE
            )
            actions = torch.full_like(actions, int(forced_action))
            if action_probabilities is not None:
                action_probabilities = torch.zeros_like(action_probabilities)
                action_probabilities[:, int(forced_action)] = 1.0
        elif self.action_policy != "learned":
            forced_action = {
                "share": TAFEAction.SHARE,
                "decouple": TAFEAction.DECOUPLE,
                "exit": TAFEAction.EXIT,
            }[self.action_policy]
            actions = torch.full_like(actions, int(forced_action))
            if action_probabilities is not None:
                action_probabilities = torch.zeros_like(action_probabilities)
                action_probabilities[:, int(forced_action)] = 1.0
        self.last_result = TAFEOutput(
            utilities=utilities.detach(),
            actions=actions.detach(),
            subset_ids=subset_ids.detach(),
            adjusted_utilities=adjusted.detach(),
            action_probabilities=(
                action_probabilities.detach()
                if action_probabilities is not None
                else None
            ),
            subset_probabilities=(
                subset_probabilities.detach()
                if subset_probabilities is not None
                else None
            ),
        )
        self.last_all_exit = bool(
            actions.numel() > 0
            and torch.all(actions == int(TAFEAction.EXIT)).item()
        )
        if not self.training:
            counts = torch.bincount(
                actions.detach().to(dtype=torch.long),
                minlength=len(TAFEAction),
            )
            self._inference_action_counts = [
                total + int(counts[action].item())
                for total, action in zip(
                    self._inference_action_counts,
                    TAFEAction,
                )
            ]
            decouple_subsets = subset_ids[
                actions == int(TAFEAction.DECOUPLE)
            ].reshape(-1)
            if decouple_subsets.numel():
                subset_counts = torch.bincount(
                    decouple_subsets.to(dtype=torch.long),
                    minlength=self.num_specialized_subsets,
                )
                self._inference_subset_counts = [
                    total + int(count)
                    for total, count in zip(
                        self._inference_subset_counts,
                        subset_counts.detach().cpu().tolist(),
                    )
                ]
            self._inference_calls += 1
            self._inference_routed_tokens += int(actions.numel())
        subset_shape = (
            (*hidden_states.shape[:-1], self.routing_width)
            if self.routing_width > 1
            else hidden_states.shape[:-1]
        )
        return TAFEOutput(
            utilities=utilities.reshape(
                *hidden_states.shape[:-1], len(TAFEAction)
            ),
            actions=actions.reshape(hidden_states.shape[:-1]),
            subset_ids=subset_ids.reshape(subset_shape),
            adjusted_utilities=adjusted.reshape(
                *hidden_states.shape[:-1], len(TAFEAction)
            ),
            action_probabilities=(
                action_probabilities.reshape(
                    *hidden_states.shape[:-1], len(TAFEAction)
                )
                if action_probabilities is not None
                else None
            ),
            subset_probabilities=(
                subset_probabilities.reshape(
                    *hidden_states.shape[:-1], self.num_specialized_subsets
                )
                if subset_probabilities is not None
                else None
            ),
        )

    @staticmethod
    def oracle_actions(
        quality_by_action: torch.Tensor,
        action_costs: torch.Tensor | Sequence[float],
        *,
        lambda_cost: float,
    ) -> torch.Tensor:
        """Choose oracle SHARE/DECOUPLE/EXIT actions from measured utility."""

        if quality_by_action.shape[-1] != len(TAFEAction):
            raise ValueError("quality_by_action must have three action columns")
        costs = torch.as_tensor(
            action_costs,
            device=quality_by_action.device,
            dtype=quality_by_action.dtype,
        )
        if costs.shape != (len(TAFEAction),):
            raise ValueError("action_costs must have shape [3]")
        return (quality_by_action - lambda_cost * costs).argmax(dim=-1)

    @staticmethod
    def utility_loss(
        predicted_utilities: torch.Tensor,
        measured_quality: torch.Tensor,
        action_costs: torch.Tensor | Sequence[float],
        *,
        lambda_cost: float,
    ) -> torch.Tensor:
        """Regress measured quality-cost utility for TAFE post-training."""

        costs = torch.as_tensor(
            action_costs,
            device=predicted_utilities.device,
            dtype=predicted_utilities.dtype,
        )
        target = measured_quality - lambda_cost * costs
        return torch.nn.functional.smooth_l1_loss(
            predicted_utilities,
            target,
        )

    @staticmethod
    def action_loss(
        predicted_utilities: torch.Tensor,
        measured_quality: torch.Tensor,
        action_costs: torch.Tensor | Sequence[float],
        *,
        lambda_cost: float,
    ) -> torch.Tensor:
        """Train TAFE against the measured quality-cost oracle action."""

        target = TAFEGate.oracle_actions(
            measured_quality,
            action_costs,
            lambda_cost=lambda_cost,
        )
        return torch.nn.functional.cross_entropy(
            predicted_utilities.reshape(-1, len(TAFEAction)),
            target.reshape(-1),
        )


class RoutedFFN(nn.Module):
    """Checkpoint-compatible FFN wrapper with TAFE SHARE/DECOUPLE/EXIT."""

    def __init__(
        self,
        base_ffn: nn.Module,
        gate: TAFEGate,
        *,
        layer: int,
        kind: str,
        adapter_rank: int,
        adapter_scale: float,
        num_specialized_subsets: int,
        routing_width: int = 1,
        parameter_budget: int | None = None,
    ) -> None:
        super().__init__()
        self.layer = int(layer)
        self.kind = str(kind)
        self.hidden_size = int(getattr(base_ffn, "hidden_size", 0))
        if self.hidden_size < 1:
            self.hidden_size = int(
                getattr(getattr(base_ffn, "gate_proj", None), "in_features", 0)
            )
        if self.hidden_size < 1:
            raise ValueError("base_ffn must expose hidden_size or gate_proj")
        self._gate_ref = weakref.ref(gate)

        # Keep the base Qwen2 parameter names unchanged. This lets an
        # original pretrained checkpoint populate gate/up/down directly.
        self._base_is_qwen_mlp = all(
            hasattr(base_ffn, name)
            for name in ("gate_proj", "up_proj", "down_proj")
        )
        if self._base_is_qwen_mlp:
            self.gate_proj = base_ffn.gate_proj
            self.up_proj = base_ffn.up_proj
            self.down_proj = base_ffn.down_proj
            self.act_fn = base_ffn.act_fn
        else:
            self.base_ffn = base_ffn

        self.specialized_banks = nn.ModuleList(
            [
                FFNResidualSubset(
                    self.hidden_size,
                    adapter_rank,
                    scale=adapter_scale,
                )
                for _ in range(num_specialized_subsets)
            ]
        )
        self.num_specialized_subsets = int(num_specialized_subsets)
        self.routing_width = int(routing_width)
        if not 1 <= self.routing_width <= self.num_specialized_subsets:
            raise ValueError(
                "routing_width must be within the specialized subset count"
            )
        # The decoder layer sets this before calling the wrapper when the
        # packed sequence is narrowed to text/vision modality rows.  Keeping
        # it as module state preserves the routing metadata during activation
        # checkpoint recomputation, when the depth-request context is gone.
        self._flexumm_route_token_indexes: torch.Tensor | None = None
        self._flexumm_checkpoint_sample_tasks: tuple[str, ...] | None = None
        self._flexumm_checkpoint_sample_lens: tuple[int, ...] | None = None
        self._flexumm_checkpoint_route_task: str | None = None
        self._flexumm_checkpoint_task_ids: tuple[int, ...] | None = None
        self._flexumm_checkpoint_task_id_labels: tuple[str, ...] | None = None
        if parameter_budget is not None:
            count = sum(
                parameter.numel()
                for parameter in self.specialized_banks.parameters()
            )
            if count > parameter_budget:
                raise ValueError(
                    "Routed FFN specialized capacity exceeds parameter budget: "
                    f"{count} > {parameter_budget}"
                )

    def _base_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._base_is_qwen_mlp:
            return self.down_proj(
                self.act_fn(self.gate_proj(hidden_states))
                * self.up_proj(hidden_states)
            )
        return self.base_ffn(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self._gate_ref()
        if gate is None:
            return self._base_forward(hidden_states)

        state = current_tafe_state()
        try:
            from .config import current_depth_request

            request = current_depth_request()
        except ImportError:
            request = None

        if request is not None:
            self._flexumm_checkpoint_sample_tasks = (
                tuple(request.sample_tasks)
                if request.sample_tasks is not None
                else None
            )
            self._flexumm_checkpoint_sample_lens = (
                tuple(int(length) for length in request.sample_lens)
                if request.sample_lens is not None
                else None
            )
            self._flexumm_checkpoint_route_task = request.task
        sample_tasks = (
            tuple(request.sample_tasks)
            if request is not None and request.sample_tasks is not None
            else self._flexumm_checkpoint_sample_tasks
        )
        sample_lens = (
            tuple(int(length) for length in request.sample_lens)
            if request is not None and request.sample_lens is not None
            else self._flexumm_checkpoint_sample_lens
        )
        route_task = (
            request.task
            if request is not None
            else self._flexumm_checkpoint_route_task
        )
        route_token_indexes = self._flexumm_route_token_indexes
        task = (
            state.task
            if state is not None
            else route_task
        )
        token_count = hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0]
        task_id_lookup = getattr(gate, "_task_id", None)
        if task_id_lookup is not None and sample_tasks is not None:
            if self._flexumm_checkpoint_task_id_labels != sample_tasks:
                self._flexumm_checkpoint_task_ids = tuple(
                    int(task_id_lookup(label)) for label in sample_tasks
                )
                self._flexumm_checkpoint_task_id_labels = sample_tasks
            sample_task_ids = _sample_task_ids_for_tokens(
                self._flexumm_checkpoint_task_ids,
                sample_lens,
                route_token_indexes,
                token_count,
                device=hidden_states.device,
                task_id_lookup=lambda value: int(value),
            )
            if sample_task_ids is not None:
                task = sample_task_ids
        else:
            per_token_tasks = _sample_tasks_for_tokens(
                sample_tasks,
                sample_lens,
                route_token_indexes,
                token_count,
                device=hidden_states.device,
            )
            if per_token_tasks is not None:
                task = per_token_tasks
        # The wrapper itself identifies the modality-specific route.  Do not
        # let a mixed request overwrite this value: understanding and
        # generation must use different TAFE controllers and FFN banks even
        # when both token streams share one transformer forward.
        kind = self.kind
        budget = (
            state.remaining_budget
            if state is not None
            else 1.0
        )
        result = gate(
            hidden_states,
            task=task,
            layer=self.layer,
            kind=kind,
            remaining_budget=budget,
        )
        output = self._base_forward(hidden_states)
        flat_output = output.reshape(-1, output.shape[-1]).clone()
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        actions = result.actions.reshape(-1)
        subset_ids = result.subset_ids.reshape(-1, self.routing_width)
        if (
            result.action_probabilities is not None
            and result.subset_probabilities is not None
        ):
            # Differentiable training path. SHARE and DECOUPLE retain the
            # shared FFN output; DECOUPLE adds a soft mixture of specialized
            # residual banks; EXIT contributes zero. Evaluation switches to
            # the hard argmax path below because the gate's ``training`` flag
            # is then false.
            action_probabilities = result.action_probabilities.reshape(
                -1, len(TAFEAction)
            )
            subset_probabilities = result.subset_probabilities.reshape(
                -1, self.num_specialized_subsets
            )
            specialized = flat_hidden.new_zeros(flat_output.shape)
            for bank_id, bank in enumerate(self.specialized_banks):
                specialized = specialized + subset_probabilities[:, bank_id, None] * bank(
                    flat_hidden
                )
            keep_shared = (
                action_probabilities[:, int(TAFEAction.SHARE)]
                + action_probabilities[:, int(TAFEAction.DECOUPLE)]
            )
            flat_output = (
                keep_shared[:, None] * flat_output
                + action_probabilities[:, int(TAFEAction.DECOUPLE), None]
                * specialized
            )
            output = flat_output.reshape_as(output).to(dtype=hidden_states.dtype)
            if state is not None:
                state.record(layer=self.layer, kind=self.kind, result=result)
            return output
        decouple_positions = (
            actions == int(TAFEAction.DECOUPLE)
        ).nonzero(as_tuple=False).flatten()
        selected_subsets = subset_ids[decouple_positions]
        for bank_id in selected_subsets.unique(sorted=True).tolist():
            if bank_id < 0 or bank_id >= self.num_specialized_subsets:
                raise ValueError(f"Invalid specialized FFN subset id {bank_id}")
            positions = decouple_positions[
                (selected_subsets == bank_id).any(dim=-1)
            ]
            residual = self.specialized_banks[bank_id](
                flat_hidden[positions]
            ).to(dtype=flat_output.dtype)
            if self.routing_width > 1:
                residual = residual / float(self.routing_width)
            flat_output[positions] = flat_output[positions] + residual
        exit_positions = (
            actions == int(TAFEAction.EXIT)
        ).nonzero(as_tuple=False).flatten()
        if exit_positions.numel():
            flat_output[exit_positions] = 0
        output = flat_output.reshape_as(output).to(dtype=hidden_states.dtype)
        if state is not None:
            state.record(layer=self.layer, kind=self.kind, result=result)
        return output


class RoutedAttention(nn.Module):
    """Apply TAFE residual routing to an existing attention output.

    This module is installed as a post-forward hook target rather than as a
    wrapper around ``self_attn``.  That preserves the original Qwen attention
    parameter names and lets the route operate on both BAGEL MoT streams in a
    packed batch.  Attention routing has no EXIT action: depth exits remain
    owned exclusively by the FFN TAFE controller.
    """

    def __init__(
        self,
        gate: TAFEGate,
        *,
        layer: int,
        kind: str,
        hidden_size: int,
        adapter_rank: int,
        adapter_scale: float,
        num_specialized_subsets: int,
        parameter_budget: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if num_specialized_subsets < 1:
            raise ValueError("num_specialized_subsets must be positive")
        self.layer = int(layer)
        self.kind = str(kind)
        self.hidden_size = int(hidden_size)
        self.num_specialized_subsets = int(num_specialized_subsets)
        self._gate_ref = weakref.ref(gate)
        self.specialized_banks = nn.ModuleList(
            [
                AttentionResidualSubset(
                    hidden_size,
                    adapter_rank,
                    scale=adapter_scale,
                )
                for _ in range(num_specialized_subsets)
            ]
        )
        self._flexumm_checkpoint_sample_tasks: tuple[str, ...] | None = None
        self._flexumm_checkpoint_sample_lens: tuple[int, ...] | None = None
        self._flexumm_checkpoint_route_task: str | None = None
        if parameter_budget is not None:
            count = sum(
                parameter.numel()
                for parameter in self.specialized_banks.parameters()
            )
            if count > parameter_budget:
                raise ValueError(
                    "Routed attention specialized capacity exceeds parameter "
                    f"budget: {count} > {parameter_budget}"
                )

    @staticmethod
    def _positions(
        token_indexes: torch.Tensor | None,
        token_count: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if token_indexes is None:
            return torch.arange(token_count, device=device, dtype=torch.long)
        indexes = token_indexes.to(device=device)
        if indexes.dtype == torch.bool:
            if indexes.numel() != token_count:
                raise ValueError(
                    "Boolean attention route indexes must match the output "
                    "token count"
                )
            return indexes.reshape(-1).nonzero(as_tuple=False).flatten()
        positions = indexes.to(dtype=torch.long).reshape(-1)
        if positions.numel() and (
            int(positions.min().item()) < 0
            or int(positions.max().item()) >= token_count
        ):
            raise ValueError("Attention route indexes exceed output rows")
        return positions

    def _resolve_task(
        self,
        hidden_states: torch.Tensor,
        token_indexes: torch.Tensor | None,
        *,
        task: str | None,
        sample_tasks: Sequence[str] | None,
        sample_lens: Sequence[int] | None,
    ) -> str | torch.Tensor | tuple[str, ...] | None:
        gate = self._gate_ref()
        if gate is None:
            return task
        token_count = hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0]
        task_id_lookup = getattr(gate, "_task_id", None)
        if task_id_lookup is not None and sample_tasks is not None:
            sample_task_ids = _sample_task_ids_for_tokens(
                sample_tasks,
                sample_lens,
                token_indexes,
                token_count,
                device=hidden_states.device,
                task_id_lookup=task_id_lookup,
            )
            if sample_task_ids is not None:
                return sample_task_ids
        per_token_tasks = _sample_tasks_for_tokens(
            sample_tasks,
            sample_lens,
            token_indexes,
            token_count,
            device=hidden_states.device,
        )
        return per_token_tasks if per_token_tasks is not None else task

    def apply(
        self,
        attention_output: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        token_indexes: torch.Tensor | None = None,
        task: str | None = None,
        sample_tasks: Sequence[str] | None = None,
        sample_lens: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Route selected packed rows and return an attention-shaped tensor."""

        gate = self._gate_ref()
        if gate is None:
            return attention_output
        if attention_output.shape[-1] != self.hidden_size:
            raise ValueError("attention output width does not match hidden size")
        flat_output = attention_output.reshape(-1, self.hidden_size)
        flat_hidden = hidden_states.reshape(-1, self.hidden_size)
        if flat_output.shape[0] != flat_hidden.shape[0]:
            raise ValueError("attention output and condition rows must align")
        positions = self._positions(
            token_indexes,
            flat_output.shape[0],
            device=flat_output.device,
        )
        if positions.numel() == 0:
            return attention_output

        try:
            from .config import current_depth_request

            request = current_depth_request()
        except ImportError:
            request = None
        if request is not None:
            self._flexumm_checkpoint_sample_tasks = (
                tuple(request.sample_tasks)
                if request.sample_tasks is not None
                else None
            )
            self._flexumm_checkpoint_sample_lens = (
                tuple(int(length) for length in request.sample_lens)
                if request.sample_lens is not None
                else None
            )
            self._flexumm_checkpoint_route_task = request.task
        sample_tasks = (
            tuple(request.sample_tasks)
            if request is not None and request.sample_tasks is not None
            else self._flexumm_checkpoint_sample_tasks
        )
        sample_lens = (
            tuple(int(length) for length in request.sample_lens)
            if request is not None and request.sample_lens is not None
            else self._flexumm_checkpoint_sample_lens
        )
        route_task = (
            request.task
            if request is not None
            else self._flexumm_checkpoint_route_task
        )
        state = current_tafe_state()
        route_task = state.task if state is not None else route_task
        resolved_task = self._resolve_task(
            flat_hidden[positions],
            positions,
            task=task if route_task is None else route_task,
            sample_tasks=sample_tasks,
            sample_lens=sample_lens,
        )
        result = gate(
            flat_hidden[positions],
            task=resolved_task,
            layer=self.layer,
            kind=self.kind,
            remaining_budget=(state.remaining_budget if state is not None else 1.0),
        )
        selected_output = flat_output[positions].clone()
        actions = result.actions.reshape(-1)
        subset_ids = result.subset_ids.reshape(-1)
        if (
            result.action_probabilities is not None
            and result.subset_probabilities is not None
        ):
            action_probabilities = result.action_probabilities.reshape(
                -1, len(TAFEAction)
            )
            subset_probabilities = result.subset_probabilities.reshape(
                -1, self.num_specialized_subsets
            )
            specialized = selected_output.new_zeros(selected_output.shape)
            for bank_id, bank in enumerate(self.specialized_banks):
                bank_output = bank(flat_hidden[positions]).to(
                    dtype=selected_output.dtype
                )
                specialized = specialized + (
                    subset_probabilities[:, bank_id, None] * bank_output
                ).to(dtype=selected_output.dtype)
            selected_output = selected_output + (
                action_probabilities[:, int(TAFEAction.DECOUPLE), None]
                * specialized
            ).to(dtype=selected_output.dtype)
        else:
            decouple_positions = (
                (actions == int(TAFEAction.DECOUPLE))
                .nonzero(as_tuple=False)
                .flatten()
            )
            for bank_id in subset_ids[decouple_positions].unique(
                sorted=True
            ).tolist():
                if bank_id < 0 or bank_id >= self.num_specialized_subsets:
                    raise ValueError(
                        f"Invalid specialized attention subset id {bank_id}"
                    )
                local_positions = decouple_positions[
                    subset_ids[decouple_positions] == bank_id
                ]
                selected_output[local_positions] = (
                    selected_output[local_positions]
                    + self.specialized_banks[bank_id](
                        flat_hidden[positions][local_positions]
                    ).to(dtype=selected_output.dtype)
                )
        # Attention gates are configured with allow_exit=False. Treat EXIT as
        # SHARE defensively if a custom gate is supplied; only the FFN route
        # is allowed to stop the decoder stack.
        flat_output = flat_output.clone()
        flat_output[positions] = selected_output.to(dtype=flat_output.dtype)
        if state is not None:
            state.record(layer=self.layer, kind=f"attention_{self.kind}", result=result)
        return flat_output.reshape_as(attention_output).to(
            dtype=attention_output.dtype
        )


__all__ = [
    "ACTION_NAMES",
    "AttentionResidualSubset",
    "FFNResidualSubset",
    "FFNSubsetPool",
    "RoutedAttention",
    "RoutedFFN",
    "TAFEAction",
    "TAFEExecutionState",
    "TAFEGate",
    "TAFEOutput",
    "current_tafe_state",
    "tafe_execution",
]
