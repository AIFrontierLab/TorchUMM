"""Task-specific fusion capsules and post-release realization modules.

The modules in this file are deliberately small.  They provide the first
executable form of FlexUMM's fusion-sealing idea without changing the
pretrained shared transformer when the feature is disabled.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _positions_from_indexes(
    indexes: torch.Tensor | None,
    *,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    if indexes is None or indexes.numel() == 0:
        return torch.arange(sequence_length, device=device, dtype=torch.long)
    if indexes.dtype == torch.bool:
        if indexes.numel() != sequence_length:
            raise ValueError(
                "Boolean token indexes must match the sequence length"
            )
        return indexes.to(device=device).nonzero(as_tuple=False).flatten()
    return indexes.to(device=device, dtype=torch.long)


def _sample_ids(
    positions: torch.Tensor,
    sample_lens: Sequence[int] | None,
) -> torch.Tensor:
    if sample_lens is None:
        return torch.zeros_like(positions)
    boundaries = torch.tensor(
        list(sample_lens), device=positions.device, dtype=torch.long
    ).cumsum(dim=0)[:-1]
    return torch.bucketize(positions, boundaries, right=True)


def _pool_by_sample(
    hidden_states: torch.Tensor,
    *,
    token_indexes: torch.Tensor | None = None,
    sample_lens: Sequence[int] | None = None,
    active_sample_ids: torch.Tensor | Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return pooled states, sample ids for selected positions, and batching."""

    positions = _positions_from_indexes(
        token_indexes,
        sequence_length=hidden_states.shape[0],
        device=hidden_states.device,
    )
    if positions.numel() == 0:
        raise ValueError("Cannot build a fusion capsule from no tokens")
    if sample_lens is None:
        selected = hidden_states[positions]
        return selected.float().mean(dim=0, keepdim=True), torch.zeros(
            positions.numel(), device=positions.device, dtype=torch.long
        ), False

    if sum(int(length) for length in sample_lens) < hidden_states.shape[0]:
        raise ValueError("sample_lens cannot be shorter than the packed state")
    ids = _sample_ids(positions, sample_lens)
    if active_sample_ids is None:
        active_ids = list(range(len(sample_lens)))
    elif isinstance(active_sample_ids, torch.Tensor):
        active_ids = active_sample_ids.detach().cpu().tolist()
    else:
        active_ids = [int(sample_id) for sample_id in active_sample_ids]
    active_ids = [int(sample_id) for sample_id in active_ids]
    pooled = []
    for sample_id in active_ids:
        sample_positions = positions[ids == sample_id]
        if sample_positions.numel() == 0:
            # A packed batch can contain a sample with no selected task tokens.
            # Fall back to its complete segment so every sample receives a
            # well-defined capsule and controller decision.
            start = sum(int(length) for length in sample_lens[:sample_id])
            end = min(
                start + int(sample_lens[sample_id]),
                hidden_states.shape[0],
            )
            sample_states = hidden_states[start:end]
        else:
            sample_states = hidden_states[sample_positions]
        if sample_states.numel() == 0:
            raise ValueError(f"Cannot build a capsule for sample {sample_id}")
        pooled.append(sample_states.float().mean(dim=0))
    return torch.stack(pooled, dim=0), ids, True


def _condition_per_position(
    capsule: torch.Tensor,
    *,
    sequence_length: int,
    sample_lens: Sequence[int] | None,
    device: torch.device,
    active_sample_ids: torch.Tensor | Sequence[int] | None = None,
) -> torch.Tensor:
    if capsule.ndim == 2:
        capsule = capsule.unsqueeze(0)
    summary = capsule.mean(dim=1)
    if sample_lens is None:
        return summary[:1].expand(sequence_length, -1)
    position_ids = torch.arange(sequence_length, device=device)
    ids = _sample_ids(position_ids, sample_lens)
    if active_sample_ids is None:
        return summary[ids]
    if isinstance(active_sample_ids, torch.Tensor):
        active_ids = active_sample_ids.to(device=device, dtype=torch.long)
    else:
        active_ids = torch.tensor(
            list(active_sample_ids), device=device, dtype=torch.long
        )
    local_ids = torch.searchsorted(active_ids, ids)
    output = summary.new_zeros((sequence_length, summary.shape[-1]))
    valid = local_ids < active_ids.numel()
    if valid.any():
        output[valid] = summary[local_ids[valid]]
    return output


class FusionCompiler(nn.Module):
    """Compress a task-relevant shared state into learned capsule tokens.

    This is a lightweight learned pooling compiler rather than a full extra
    transformer.  The bottleneck keeps its checkpoint and runtime overhead
    small enough for a first BAGEL implementation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_tokens: int,
        bottleneck_size: int,
        task_names: Sequence[str] = (),
        condition_on_task: bool = False,
        condition_on_fine_task: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_tokens < 1 or bottleneck_size < 1:
            raise ValueError("FusionCompiler dimensions must be positive")
        self.hidden_size = int(hidden_size)
        self.num_tokens = int(num_tokens)
        self.bottleneck_size = int(bottleneck_size)
        self.condition_on_task = bool(condition_on_task)
        self.condition_on_fine_task = bool(condition_on_fine_task)
        names = []
        for task in ("unknown", *task_names):
            name = str(task).strip().lower().replace("-", "_")
            if name not in names:
                names.append(name)
        self.task_names = tuple(names)
        self.task_to_id = {name: index for index, name in enumerate(names)}
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.query = nn.Parameter(torch.empty(num_tokens, hidden_size))
        self.task_embedding = (
            nn.Embedding(len(names), bottleneck_size)
            if self.condition_on_task
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.bias)
        if self.task_embedding is not None:
            if self.condition_on_fine_task:
                # Fine-task conditioning is an optional expansion over an
                # already-trained coarse controller.  Start as an exact
                # no-op so the expansion does not perturb the inherited
                # shared path before its metadata has been learned.
                nn.init.zeros_(self.task_embedding.weight)
            else:
                nn.init.normal_(self.task_embedding.weight, mean=0.0, std=0.02)

    def task_id(self, task: str | None) -> int:
        name = str(task or "unknown").strip().lower().replace("-", "_")
        if self.condition_on_fine_task and name in self.task_to_id:
            return self.task_to_id[name]
        for prefix, family in (
            ("mmbench_en", "mmbench_en"),
            ("mmbench_cn", "mmbench_cn"),
            ("mmmu", "mmmu"),
            ("mathvista", "mathvista"),
            ("mmstar", "mmstar"),
        ):
            if name == prefix or name.startswith(prefix + "_"):
                name = family
                break
        return self.task_to_id.get(name, self.task_to_id["unknown"])

    def _task_ids(
        self,
        task: str | Sequence[str] | None,
        *,
        batch_size: int,
        sample_lens: Sequence[int] | None,
        active_sample_ids: torch.Tensor | Sequence[int] | None,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(task, str) or task is None:
            return torch.full(
                (batch_size,), self.task_id(task), device=device, dtype=torch.long
            )
        tasks = list(task)
        if (
            active_sample_ids is not None
            and len(tasks) == len(sample_lens or ())
        ):
            if isinstance(active_sample_ids, torch.Tensor):
                active_ids = active_sample_ids.detach().cpu().tolist()
            else:
                active_ids = list(active_sample_ids)
            tasks = [tasks[int(sample_id)] for sample_id in active_ids]
        if len(tasks) != batch_size:
            raise ValueError("Fusion compiler tasks must match the pooled batch")
        return torch.tensor(
            [self.task_id(item) for item in tasks],
            device=device,
            dtype=torch.long,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        token_indexes: torch.Tensor | None = None,
        sample_lens: Sequence[int] | None = None,
        active_sample_ids: torch.Tensor | Sequence[int] | None = None,
        task: str | Sequence[str] | None = None,
    ) -> torch.Tensor:
        pooled, _, _ = _pool_by_sample(
            hidden_states,
            token_indexes=token_indexes,
            sample_lens=sample_lens,
            active_sample_ids=active_sample_ids,
        )
        parameter = self.query
        pooled = pooled.to(device=parameter.device, dtype=parameter.dtype)
        bottleneck = self.down(self.norm(pooled))
        if self.task_embedding is not None:
            task_ids = self._task_ids(
                task,
                batch_size=pooled.shape[0],
                sample_lens=sample_lens,
                active_sample_ids=active_sample_ids,
                device=parameter.device,
            )
            bottleneck = bottleneck + self.task_embedding(task_ids)
        content = self.up(F.silu(bottleneck))
        return self.query.unsqueeze(0) + content.unsqueeze(1)


class FusionRealizationAdapter(nn.Module):
    """Continue task realization after the shared path has been sealed.

    The final projection is zero initialized, making the module an exact
    no-op at initialization.  This allows a new sealing configuration to be
    loaded on top of an existing checkpoint while the controller is trained
    conservatively toward ``continue``.
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
            raise ValueError("FusionRealizationAdapter dimensions must be positive")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.scale = float(scale)
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.condition = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.condition.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        capsule: torch.Tensor,
        *,
        token_indexes: torch.Tensor | None = None,
        sample_lens: Sequence[int] | None = None,
        active_sample_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("Fusion realization expects [tokens, hidden] states")
        condition = _condition_per_position(
            capsule,
            sequence_length=hidden_states.shape[0],
            sample_lens=sample_lens,
            device=hidden_states.device,
            active_sample_ids=active_sample_ids,
        ).to(device=hidden_states.device, dtype=hidden_states.dtype)
        positions = _positions_from_indexes(
            token_indexes,
            sequence_length=hidden_states.shape[0],
            device=hidden_states.device,
        )
        selected = hidden_states[positions]
        selected_condition = condition[positions]
        parameter = self.down.weight
        selected_for_module = selected.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        condition_for_module = selected_condition.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        delta = self.up(
            F.silu(
                self.down(self.norm(selected_for_module))
                + self.condition(condition_for_module)
            )
        ) * self.scale
        output = hidden_states.clone()
        output[positions] = selected + delta.to(dtype=selected.dtype)
        return output


class FusionReleaseController(nn.Module):
    """Predict continue/seal values from shared state and task capsule."""

    def __init__(
        self,
        *,
        hidden_size: int,
        controller_hidden_size: int,
        num_layers: int,
        task_names: Sequence[str],
        initial_seal_bias: float = -6.0,
        condition_on_task: bool = True,
        condition_on_fine_task: bool = False,
        condition_on_realization_delta: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.controller_hidden_size = int(controller_hidden_size)
        self.num_layers = int(num_layers)
        self.condition_on_task = bool(condition_on_task)
        self.condition_on_fine_task = bool(condition_on_fine_task)
        self.condition_on_realization_delta = bool(
            condition_on_realization_delta
        )
        names = []
        for task in ("unknown", *task_names):
            name = str(task).strip().lower().replace("-", "_")
            if name not in names:
                names.append(name)
        self.task_names = tuple(names)
        self.task_to_id = {name: index for index, name in enumerate(names)}
        self.state_norm = nn.LayerNorm(hidden_size)
        self.capsule_norm = nn.LayerNorm(hidden_size)
        self.state_proj = nn.Linear(hidden_size, controller_hidden_size)
        self.capsule_proj = nn.Linear(hidden_size, controller_hidden_size)
        self.delta_norm = (
            nn.LayerNorm(hidden_size)
            if self.condition_on_realization_delta
            else None
        )
        self.delta_proj = (
            nn.Linear(hidden_size, controller_hidden_size)
            if self.condition_on_realization_delta
            else None
        )
        self.depth_embedding = nn.Embedding(num_layers + 1, controller_hidden_size)
        self.kind_embedding = nn.Embedding(2, controller_hidden_size)
        self.timestep_proj = nn.Linear(3, controller_hidden_size)
        self.task_embedding = (
            nn.Embedding(len(names), controller_hidden_size)
            if condition_on_task
            else None
        )
        self.output = nn.Sequential(
            nn.SiLU(),
            nn.Linear(controller_hidden_size, controller_hidden_size),
            nn.SiLU(),
            nn.Linear(controller_hidden_size, 2),
        )
        self.initial_seal_bias = float(initial_seal_bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.state_norm.reset_parameters()
        self.capsule_norm.reset_parameters()
        self.state_proj.reset_parameters()
        self.capsule_proj.reset_parameters()
        if self.delta_norm is not None:
            self.delta_norm.reset_parameters()
        if self.delta_proj is not None:
            self.delta_proj.reset_parameters()
        self.depth_embedding.reset_parameters()
        self.kind_embedding.reset_parameters()
        self.timestep_proj.reset_parameters()
        if self.task_embedding is not None:
            self.task_embedding.reset_parameters()
        self.output[1].reset_parameters()
        self.output[3].reset_parameters()
        if self.task_embedding is not None and self.condition_on_fine_task:
            # Preserve the inherited coarse policy when fine-task rows are
            # newly introduced during a shape-compatible continuation.
            nn.init.zeros_(self.task_embedding.weight)
        with torch.no_grad():
            self.output[-1].bias.zero_()
            self.output[-1].bias[1] = self.initial_seal_bias

    def task_id(self, task: str | None) -> int:
        name = str(task or "unknown").strip().lower().replace("-", "_")
        if self.condition_on_fine_task and name in self.task_to_id:
            return self.task_to_id[name]
        # Training metadata is fine-grained (for example
        # ``mmmu_subject_physics``), while inference may provide either the
        # family or the fine-grained task.  The quality gate only needs a
        # compact benchmark-family condition, so map both forms to one shared
        # controller embedding.
        for prefix, family in (
            ("mmbench_en", "mmbench_en"),
            ("mmbench_cn", "mmbench_cn"),
            ("mmmu", "mmmu"),
            ("mathvista", "mathvista"),
            ("mmstar", "mmstar"),
        ):
            if name == prefix or name.startswith(prefix + "_"):
                name = family
                break
        return self.task_to_id.get(name, self.task_to_id["unknown"])

    def forward(
        self,
        hidden_states: torch.Tensor,
        capsule: torch.Tensor,
        *,
        depth: int,
        kind: str,
        task: str | Sequence[str] | None,
        timestep: float | torch.Tensor | None = None,
        token_indexes: torch.Tensor | None = None,
        sample_lens: Sequence[int] | None = None,
        active_sample_ids: torch.Tensor | Sequence[int] | None = None,
        realization_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled, _, per_sample = _pool_by_sample(
            hidden_states,
            token_indexes=token_indexes,
            sample_lens=sample_lens,
            active_sample_ids=active_sample_ids,
        )
        if capsule.ndim == 2:
            capsule = capsule.unsqueeze(0)
        capsule_summary = capsule.float().mean(dim=1)
        parameter = self.state_proj.weight
        pooled = pooled.to(device=parameter.device, dtype=parameter.dtype)
        capsule_summary = capsule_summary.to(
            device=parameter.device, dtype=parameter.dtype
        )
        if self.condition_on_realization_delta:
            if realization_delta is None:
                raise ValueError(
                    "Fusion controller requires realization_delta when "
                    "condition_on_realization_delta is enabled"
                )
            delta_summary, _, delta_per_sample = _pool_by_sample(
                realization_delta,
                token_indexes=token_indexes,
                sample_lens=sample_lens,
                active_sample_ids=active_sample_ids,
            )
            delta_summary = delta_summary.to(
                device=parameter.device, dtype=parameter.dtype
            )
        else:
            delta_summary = None
        batch_size = pooled.shape[0]
        depth_ids = torch.full(
            (batch_size,), int(depth), device=parameter.device, dtype=torch.long
        )
        kind_ids = torch.full(
            (batch_size,),
            1 if kind == "generation" else 0,
            device=parameter.device,
            dtype=torch.long,
        )
        if timestep is None:
            timestep_value = 0.0
        elif isinstance(timestep, torch.Tensor):
            values = timestep.detach().float()
            timestep_value = (
                0.0 if values.numel() == 0 else float(values.min().item())
            )
        else:
            timestep_value = float(timestep)
        timestep_value = min(max(timestep_value, 0.0), 1.0)
        timestep_features = parameter.new_tensor(
            [
                timestep_value,
                math.sin(math.pi * timestep_value),
                math.cos(math.pi * timestep_value),
            ]
        ).expand(batch_size, -1)
        features = (
            self.state_proj(self.state_norm(pooled))
            + self.capsule_proj(self.capsule_norm(capsule_summary))
            + self.depth_embedding(depth_ids)
            + self.kind_embedding(kind_ids)
            + self.timestep_proj(timestep_features)
        )
        if self.delta_proj is not None and self.delta_norm is not None:
            features = features + self.delta_proj(
                self.delta_norm(delta_summary)
            )
        if self.task_embedding is not None:
            if isinstance(task, str) or task is None:
                task_ids = torch.full(
                    (batch_size,),
                    self.task_id(task),
                    device=parameter.device,
                    dtype=torch.long,
                )
            else:
                tasks = list(task)
                if (
                    active_sample_ids is not None
                    and len(tasks) == len(sample_lens or ())
                ):
                    if isinstance(active_sample_ids, torch.Tensor):
                        active_ids = active_sample_ids.detach().cpu().tolist()
                    else:
                        active_ids = list(active_sample_ids)
                    tasks = [tasks[int(sample_id)] for sample_id in active_ids]
                if len(tasks) != batch_size:
                    raise ValueError(
                        "Fusion controller tasks must match the pooled batch"
                    )
                task_ids = torch.tensor(
                    [self.task_id(item) for item in tasks],
                    device=parameter.device,
                    dtype=torch.long,
                )
            features = features + self.task_embedding(task_ids)
        logits = self.output(features)
        return logits if per_sample else logits[:1]


__all__ = [
    "FusionCompiler",
    "FusionRealizationAdapter",
    "FusionReleaseController",
]
