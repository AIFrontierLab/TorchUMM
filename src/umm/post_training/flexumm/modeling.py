# Copyright 2026 FlexUMM contributors
# SPDX-License-Identifier: Apache-2.0
#
# The forward-path structure in this file is adapted from ByteDance BAGEL,
# which is distributed under the Apache License 2.0.

"""Checkpoint-compatible dynamic depth for BAGEL's shared Qwen2 backbone."""

from __future__ import annotations

import inspect
import fnmatch
import weakref
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .bootstrap import ensure_bagel_importable
from .config import (
    DepthRequest,
    DynamicDepthConfig,
    DynamicDepthController,
    current_depth_request,
)
from .router import CandidateDepthRouter
from .fusion import (
    FusionCompiler,
    FusionRealizationAdapter,
    FusionReleaseController,
)
from .tafe import (
    RoutedAttention,
    RoutedFFN,
    TAFEAction,
    TAFEGate,
)

ensure_bagel_importable()

from modeling.bagel.bagel import Bagel as _Bagel  # noqa: E402
from modeling.bagel.qwen2_navit import (  # noqa: E402
    BaseNavitOutputWithPast,
    Qwen2Model as _Qwen2Model,
)


_BAGEL_FORWARD_SIGNATURE = inspect.signature(_Bagel.forward)
_BAGEL_FLOW_SIGNATURE = inspect.signature(_Bagel._forward_flow)


def _nonempty(indexes: torch.Tensor | None) -> bool:
    return indexes is not None and indexes.numel() > 0


def _clone_optional(tensor: torch.Tensor | None) -> torch.Tensor | None:
    return None if tensor is None else tensor.detach().clone()


def _positions_from_indexes(
    indexes: torch.Tensor, *, sequence_length: int
) -> torch.Tensor:
    if indexes.dtype == torch.bool:
        if indexes.numel() != sequence_length:
            raise ValueError("Boolean indexes must match packed sequence length")
        return indexes.nonzero(as_tuple=False).flatten()
    return indexes.to(dtype=torch.long)


def _sample_ids_for_positions(
    positions: torch.Tensor, sample_lens: list[int]
) -> torch.Tensor:
    boundaries = torch.tensor(
        sample_lens, device=positions.device, dtype=torch.long
    ).cumsum(dim=0)[:-1]
    return torch.bucketize(positions, boundaries, right=True)


def _router_prefix_positions(
    understanding_indexes: torch.Tensor | None,
    label_positions: torch.Tensor,
    label_sample_ids: torch.Tensor,
    sample_lens: list[int],
    active_sample_ids: torch.Tensor,
) -> torch.Tensor:
    """Select inference-available conditioning states for router training.

    A causal CE position contains only the prefix and the shifted BOS/current
    input token needed to predict its label.  Later CE positions contain
    teacher-forced answer tokens that are unavailable when the first routing
    decision is made.  Pooling through only the first CE position therefore
    keeps router features aligned with inference.
    """

    sequence_length = sum(int(length) for length in sample_lens)
    if understanding_indexes is None:
        understanding_positions = torch.arange(
            sequence_length, device=label_positions.device
        )
    else:
        understanding_positions = _positions_from_indexes(
            understanding_indexes, sequence_length=sequence_length
        ).to(device=label_positions.device)
    understanding_sample_ids = _sample_ids_for_positions(
        understanding_positions, sample_lens
    )

    selected = []
    for sample_id in active_sample_ids.detach().cpu().tolist():
        sample_labels = label_positions[label_sample_ids == sample_id]
        if sample_labels.numel() == 0:
            continue
        first_label = sample_labels.min()
        sample_prefix = understanding_positions[
            (understanding_sample_ids == sample_id)
            & (understanding_positions <= first_label)
        ]
        if sample_prefix.numel() == 0:
            # Defensive fallback for unusual data layouts where the supervised
            # state is not marked as an understanding token.
            sample_prefix = first_label.reshape(1)
        selected.append(sample_prefix)
    if not selected:
        raise ValueError("No inference-available router prefix positions found")
    return torch.cat(selected)


@dataclass
class _ReplaySegment:
    """One cache-update query retained at its current raw hidden depth."""

    hidden: torch.Tensor | None
    depth: int
    query_lens: torch.Tensor
    position_ids: torch.Tensor
    query_indexes: torch.Tensor
    key_values_lens: torch.Tensor
    key_value_indexes: torch.Tensor
    is_causal: bool
    mode: str
    vae_token_indexes: torch.Tensor | None = None
    text_indexes: torch.Tensor | None = None


@dataclass
class _ReplayState:
    segments: list[_ReplaySegment] = field(default_factory=list)


def _normalize_delta_task(task: str | None) -> str:
    return (task or "").strip().lower().replace("-", "_")


def _task_matches_delta_patterns(
    task: str | None,
    patterns: Sequence[str] | None,
) -> bool:
    if not patterns:
        return True
    task_name = _normalize_delta_task(task)
    if not task_name:
        return False
    return any(
        fnmatch.fnmatchcase(
            task_name,
            _normalize_delta_task(pattern),
        )
        for pattern in patterns
    )


def _request_matches_delta_patterns(
    request,
    patterns: Sequence[str] | None,
) -> bool:
    if not patterns:
        return True
    if request is None:
        return False
    if _task_matches_delta_patterns(request.task, patterns):
        return True
    sample_tasks = getattr(request, "sample_tasks", None) or ()
    return any(_task_matches_delta_patterns(task, patterns) for task in sample_tasks)


def _route_matches_delta_patterns(
    task: str | None,
    sample_tasks: Sequence[str] | None,
    patterns: Sequence[str] | None,
) -> bool:
    if not patterns:
        return True
    return _task_matches_delta_patterns(task, patterns) or any(
        _task_matches_delta_patterns(sample_task, patterns)
        for sample_task in (sample_tasks or ())
    )


def _route_token_positions_for_patterns(
    request,
    patterns: Sequence[str] | None,
    *,
    token_indexes: torch.Tensor | None,
    sequence_length: int,
) -> torch.Tensor:
    """Return packed token positions whose sample task matches ``patterns``.

    A packed batch may contain several understanding tasks.  A route match at
    the request level is therefore insufficient: applying a named delta to
    every packed token would leak one task's adapter into the other tasks.
    ``DepthRequest.sample_lens`` provides the packed boundaries needed to
    construct a sample-wise mask.  In single-sample inference, the task-level
    fallback remains equivalent to the old behavior.
    """

    positions = (
        torch.arange(sequence_length, device=request.und_token_indexes.device)
        if token_indexes is None and request is not None and request.und_token_indexes is not None
        else (
            torch.arange(sequence_length)
            if token_indexes is None
            else _positions_from_indexes(
                token_indexes,
                sequence_length=sequence_length,
            )
        )
    )
    if token_indexes is None and request is not None and request.und_token_indexes is None:
        positions = torch.arange(sequence_length)
    if not patterns:
        return positions
    if request is None:
        return positions.new_empty((0,), dtype=torch.long)

    sample_tasks = getattr(request, "sample_tasks", None)
    sample_lens = getattr(request, "sample_lens", None)
    if sample_tasks is not None and sample_lens is not None:
        sample_tasks = tuple(sample_tasks)
        sample_lens = tuple(int(length) for length in sample_lens)
        if len(sample_tasks) == len(sample_lens) and sum(sample_lens) == sequence_length:
            boundaries = torch.tensor(
                sample_lens,
                device=positions.device,
                dtype=torch.long,
            ).cumsum(dim=0)[:-1]
            sample_ids = torch.bucketize(positions, boundaries, right=True)
            matching_samples = torch.tensor(
                [
                    any(
                        _task_matches_delta_patterns(task, patterns)
                        for task in (sample_task,)
                    )
                    for sample_task in sample_tasks
                ],
                device=positions.device,
                dtype=torch.bool,
            )
            return positions[matching_samples[sample_ids]]

    if _task_matches_delta_patterns(getattr(request, "task", None), patterns):
        return positions
    return positions.new_empty((0,), dtype=torch.long)


def _add_routed_delta(
    output: torch.Tensor,
    inputs: torch.Tensor,
    request,
    patterns: Sequence[str] | None,
    delta_fn,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Add a task-bank delta only to matching samples in a packed batch.

    The projection wrappers see the complete packed sequence.  Checking only
    whether *any* sample matches a bank would therefore apply a category bank
    to every sample in a mixed batch.  Keep the fast full-sequence path for a
    homogeneous batch, and compute the delta only for the matching token
    positions otherwise.
    """

    positions = _route_token_positions_for_patterns(
        request,
        patterns,
        token_indexes=None,
        sequence_length=inputs.shape[0],
    )
    if positions.numel() == 0:
        return output
    if positions.numel() == inputs.shape[0]:
        return output + delta_fn(inputs) * scale
    updated = output.clone()
    updated[positions] = output[positions] + delta_fn(inputs[positions]) * scale
    return updated


def _checkpoint_route_request(module: nn.Module, request) -> DepthRequest | None:
    """Recreate packed-route metadata during activation-checkpoint replay."""

    if request is not None:
        return request
    kind = getattr(module, "_flexumm_checkpoint_route_kind", None)
    if kind is None:
        return None
    return DepthRequest(
        task=getattr(module, "_flexumm_checkpoint_route_task", None),
        sample_tasks=getattr(module, "_flexumm_checkpoint_sample_tasks", None),
        sample_lens=getattr(module, "_flexumm_checkpoint_sample_lens", None),
        kind=kind,
        und_token_indexes=getattr(
            module,
            "_flexumm_checkpoint_und_token_indexes",
            None,
        ),
    )


class UnderstandingResidualAdapter(nn.Module):
    """Small mode-isolated residual used only by understanding forwards."""

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        *,
        scale: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.scale = scale
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = torch.nn.functional.rms_norm(
            hidden_states,
            (self.hidden_size,),
        )
        return self.up(
            torch.nn.functional.silu(self.down(normalized))
        ) * self.scale


class UnderstandingDepthFusion(nn.Module):
    """Fuse an earlier semantic state into a later understanding state.

    The output projection starts at zero, so enabling the module on an
    inherited checkpoint is exactly behavior preserving before post-training.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        *,
        scale: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.scale = scale
        self.source_down = nn.Linear(hidden_size, rank, bias=False)
        self.target_down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.source_down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.target_down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(
        self,
        source_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        source = torch.nn.functional.rms_norm(
            source_hidden_states,
            (self.hidden_size,),
        )
        target = torch.nn.functional.rms_norm(
            target_hidden_states,
            (self.hidden_size,),
        )
        joint = torch.nn.functional.silu(
            self.source_down(source) + self.target_down(target)
        )
        return self.up(joint) * self.scale


class UnderstandingLoRABank(nn.Module):
    """Reusable task-scoped low-rank delta bank."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        scale: float,
        task_patterns: Sequence[str] | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.scale = scale
        self.task_patterns = tuple(task_patterns or ())
        self.down = nn.Linear(
            in_features,
            rank,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.up = nn.Linear(
            rank,
            out_features,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(inputs)) * self.scale


class UnderstandingLoRALinear(nn.Module):
    """A linear projection with a route-isolated understanding LoRA delta.

    The inherited weight and bias retain their original state-dict names.
    New low-rank weights start as an exact no-op and are skipped for every
    generation route, including generation prompt encoding.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        *,
        scale: float,
        task_patterns: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.scale = scale
        self.understanding_delta_task_patterns = tuple(task_patterns or ())
        self.understanding_extra_delta_task_patterns: tuple[str, ...] = ()
        self.understanding_extra_attention_lora_rank = 0
        self.understanding_extra_attention_lora_scale = 1.0
        self.weight = base.weight
        self.bias = base.bias
        self.understanding_named_attention_loras = nn.ModuleDict()
        self.understanding_attention_lora_down = nn.Linear(
            self.in_features,
            rank,
            bias=False,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        self.understanding_attention_lora_up = nn.Linear(
            rank,
            self.out_features,
            bias=False,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.normal_(
            self.understanding_attention_lora_down.weight,
            mean=0.0,
            std=0.02,
        )
        nn.init.zeros_(self.understanding_attention_lora_up.weight)

    def configure_extra_lora(
        self,
        rank: int,
        *,
        scale: float,
        task_patterns: Sequence[str] | None = None,
    ) -> None:
        if rank <= 0:
            self.understanding_extra_delta_task_patterns = ()
            self.understanding_extra_attention_lora_rank = 0
            return
        existing_down = getattr(
            self,
            "understanding_extra_attention_lora_down",
            None,
        )
        existing_up = getattr(
            self,
            "understanding_extra_attention_lora_up",
            None,
        )
        if existing_down is not None or existing_up is not None:
            if (
                not isinstance(existing_down, nn.Linear)
                or not isinstance(existing_up, nn.Linear)
                or existing_down.out_features != rank
                or existing_up.in_features != rank
            ):
                raise ValueError(
                    "Cannot reconfigure existing extra understanding "
                    "attention LoRA with a different rank"
                )
        else:
            self.understanding_extra_attention_lora_down = nn.Linear(
                self.in_features,
                rank,
                bias=False,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
            self.understanding_extra_attention_lora_up = nn.Linear(
                rank,
                self.out_features,
                bias=False,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
            self.reset_extra_lora_parameters()
        self.understanding_extra_delta_task_patterns = tuple(
            task_patterns or ()
        )
        self.understanding_extra_attention_lora_rank = rank
        self.understanding_extra_attention_lora_scale = scale

    def reset_extra_lora_parameters(self) -> None:
        down = getattr(self, "understanding_extra_attention_lora_down", None)
        up = getattr(self, "understanding_extra_attention_lora_up", None)
        if isinstance(down, nn.Linear):
            nn.init.normal_(down.weight, mean=0.0, std=0.02)
        if isinstance(up, nn.Linear):
            nn.init.zeros_(up.weight)

    def configure_named_lora(
        self,
        name: str,
        rank: int,
        *,
        scale: float,
        task_patterns: Sequence[str] | None = None,
    ) -> None:
        if rank <= 0:
            if name in self.understanding_named_attention_loras:
                del self.understanding_named_attention_loras[name]
            return
        existing = (
            self.understanding_named_attention_loras[name]
            if name in self.understanding_named_attention_loras
            else None
        )
        if existing is not None:
            if not isinstance(existing, UnderstandingLoRABank) or (
                existing.rank != rank
            ):
                raise ValueError(
                    "Cannot reconfigure existing named understanding "
                    f"attention LoRA bank {name!r} with a different rank"
                )
            existing.scale = scale
            existing.task_patterns = tuple(task_patterns or ())
            return
        self.understanding_named_attention_loras[name] = UnderstandingLoRABank(
            self.in_features,
            self.out_features,
            rank,
            scale=scale,
            task_patterns=task_patterns,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

    def reset_named_lora_parameters(self) -> None:
        for bank in self.understanding_named_attention_loras.values():
            if isinstance(bank, UnderstandingLoRABank):
                bank.reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = F.linear(inputs, self.weight, self.bias)
        request = current_depth_request()
        if self.training:
            # Non-reentrant activation checkpointing recomputes this module
            # during ``loss.backward()``, after BAGEL's route context has
            # closed. Cache the route observed by the original forward so
            # recomputation follows exactly the same graph.
            if request is not None:
                self._flexumm_checkpoint_route_kind = request.kind
                self._flexumm_checkpoint_route_task = request.task
                self._flexumm_checkpoint_sample_tasks = request.sample_tasks
                self._flexumm_checkpoint_sample_lens = request.sample_lens
                self._flexumm_checkpoint_und_token_indexes = request.und_token_indexes
            route_kind = (
                request.kind
                if request is not None
                else getattr(
                    self,
                    "_flexumm_checkpoint_route_kind",
                    None,
                )
            )
            route_task = (
                request.task
                if request is not None
                else getattr(self, "_flexumm_checkpoint_route_task", None)
            )
            route_sample_tasks = (
                request.sample_tasks
                if request is not None
                else getattr(self, "_flexumm_checkpoint_sample_tasks", None)
            )
        else:
            route_kind = request.kind if request is not None else None
            route_task = request.task if request is not None else None
            route_sample_tasks = (
                request.sample_tasks if request is not None else None
            )
        route_request = _checkpoint_route_request(self, request)
        patterns = self.understanding_delta_task_patterns
        if route_kind != "understanding":
            return output
        output = _add_routed_delta(
            output,
            inputs,
            route_request,
            patterns,
            lambda values: self.understanding_attention_lora_up(
                self.understanding_attention_lora_down(values)
            ),
            scale=self.scale,
        )
        extra_patterns = getattr(
            self,
            "understanding_extra_delta_task_patterns",
            (),
        )
        extra_down = getattr(
            self,
            "understanding_extra_attention_lora_down",
            None,
        )
        extra_up = getattr(
            self,
            "understanding_extra_attention_lora_up",
            None,
        )
        if (
            bool(extra_patterns)
            and isinstance(extra_down, nn.Linear)
            and isinstance(extra_up, nn.Linear)
        ):
            output = _add_routed_delta(
                output,
                inputs,
                route_request,
                extra_patterns,
                lambda values: extra_up(extra_down(values)),
                scale=getattr(
                    self,
                    "understanding_extra_attention_lora_scale",
                    1.0,
                ),
            )
        for bank in self.understanding_named_attention_loras.values():
            output = _add_routed_delta(
                output,
                inputs,
                route_request,
                bank.task_patterns,
                bank,
            )
        return output


class UnderstandingMLPLoRALinear(nn.Module):
    """A linear projection with an understanding-only MLP LoRA delta.

    BAGEL reuses the understanding MLP expert to encode text-to-image prompts.
    Checking the capability route rather than the expert name keeps these
    deltas out of both image generation and generation prompt encoding.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        *,
        scale: float,
        task_patterns: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.scale = scale
        self.understanding_delta_task_patterns = tuple(task_patterns or ())
        self.understanding_extra_delta_task_patterns: tuple[str, ...] = ()
        self.understanding_extra_mlp_lora_rank = 0
        self.understanding_extra_mlp_lora_scale = 1.0
        self.weight = base.weight
        self.bias = base.bias
        self.understanding_named_mlp_loras = nn.ModuleDict()
        self.understanding_mlp_lora_down = nn.Linear(
            self.in_features,
            rank,
            bias=False,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        self.understanding_mlp_lora_up = nn.Linear(
            rank,
            self.out_features,
            bias=False,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.normal_(
            self.understanding_mlp_lora_down.weight,
            mean=0.0,
            std=0.02,
        )
        nn.init.zeros_(self.understanding_mlp_lora_up.weight)

    def configure_extra_lora(
        self,
        rank: int,
        *,
        scale: float,
        task_patterns: Sequence[str] | None = None,
    ) -> None:
        if rank <= 0:
            self.understanding_extra_delta_task_patterns = ()
            self.understanding_extra_mlp_lora_rank = 0
            return
        existing_down = getattr(
            self,
            "understanding_extra_mlp_lora_down",
            None,
        )
        existing_up = getattr(
            self,
            "understanding_extra_mlp_lora_up",
            None,
        )
        if existing_down is not None or existing_up is not None:
            if (
                not isinstance(existing_down, nn.Linear)
                or not isinstance(existing_up, nn.Linear)
                or existing_down.out_features != rank
                or existing_up.in_features != rank
            ):
                raise ValueError(
                    "Cannot reconfigure existing extra understanding MLP "
                    "LoRA with a different rank"
                )
        else:
            self.understanding_extra_mlp_lora_down = nn.Linear(
                self.in_features,
                rank,
                bias=False,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
            self.understanding_extra_mlp_lora_up = nn.Linear(
                rank,
                self.out_features,
                bias=False,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
            self.reset_extra_lora_parameters()
        self.understanding_extra_delta_task_patterns = tuple(
            task_patterns or ()
        )
        self.understanding_extra_mlp_lora_rank = rank
        self.understanding_extra_mlp_lora_scale = scale

    def reset_extra_lora_parameters(self) -> None:
        down = getattr(self, "understanding_extra_mlp_lora_down", None)
        up = getattr(self, "understanding_extra_mlp_lora_up", None)
        if isinstance(down, nn.Linear):
            nn.init.normal_(down.weight, mean=0.0, std=0.02)
        if isinstance(up, nn.Linear):
            nn.init.zeros_(up.weight)

    def configure_named_lora(
        self,
        name: str,
        rank: int,
        *,
        scale: float,
        task_patterns: Sequence[str] | None = None,
    ) -> None:
        if rank <= 0:
            if name in self.understanding_named_mlp_loras:
                del self.understanding_named_mlp_loras[name]
            return
        existing = (
            self.understanding_named_mlp_loras[name]
            if name in self.understanding_named_mlp_loras
            else None
        )
        if existing is not None:
            if not isinstance(existing, UnderstandingLoRABank) or (
                existing.rank != rank
            ):
                raise ValueError(
                    "Cannot reconfigure existing named understanding MLP "
                    f"LoRA bank {name!r} with a different rank"
                )
            existing.scale = scale
            existing.task_patterns = tuple(task_patterns or ())
            return
        self.understanding_named_mlp_loras[name] = UnderstandingLoRABank(
            self.in_features,
            self.out_features,
            rank,
            scale=scale,
            task_patterns=task_patterns,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

    def reset_named_lora_parameters(self) -> None:
        for bank in self.understanding_named_mlp_loras.values():
            if isinstance(bank, UnderstandingLoRABank):
                bank.reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = F.linear(inputs, self.weight, self.bias)
        request = current_depth_request()
        if self.training:
            if request is not None:
                self._flexumm_checkpoint_route_kind = request.kind
                self._flexumm_checkpoint_route_task = request.task
                self._flexumm_checkpoint_sample_tasks = request.sample_tasks
                self._flexumm_checkpoint_sample_lens = request.sample_lens
                self._flexumm_checkpoint_und_token_indexes = request.und_token_indexes
            route_kind = (
                request.kind
                if request is not None
                else getattr(
                    self,
                    "_flexumm_checkpoint_route_kind",
                    None,
                )
            )
            route_task = (
                request.task
                if request is not None
                else getattr(self, "_flexumm_checkpoint_route_task", None)
            )
            route_sample_tasks = (
                request.sample_tasks
                if request is not None
                else getattr(self, "_flexumm_checkpoint_sample_tasks", None)
            )
        else:
            route_kind = request.kind if request is not None else None
            route_task = request.task if request is not None else None
            route_sample_tasks = (
                request.sample_tasks if request is not None else None
            )
        route_request = _checkpoint_route_request(self, request)
        patterns = self.understanding_delta_task_patterns
        if route_kind != "understanding":
            return output
        output = _add_routed_delta(
            output,
            inputs,
            route_request,
            patterns,
            lambda values: self.understanding_mlp_lora_up(
                self.understanding_mlp_lora_down(values)
            ),
            scale=self.scale,
        )
        extra_patterns = getattr(
            self,
            "understanding_extra_delta_task_patterns",
            (),
        )
        extra_down = getattr(self, "understanding_extra_mlp_lora_down", None)
        extra_up = getattr(self, "understanding_extra_mlp_lora_up", None)
        if (
            bool(extra_patterns)
            and isinstance(extra_down, nn.Linear)
            and isinstance(extra_up, nn.Linear)
        ):
            output = _add_routed_delta(
                output,
                inputs,
                route_request,
                extra_patterns,
                lambda values: extra_up(extra_down(values)),
                scale=getattr(
                    self,
                    "understanding_extra_mlp_lora_scale",
                    1.0,
                ),
            )
        for bank in self.understanding_named_mlp_loras.values():
            output = _add_routed_delta(
                output,
                inputs,
                route_request,
                bank.task_patterns,
                bank,
            )
        return output


class DynamicQwen2Model(_Qwen2Model):
    """Qwen2 model with FlexUMM TAFE attention/FFN routing.

    Instances are created by changing the class of an already constructed
    BAGEL ``Qwen2Model``. The current path keeps attention and normalization
    shared, adds constrained specialized FFN banks, and routes each FFN with
    the SHARE/DECOUPLE/EXIT TAFE controller. Historical depth-router and
    fusion-sealing fields remain available for compatibility.  Routed
    attention is attached after the existing attention module so pretrained
    Qwen parameter names stay unchanged.
    """

    dynamic_depth_controller: DynamicDepthController

    @staticmethod
    def _configure_understanding_attention_lora(
        layer: nn.Module,
        *,
        rank: int,
        scale: float,
        task_patterns: Sequence[str] | None = None,
    ) -> None:
        attention = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            projection = getattr(attention, name)
            if isinstance(projection, UnderstandingLoRALinear):
                if projection.rank != rank or projection.scale != scale:
                    raise ValueError(
                        "Cannot reconfigure an existing understanding "
                        f"attention LoRA on {name} with a different shape"
                    )
                projection.understanding_delta_task_patterns = tuple(
                    task_patterns or ()
                )
                continue
            if not isinstance(projection, nn.Linear):
                raise TypeError(
                    f"Expected {name} to be nn.Linear, got "
                    f"{type(projection).__name__}"
                )
            setattr(
                attention,
                name,
                UnderstandingLoRALinear(
                    projection,
                    rank,
                    scale=scale,
                    task_patterns=task_patterns,
                ),
            )

    @staticmethod
    def _configure_extra_understanding_attention_lora(
        layer: nn.Module,
        *,
        rank: int,
        scale: float,
        task_patterns: Sequence[str] | None = None,
    ) -> None:
        attention = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            projection = getattr(attention, name)
            if not isinstance(projection, UnderstandingLoRALinear):
                raise TypeError(
                    "Extra understanding attention LoRA requires the primary "
                    f"LoRA wrapper on {name}; got {type(projection).__name__}"
                )
            projection.configure_extra_lora(
                rank,
                scale=scale,
                task_patterns=task_patterns,
            )

    @staticmethod
    def _configure_named_understanding_attention_lora(
        layer: nn.Module,
        *,
        name: str,
        rank: int,
        scale: float,
        task_patterns: Sequence[str] | None = None,
    ) -> None:
        attention = layer.self_attn
        for projection_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            projection = getattr(attention, projection_name)
            if not isinstance(projection, UnderstandingLoRALinear):
                raise TypeError(
                    "Named understanding attention LoRA requires the primary "
                    f"LoRA wrapper on {projection_name}; got "
                    f"{type(projection).__name__}"
                )
            projection.configure_named_lora(
                name,
                rank,
                scale=scale,
                task_patterns=task_patterns,
            )

    @staticmethod
    def _configure_understanding_mlp_lora(
        layer: nn.Module,
        *,
        rank: int,
        scale: float,
        task_patterns: Sequence[str] | None = None,
    ) -> None:
        mlp = layer.mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(mlp, name)
            if isinstance(projection, UnderstandingMLPLoRALinear):
                if projection.rank != rank or projection.scale != scale:
                    raise ValueError(
                        "Cannot reconfigure an existing understanding MLP "
                        f"LoRA on {name} with a different shape"
                    )
                projection.understanding_delta_task_patterns = tuple(
                    task_patterns or ()
                )
                continue
            if not isinstance(projection, nn.Linear):
                raise TypeError(
                    f"Expected {name} to be nn.Linear, got "
                    f"{type(projection).__name__}"
                )
            setattr(
                mlp,
                name,
                UnderstandingMLPLoRALinear(
                    projection,
                    rank,
                    scale=scale,
                    task_patterns=task_patterns,
                ),
            )

    @staticmethod
    def _configure_extra_understanding_mlp_lora(
        layer: nn.Module,
        *,
        rank: int,
        scale: float,
        task_patterns: Sequence[str] | None = None,
    ) -> None:
        mlp = layer.mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(mlp, name)
            if not isinstance(projection, UnderstandingMLPLoRALinear):
                raise TypeError(
                    "Extra understanding MLP LoRA requires the primary LoRA "
                    f"wrapper on {name}; got {type(projection).__name__}"
                )
            projection.configure_extra_lora(
                rank,
                scale=scale,
                task_patterns=task_patterns,
            )

    @staticmethod
    def _configure_named_understanding_mlp_lora(
        layer: nn.Module,
        *,
        name: str,
        rank: int,
        scale: float,
        task_patterns: Sequence[str] | None = None,
    ) -> None:
        mlp = layer.mlp
        for projection_name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(mlp, projection_name)
            if not isinstance(projection, UnderstandingMLPLoRALinear):
                raise TypeError(
                    "Named understanding MLP LoRA requires the primary LoRA "
                    f"wrapper on {projection_name}; got "
                    f"{type(projection).__name__}"
                )
            projection.configure_named_lora(
                name,
                rank,
                scale=scale,
                task_patterns=task_patterns,
            )

    def configure_dynamic_depth(self, config: DynamicDepthConfig) -> None:
        self.dynamic_depth_controller = DynamicDepthController(
            config=config, num_layers=len(self.layers)
        )
        self._flexumm_aux: dict[str, Any] | None = None
        if config.fusion_sealing_enabled:
            task_names = list(config.router_tasks)
            if config.router_tasks_generation is not None:
                task_names.extend(config.router_tasks_generation)
            task_names.extend(config.fusion_sealing_task_names)
            active_fusion_kinds = []
            if config.fusion_sealing_candidates_understanding:
                active_fusion_kinds.append("understanding")
            if config.fusion_sealing_candidates_generation:
                active_fusion_kinds.append("generation")
            self.fusion_sealing_compilers = nn.ModuleDict(
                {
                    kind: FusionCompiler(
                        self.config.hidden_size,
                        config.fusion_sealing_num_tokens,
                        config.fusion_sealing_bottleneck_size,
                        task_names=task_names,
                        condition_on_task=(
                            config.fusion_sealing_condition_compiler_on_task
                        ),
                        condition_on_fine_task=(
                            config.fusion_sealing_condition_on_fine_task
                        ),
                    )
                    for kind in active_fusion_kinds
                }
            )
            self.fusion_sealing_realizers = nn.ModuleDict(
                {
                    kind: FusionRealizationAdapter(
                        self.config.hidden_size,
                        config.fusion_sealing_realizer_rank,
                        scale=config.fusion_sealing_realizer_scale,
                    )
                    for kind in active_fusion_kinds
                }
            )
            self.fusion_release_controller = FusionReleaseController(
                hidden_size=self.config.hidden_size,
                controller_hidden_size=config.fusion_sealing_router_hidden_size,
                num_layers=len(self.layers),
                task_names=task_names,
                initial_seal_bias=config.fusion_sealing_initial_seal_bias,
                condition_on_task=config.fusion_sealing_condition_on_task,
                condition_on_fine_task=(
                    config.fusion_sealing_condition_on_fine_task
                ),
                condition_on_realization_delta=(
                    config.fusion_sealing_condition_on_realization_delta
                ),
            )
        else:
            for module_name in (
                "fusion_sealing_compilers",
                "fusion_sealing_realizers",
                "fusion_release_controller",
            ):
                if hasattr(self, module_name):
                    delattr(self, module_name)
        if config.router_enabled and config.router_candidates_understanding:
            understanding_feature_mode = (
                "full"
                if config.router_conditioning == "prediction_entropy"
                else config.router_conditioning
            )
            router = getattr(self, "depth_router", None)
            expected_tasks = tuple(
                dict.fromkeys(
                    CandidateDepthRouter.normalize_task(task)
                    for task in ("unknown", *config.router_tasks)
                )
            )
            if (
                not isinstance(router, CandidateDepthRouter)
                or router.state_proj.in_features != self.config.hidden_size
                or router.state_proj.out_features != config.router_hidden_size
                or router.task_names != expected_tasks
                or router.condition_on_task != config.router_condition_on_task
                or router.use_state_drift != config.router_use_state_drift
                or router.use_last_token != config.router_use_last_token
                or router.feature_mode != understanding_feature_mode
                or router.compute_in_float32
                != config.understanding_router_float32
            ):
                self.depth_router = CandidateDepthRouter(
                    backbone_hidden_size=self.config.hidden_size,
                    router_hidden_size=config.router_hidden_size,
                    num_layers=len(self.layers),
                    task_names=config.router_tasks,
                    initial_bias=config.router_initial_bias,
                    condition_on_task=config.router_condition_on_task,
                    use_state_drift=config.router_use_state_drift,
                    use_last_token=config.router_use_last_token,
                    feature_mode=understanding_feature_mode,
                    compute_in_float32=config.understanding_router_float32,
                )
        elif hasattr(self, "depth_router"):
            del self.depth_router
        if config.router_enabled and config.router_candidates_generation:
            router = getattr(self, "generation_depth_router", None)
            generation_router_tasks = (
                config.router_tasks_generation
                if config.router_tasks_generation is not None
                else config.router_tasks
            )
            generation_use_state_drift = (
                config.router_use_state_drift_generation
                if config.router_use_state_drift_generation is not None
                else config.router_use_state_drift
            )
            generation_use_last_token = (
                config.router_use_last_token_generation
                if config.router_use_last_token_generation is not None
                else config.router_use_last_token
            )
            generation_conditioning = (
                config.router_conditioning_generation
                if config.router_conditioning_generation is not None
                else config.router_conditioning
            )
            generation_feature_mode = (
                "full"
                if generation_conditioning == "prediction_entropy"
                else generation_conditioning
            )
            expected_tasks = tuple(
                dict.fromkeys(
                    CandidateDepthRouter.normalize_task(task)
                    for task in ("unknown", *generation_router_tasks)
                )
            )
            if (
                not isinstance(router, CandidateDepthRouter)
                or router.state_proj.in_features != self.config.hidden_size
                or router.state_proj.out_features != config.router_hidden_size
                or router.task_names != expected_tasks
                or router.condition_on_task != config.router_condition_on_task
                or router.use_state_drift != generation_use_state_drift
                or router.use_last_token != generation_use_last_token
                or router.feature_mode != generation_feature_mode
                or router.compute_in_float32
                != config.generation_router_float32
            ):
                self.generation_depth_router = CandidateDepthRouter(
                    backbone_hidden_size=self.config.hidden_size,
                    router_hidden_size=config.router_hidden_size,
                    num_layers=len(self.layers),
                    task_names=generation_router_tasks,
                    initial_bias=config.router_initial_bias,
                    condition_on_task=config.router_condition_on_task,
                    use_state_drift=generation_use_state_drift,
                    use_last_token=generation_use_last_token,
                    feature_mode=generation_feature_mode,
                    compute_in_float32=config.generation_router_float32,
                )
        elif hasattr(self, "generation_depth_router"):
            del self.generation_depth_router
        attention_lora_layers = set(
            config.understanding_attention_lora_layers
        )
        for layer_number, layer in enumerate(self.layers, start=1):
            if layer_number in attention_lora_layers:
                self._configure_understanding_attention_lora(
                    layer,
                    rank=config.understanding_attention_lora_rank,
                    scale=config.understanding_attention_lora_scale,
                    task_patterns=config.understanding_delta_task_patterns,
                )
        extra_attention_lora_layers = set(
            config.understanding_extra_attention_lora_layers
        )
        for layer_number, layer in enumerate(self.layers, start=1):
            if layer_number in extra_attention_lora_layers:
                self._configure_extra_understanding_attention_lora(
                    layer,
                    rank=config.understanding_extra_attention_lora_rank,
                    scale=config.understanding_extra_attention_lora_scale,
                    task_patterns=(
                        config.understanding_extra_delta_task_patterns
                    ),
                )
        mlp_lora_layers = set(config.understanding_mlp_lora_layers)
        for layer_number, layer in enumerate(self.layers, start=1):
            if layer_number in mlp_lora_layers:
                self._configure_understanding_mlp_lora(
                    layer,
                    rank=config.understanding_mlp_lora_rank,
                    scale=config.understanding_mlp_lora_scale,
                    task_patterns=config.understanding_delta_task_patterns,
                )
        extra_mlp_lora_layers = set(
            config.understanding_extra_mlp_lora_layers
        )
        for layer_number, layer in enumerate(self.layers, start=1):
            if layer_number in extra_mlp_lora_layers:
                self._configure_extra_understanding_mlp_lora(
                    layer,
                    rank=config.understanding_extra_mlp_lora_rank,
                    scale=config.understanding_extra_mlp_lora_scale,
                    task_patterns=(
                        config.understanding_extra_delta_task_patterns
                    ),
                )
        named_delta_banks = list(config.understanding_named_delta_banks or ())
        for bank in named_delta_banks:
            bank_name = str(bank["name"])
            task_patterns = tuple(bank.get("task_patterns", ()) or ())
            attention_layers = set(
                int(layer)
                for layer in (bank.get("attention_lora_layers", ()) or ())
            )
            attention_rank = int(bank.get("attention_lora_rank", 0) or 0)
            if attention_layers and attention_rank:
                for layer_number, layer in enumerate(self.layers, start=1):
                    if layer_number in attention_layers:
                        self._configure_named_understanding_attention_lora(
                            layer,
                            name=bank_name,
                            rank=attention_rank,
                            scale=float(bank.get("attention_lora_scale", 1.0)),
                            task_patterns=task_patterns,
                        )
            mlp_layers = set(
                int(layer) for layer in (bank.get("mlp_lora_layers", ()) or ())
            )
            mlp_rank = int(bank.get("mlp_lora_rank", 0) or 0)
            if mlp_layers and mlp_rank:
                for layer_number, layer in enumerate(self.layers, start=1):
                    if layer_number in mlp_layers:
                        self._configure_named_understanding_mlp_lora(
                            layer,
                            name=bank_name,
                            rank=mlp_rank,
                            scale=float(bank.get("mlp_lora_scale", 1.0)),
                            task_patterns=task_patterns,
                        )
        expected_adapter_layers = tuple(config.understanding_adapter_layers)
        adapters = getattr(self, "understanding_adapters", None)
        adapter_matches = (
            isinstance(adapters, nn.ModuleDict)
            and tuple(int(name) for name in adapters)
            == expected_adapter_layers
            and all(
                isinstance(adapter, UnderstandingResidualAdapter)
                and adapter.hidden_size == self.config.hidden_size
                and adapter.rank == config.understanding_adapter_rank
                and adapter.scale == config.understanding_adapter_scale
                for adapter in adapters.values()
            )
        )
        if expected_adapter_layers and not adapter_matches:
            self.understanding_adapters = nn.ModuleDict(
                {
                    str(layer_number): UnderstandingResidualAdapter(
                        self.config.hidden_size,
                        config.understanding_adapter_rank,
                        scale=config.understanding_adapter_scale,
                    )
                    for layer_number in expected_adapter_layers
                }
            )
        elif not expected_adapter_layers and hasattr(
            self,
            "understanding_adapters",
        ):
            del self.understanding_adapters
        expected_extra_adapter_layers = tuple(
            config.understanding_extra_adapter_layers
        )
        extra_adapters = getattr(self, "understanding_extra_adapters", None)
        extra_adapter_matches = (
            isinstance(extra_adapters, nn.ModuleDict)
            and tuple(int(name) for name in extra_adapters)
            == expected_extra_adapter_layers
            and all(
                isinstance(adapter, UnderstandingResidualAdapter)
                and adapter.hidden_size == self.config.hidden_size
                and adapter.rank == config.understanding_extra_adapter_rank
                and adapter.scale == config.understanding_extra_adapter_scale
                for adapter in extra_adapters.values()
            )
        )
        if expected_extra_adapter_layers and not extra_adapter_matches:
            self.understanding_extra_adapters = nn.ModuleDict(
                {
                    str(layer_number): UnderstandingResidualAdapter(
                        self.config.hidden_size,
                        config.understanding_extra_adapter_rank,
                        scale=config.understanding_extra_adapter_scale,
                    )
                    for layer_number in expected_extra_adapter_layers
                }
            )
        elif not expected_extra_adapter_layers and hasattr(
            self,
            "understanding_extra_adapters",
        ):
            del self.understanding_extra_adapters
        named_adapter_banks = [
            bank
            for bank in named_delta_banks
            if bank.get("adapter_layers") and int(bank.get("adapter_rank", 0) or 0)
        ]
        if named_adapter_banks:
            named_adapters = getattr(self, "understanding_named_adapters", None)
            if not isinstance(named_adapters, nn.ModuleDict):
                self.understanding_named_adapters = nn.ModuleDict()
                named_adapters = self.understanding_named_adapters
            expected_names = {str(bank["name"]) for bank in named_adapter_banks}
            for bank in named_adapter_banks:
                bank_name = str(bank["name"])
                adapter_layers = tuple(
                    int(layer) for layer in bank.get("adapter_layers", ())
                )
                adapter_rank = int(bank.get("adapter_rank", 0) or 0)
                adapter_scale = float(bank.get("adapter_scale", 1.0))
                bank_adapters = (
                    named_adapters[bank_name]
                    if bank_name in named_adapters
                    else None
                )
                bank_matches = (
                    isinstance(bank_adapters, nn.ModuleDict)
                    and tuple(int(name) for name in bank_adapters)
                    == adapter_layers
                    and all(
                        isinstance(adapter, UnderstandingResidualAdapter)
                        and adapter.hidden_size == self.config.hidden_size
                        and adapter.rank == adapter_rank
                        and adapter.scale == adapter_scale
                        for adapter in bank_adapters.values()
                    )
                )
                if not bank_matches:
                    named_adapters[bank_name] = nn.ModuleDict(
                        {
                            str(layer_number): UnderstandingResidualAdapter(
                                self.config.hidden_size,
                                adapter_rank,
                                scale=adapter_scale,
                            )
                            for layer_number in adapter_layers
                        }
                    )
            for bank_name in list(named_adapters.keys()):
                if bank_name not in expected_names:
                    del named_adapters[bank_name]
        elif hasattr(self, "understanding_named_adapters"):
            del self.understanding_named_adapters
        fusion = getattr(self, "understanding_depth_fusion", None)
        fusion_matches = (
            isinstance(fusion, UnderstandingDepthFusion)
            and fusion.hidden_size == self.config.hidden_size
            and fusion.rank == config.understanding_fusion_rank
            and fusion.scale == config.understanding_fusion_scale
        )
        if config.understanding_fusion_rank and not fusion_matches:
            self.understanding_depth_fusion = UnderstandingDepthFusion(
                self.config.hidden_size,
                config.understanding_fusion_rank,
                scale=config.understanding_fusion_scale,
            )
        elif not config.understanding_fusion_rank and hasattr(
            self,
            "understanding_depth_fusion",
        ):
            del self.understanding_depth_fusion
        extra_fusion = getattr(self, "understanding_extra_depth_fusion", None)
        extra_fusion_matches = (
            isinstance(extra_fusion, UnderstandingDepthFusion)
            and extra_fusion.hidden_size == self.config.hidden_size
            and extra_fusion.rank == config.understanding_extra_fusion_rank
            and extra_fusion.scale == config.understanding_extra_fusion_scale
        )
        if config.understanding_extra_fusion_rank and not extra_fusion_matches:
            self.understanding_extra_depth_fusion = UnderstandingDepthFusion(
                self.config.hidden_size,
                config.understanding_extra_fusion_rank,
                scale=config.understanding_extra_fusion_scale,
            )
        elif not config.understanding_extra_fusion_rank and hasattr(
            self,
            "understanding_extra_depth_fusion",
        ):
            del self.understanding_extra_depth_fusion
        named_fusion_banks = [
            bank
            for bank in named_delta_banks
            if int(bank.get("fusion_rank", 0) or 0)
        ]
        if named_fusion_banks:
            named_fusions = getattr(
                self,
                "understanding_named_depth_fusions",
                None,
            )
            if not isinstance(named_fusions, nn.ModuleDict):
                self.understanding_named_depth_fusions = nn.ModuleDict()
                named_fusions = self.understanding_named_depth_fusions
            expected_names = {str(bank["name"]) for bank in named_fusion_banks}
            for bank in named_fusion_banks:
                bank_name = str(bank["name"])
                fusion_rank = int(bank.get("fusion_rank", 0) or 0)
                fusion_scale = float(bank.get("fusion_scale", 1.0))
                fusion = (
                    named_fusions[bank_name]
                    if bank_name in named_fusions
                    else None
                )
                fusion_matches = (
                    isinstance(fusion, UnderstandingDepthFusion)
                    and fusion.hidden_size == self.config.hidden_size
                    and fusion.rank == fusion_rank
                    and fusion.scale == fusion_scale
                )
                if not fusion_matches:
                    named_fusions[bank_name] = UnderstandingDepthFusion(
                        self.config.hidden_size,
                        fusion_rank,
                        scale=fusion_scale,
                    )
            for bank_name in list(named_fusions.keys()):
                if bank_name not in expected_names:
                    del named_fusions[bank_name]
        elif hasattr(self, "understanding_named_depth_fusions"):
            del self.understanding_named_depth_fusions

        self._configure_tafe(config)

    def _configure_tafe(self, config: DynamicDepthConfig) -> None:
        """Install independent FlexUMM routes for both BAGEL objectives.

        The original Qwen FFN projections remain registered directly under
        ``layer.mlp`` (and ``layer.mlp_moe_gen``), so a pretrained checkpoint
        still matches its base parameter names.  Understanding and generation
        each receive their own controller and residual subset pool.  Optional
        attention residual banks are registered alongside each decoder layer;
        the underlying attention projections remain shared.
        """

        self._remove_tafe_attention_routes()

        for gate_name in (
            "tafe_gate_understanding",
            "tafe_gate_generation",
        ):
            if not config.tafe_enabled and hasattr(self, gate_name):
                delattr(self, gate_name)

        def make_gate(
            kind: str,
            *,
            allow_exit: bool | None = None,
            action_policy: str | None = None,
        ) -> TAFEGate:
            task_names = (
                config.tafe_task_names_generation
                if kind == "generation"
                else config.tafe_task_names_understanding
            )
            return TAFEGate(
                backbone_hidden_size=self.config.hidden_size,
                controller_hidden_size=config.tafe_controller_hidden_size,
                num_layers=len(self.layers),
                task_names=task_names,
                num_specialized_subsets=config.tafe_num_specialized_subsets,
                routing_width=config.tafe_routing_width,
                condition_on_task=config.tafe_condition_on_task,
                use_hidden_state=config.tafe_use_hidden_state,
                action_policy=(
                    config.tafe_action_policy
                    if action_policy is None
                    else action_policy
                ),
                static_boundary_layer=config.tafe_static_boundary_layer,
                allow_exit=(
                    config.tafe_allow_exit
                    if allow_exit is None
                    else allow_exit
                ),
                initial_action_bias=config.tafe_initial_action_bias,
                action_costs=config.tafe_action_costs,
                task_action_costs=config.tafe_task_action_costs,
                lambda_cost=config.tafe_lambda_cost,
                soft_routing_train=config.tafe_soft_routing_train,
                soft_routing_eval=config.tafe_soft_routing_eval,
                soft_routing_temperature=config.tafe_soft_routing_temperature,
            )

        if config.tafe_enabled:
            self.tafe_gate_understanding = make_gate("understanding")
            self.tafe_gate_generation = make_gate("generation")

            for layer_number, layer in enumerate(self.layers, start=1):
                for module_name, kind in (
                    ("mlp", "understanding"),
                    ("mlp_moe_gen", "generation"),
                ):
                    gate = (
                        self.tafe_gate_generation
                        if kind == "generation"
                        else self.tafe_gate_understanding
                    )
                    base_ffn = getattr(layer, module_name, None)
                    if base_ffn is None:
                        continue
                    if isinstance(base_ffn, RoutedFFN):
                        # Reconfiguration can happen in tests or interactive
                        # notebooks. Reuse the wrapper rather than nesting it.
                        base_ffn._gate_ref = weakref.ref(gate)
                        continue
                    setattr(
                        layer,
                        module_name,
                        RoutedFFN(
                            base_ffn,
                            gate,
                            layer=layer_number,
                            kind=kind,
                            adapter_rank=config.tafe_adapter_rank,
                            adapter_scale=config.tafe_adapter_scale,
                            num_specialized_subsets=(
                                config.tafe_num_specialized_subsets
                            ),
                            routing_width=config.tafe_routing_width,
                            parameter_budget=config.tafe_parameter_budget,
                        ),
                    )

        if config.tafe_attention_enabled:
            # Attention specialization uses only SHARE/DECOUPLE.  Keep the
            # three-column utility head/checkpoint shape for compatibility,
            # but prevent attention routes from terminating the stack.
            attention_action_policy = (
                "share"
                if config.tafe_action_policy == "exit"
                else config.tafe_action_policy
            )
            self.tafe_attention_gate_understanding = make_gate(
                "understanding",
                allow_exit=False,
                action_policy=attention_action_policy,
            )
            self.tafe_attention_gate_generation = make_gate(
                "generation",
                allow_exit=False,
                action_policy=attention_action_policy,
            )
            for layer_number, layer in enumerate(self.layers, start=1):
                self._install_tafe_attention_route(
                    layer,
                    layer_number=layer_number,
                    kind="understanding",
                    gate=self.tafe_attention_gate_understanding,
                    config=config,
                )
                self._install_tafe_attention_route(
                    layer,
                    layer_number=layer_number,
                    kind="generation",
                    gate=self.tafe_attention_gate_generation,
                    config=config,
                )

    def _remove_tafe_attention_routes(self) -> None:
        """Remove attention hooks/modules before dynamic reconfiguration."""

        for layer in getattr(self, "layers", ()):
            handle = getattr(layer, "_flexumm_tafe_attention_hook", None)
            if handle is not None:
                handle.remove()
                delattr(layer, "_flexumm_tafe_attention_hook")
            for name in (
                "tafe_attention_understanding",
                "tafe_attention_generation",
            ):
                if hasattr(layer, name):
                    delattr(layer, name)
        for gate_name in (
            "tafe_attention_gate_understanding",
            "tafe_attention_gate_generation",
        ):
            if hasattr(self, gate_name):
                delattr(self, gate_name)

    def _install_tafe_attention_route(
        self,
        layer: nn.Module,
        *,
        layer_number: int,
        kind: str,
        gate: TAFEGate,
        config: DynamicDepthConfig,
    ) -> None:
        setattr(
            layer,
            f"tafe_attention_{kind}",
            RoutedAttention(
                gate,
                layer=layer_number,
                kind=kind,
                hidden_size=self.config.hidden_size,
                adapter_rank=config.tafe_attention_adapter_rank,
                adapter_scale=config.tafe_attention_adapter_scale,
                num_specialized_subsets=config.tafe_num_specialized_subsets,
                parameter_budget=config.tafe_attention_parameter_budget,
            ),
        )
        model_ref = weakref.ref(self)
        layer_ref = weakref.ref(layer)

        def attention_hook(module, args, kwargs, output):
            model = model_ref()
            target_layer = layer_ref()
            if model is None or target_layer is None:
                return output
            return model._route_tafe_attention_output(
                target_layer,
                args,
                kwargs,
                output,
            )

        layer._flexumm_tafe_attention_hook = layer.self_attn.register_forward_hook(
            attention_hook,
            with_kwargs=True,
        )

    def _route_tafe_attention_output(
        self,
        layer: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        """Apply the two attention routes to the appropriate packed rows."""

        if isinstance(output, tuple):
            if not output or not isinstance(output[0], torch.Tensor):
                return output
            attention_output = output[0]
        elif isinstance(output, torch.Tensor):
            attention_output = output
        else:
            return output
        hidden_states = (
            kwargs.get("packed_sequence")
            if kwargs.get("packed_sequence") is not None
            else kwargs.get("packed_query_sequence")
        )
        if hidden_states is None and args and isinstance(args[0], torch.Tensor):
            hidden_states = args[0]
        if not isinstance(hidden_states, torch.Tensor):
            return output

        route_specs: list[tuple[str, torch.Tensor | None]] = []
        if (
            "packed_und_token_indexes" in kwargs
            or "packed_gen_token_indexes" in kwargs
        ):
            route_specs.extend(
                [
                    (
                        "understanding",
                        kwargs.get("packed_und_token_indexes"),
                    ),
                    (
                        "generation",
                        kwargs.get("packed_gen_token_indexes"),
                    ),
                ]
            )
        elif str(kwargs.get("mode", "und")).lower() == "gen":
            route_specs.extend(
                [
                    (
                        "understanding",
                        kwargs.get("packed_text_indexes"),
                    ),
                    (
                        "generation",
                        kwargs.get("packed_vae_token_indexes"),
                    ),
                ]
            )
        else:
            request = current_depth_request()
            kind = (
                request.kind
                if request is not None and request.kind in {
                    "understanding",
                    "generation",
                }
                else "understanding"
            )
            route_specs.append((kind, None))

        routed_output = attention_output
        for kind, token_indexes in route_specs:
            if token_indexes is None:
                if route_specs and len(route_specs) > 1:
                    continue
            router = getattr(layer, f"tafe_attention_{kind}", None)
            if router is None:
                continue
            routed_output = router.apply(
                routed_output,
                hidden_states,
                token_indexes=token_indexes,
            )
        if isinstance(output, tuple):
            return (routed_output, *output[1:])
        return routed_output

    def _reset_tafe_route_state(self) -> None:
        """Clear per-layer route state before running decoder branches."""

        for namespace in ("tafe_gate", "tafe_attention_gate"):
            for kind in ("understanding", "generation"):
                gate = getattr(self, f"{namespace}_{kind}", None)
                if gate is not None:
                    gate.last_result = None
                    gate.last_all_exit = False

    def _tafe_all_active_routes_exited(
        self,
        *,
        has_understanding: bool,
        has_generation: bool,
    ) -> bool:
        """Return whether every active modality exited its FFN at this layer."""

        active_gates = []
        if has_understanding:
            active_gates.append(
                getattr(self, "tafe_gate_understanding", None)
            )
        if has_generation:
            active_gates.append(getattr(self, "tafe_gate_generation", None))
        active_gates = [gate for gate in active_gates if gate is not None]
        return bool(active_gates) and all(
            bool(gate.last_all_exit) for gate in active_gates
        )

    def _fusion_sealing_candidates(self, kind: str) -> list[int]:
        config = self.dynamic_depth_controller.config
        if not config.fusion_sealing_enabled:
            return []
        if kind == "generation":
            return list(config.fusion_sealing_candidates_generation)
        return list(config.fusion_sealing_candidates_understanding)

    def _fusion_sealing_active(self, kind: str) -> bool:
        config = self.dynamic_depth_controller.config
        return bool(
            kind in {"understanding", "generation"}
            and config.fusion_sealing_enabled
            and config.fusion_sealing_router_enabled
            and self._fusion_sealing_candidates(kind)
            and hasattr(self, "fusion_release_controller")
        )

    def _compile_fusion_capsule(
        self,
        hidden_states: torch.Tensor,
        *,
        kind: str,
        task: str | Sequence[str] | None = None,
        token_indexes: torch.Tensor | None = None,
        sample_lens: Sequence[int] | None = None,
        active_sample_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        compilers = getattr(self, "fusion_sealing_compilers", None)
        if not isinstance(compilers, nn.ModuleDict) or kind not in compilers:
            raise RuntimeError(
                "Fusion sealing is active but its compiler is not configured"
            )
        return compilers[kind](
            hidden_states,
            token_indexes=token_indexes,
            sample_lens=sample_lens,
            active_sample_ids=active_sample_ids,
            task=task,
        )

    @staticmethod
    def _fusion_sealing_threshold_tensor(
        overrides: dict[str, float],
        default: float,
        *,
        request: Any,
        active_sample_ids: torch.Tensor,
        sample_lens: Sequence[int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Resolve one scalar gate threshold for each active packed sample."""
        if not overrides:
            return torch.full(
                (batch_size,),
                float(default),
                device=device,
                dtype=dtype,
            )
        tasks = getattr(request, "sample_tasks", None) if request is not None else None
        if tasks is None:
            task = getattr(request, "task", "unknown") if request is not None else "unknown"
            tasks = [task] * len(sample_lens)
        elif isinstance(tasks, str):
            tasks = [tasks] * len(sample_lens)
        else:
            tasks = list(tasks)
        if len(tasks) == len(sample_lens):
            active_ids = active_sample_ids.detach().cpu().tolist()
            tasks = [tasks[int(sample_id)] for sample_id in active_ids]
        if len(tasks) != batch_size:
            tasks = [str(getattr(request, "task", "unknown"))] * batch_size
        values = []
        for task in tasks:
            name = str(task or "unknown").strip().lower().replace("-", "_")
            value = float(default)
            for pattern, threshold in overrides.items():
                if fnmatch.fnmatchcase(name, str(pattern).lower().replace("-", "_")):
                    value = float(threshold)
                    break
            values.append(value)
        return torch.tensor(values, device=device, dtype=dtype)

    def _apply_fusion_realization(
        self,
        hidden_states: torch.Tensor,
        capsule: torch.Tensor,
        *,
        kind: str,
        token_indexes: torch.Tensor | None = None,
        sample_lens: Sequence[int] | None = None,
        active_sample_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        realizers = getattr(self, "fusion_sealing_realizers", None)
        if not isinstance(realizers, nn.ModuleDict) or kind not in realizers:
            raise RuntimeError(
                "Fusion sealing is active but its realization module is not configured"
            )
        return realizers[kind](
            hidden_states,
            capsule,
            token_indexes=token_indexes,
            sample_lens=sample_lens,
            active_sample_ids=active_sample_ids,
        )

    @staticmethod
    def _fusion_active_sample_ids(
        hidden_states: torch.Tensor,
        *,
        token_indexes: torch.Tensor | None,
        sample_lens: Sequence[int],
    ) -> torch.Tensor:
        positions = (
            torch.arange(
                hidden_states.shape[0],
                device=hidden_states.device,
                dtype=torch.long,
            )
            if token_indexes is None
            else _positions_from_indexes(
                token_indexes,
                sequence_length=hidden_states.shape[0],
            )
        )
        if positions.numel() == 0:
            raise ValueError("Fusion sealing has no active task tokens")
        sample_ids = _sample_ids_for_positions(positions, list(sample_lens))
        active_sample_ids = torch.unique(sample_ids, sorted=True)
        if active_sample_ids.numel() == 0:
            raise ValueError("Fusion sealing has no active packed samples")
        return active_sample_ids

    @staticmethod
    def _fusion_token_indexes_for_samples(
        hidden_states: torch.Tensor,
        *,
        token_indexes: torch.Tensor | None,
        sample_lens: Sequence[int],
        selected_sample_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Restrict realization to the packed tokens of selected samples."""

        positions = (
            torch.arange(
                hidden_states.shape[0],
                device=hidden_states.device,
                dtype=torch.long,
            )
            if token_indexes is None
            else _positions_from_indexes(
                token_indexes,
                sequence_length=hidden_states.shape[0],
            )
        )
        sample_ids = _sample_ids_for_positions(positions, list(sample_lens))
        selected_sample_ids = selected_sample_ids.to(
            device=sample_ids.device, dtype=torch.long
        )
        return positions[torch.isin(sample_ids, selected_sample_ids)]

    def _apply_understanding_adapter(
        self,
        hidden_states: torch.Tensor,
        layer_number: int,
        token_indexes: torch.Tensor | None = None,
        sample_lens: Sequence[int] | None = None,
    ) -> torch.Tensor:
        depth_config = self.dynamic_depth_controller.config
        request = current_depth_request()
        output = hidden_states
        layer_key = str(layer_number)
        primary_positions = _route_token_positions_for_patterns(
            request,
            depth_config.understanding_delta_task_patterns,
            token_indexes=token_indexes,
            sequence_length=hidden_states.shape[0],
        )
        if primary_positions.numel() != 0:
            adapters = getattr(self, "understanding_adapters", None)
            if isinstance(adapters, nn.ModuleDict) and layer_key in adapters:
                adapter = adapters[layer_key]
                updated = output.clone()
                updated[primary_positions] = output[primary_positions] + adapter(
                    output[primary_positions]
                )
                output = updated
        extra_positions = _route_token_positions_for_patterns(
            request,
            depth_config.understanding_extra_delta_task_patterns,
            token_indexes=token_indexes,
            sequence_length=hidden_states.shape[0],
        )
        if extra_positions.numel() != 0:
            extra_adapters = getattr(
                self,
                "understanding_extra_adapters",
                None,
            )
            if (
                isinstance(extra_adapters, nn.ModuleDict)
                and layer_key in extra_adapters
            ):
                extra_adapter = extra_adapters[layer_key]
                updated = output.clone()
                updated[extra_positions] = output[extra_positions] + extra_adapter(
                    output[extra_positions]
                )
                output = updated
        named_adapters = getattr(self, "understanding_named_adapters", None)
        if isinstance(named_adapters, nn.ModuleDict):
            for bank in depth_config.understanding_named_delta_banks:
                bank_name = str(bank["name"])
                bank_adapters = (
                    named_adapters[bank_name]
                    if bank_name in named_adapters
                    else None
                )
                if not (
                    isinstance(bank_adapters, nn.ModuleDict)
                    and layer_key in bank_adapters
                ):
                    continue
                adapter = bank_adapters[layer_key]
                bank_positions = _route_token_positions_for_patterns(
                    request,
                    bank.get("task_patterns", ()),
                    token_indexes=token_indexes,
                    sequence_length=hidden_states.shape[0],
                )
                if bank_positions.numel() == 0:
                    continue
                updated = output.clone()
                updated[bank_positions] = output[bank_positions] + adapter(
                    output[bank_positions]
                )
                output = updated
        return output

    def _apply_understanding_fusion(
        self,
        source_hidden_states: torch.Tensor | None,
        target_hidden_states: torch.Tensor,
        token_indexes: torch.Tensor | None = None,
        sample_lens: Sequence[int] | None = None,
    ) -> torch.Tensor:
        depth_config = self.dynamic_depth_controller.config
        request = current_depth_request()
        output = target_hidden_states
        primary_positions = _route_token_positions_for_patterns(
            request,
            depth_config.understanding_delta_task_patterns,
            token_indexes=token_indexes,
            sequence_length=target_hidden_states.shape[0],
        )
        if primary_positions.numel() != 0:
            fusion = getattr(self, "understanding_depth_fusion", None)
            if source_hidden_states is not None and isinstance(
                fusion,
                UnderstandingDepthFusion,
            ):
                if source_hidden_states.shape != output.shape:
                    raise ValueError(
                        "Understanding fusion source and target states must align"
                    )
                updated = output.clone()
                updated[primary_positions] = output[primary_positions] + fusion(
                    source_hidden_states[primary_positions],
                    output[primary_positions],
                )
                output = updated
        extra_positions = _route_token_positions_for_patterns(
            request,
            depth_config.understanding_extra_delta_task_patterns,
            token_indexes=token_indexes,
            sequence_length=target_hidden_states.shape[0],
        )
        if extra_positions.numel() != 0:
            extra_fusion = getattr(
                self,
                "understanding_extra_depth_fusion",
                None,
            )
            if source_hidden_states is not None and isinstance(
                extra_fusion,
                UnderstandingDepthFusion,
            ):
                if source_hidden_states.shape != output.shape:
                    raise ValueError(
                        "Extra understanding fusion source and target states "
                        "must align"
                    )
                updated = output.clone()
                updated[extra_positions] = output[extra_positions] + extra_fusion(
                    source_hidden_states[extra_positions],
                    output[extra_positions],
                )
                output = updated
        named_fusions = getattr(self, "understanding_named_depth_fusions", None)
        if source_hidden_states is not None and isinstance(
            named_fusions,
            nn.ModuleDict,
        ):
            if source_hidden_states.shape != output.shape:
                raise ValueError(
                    "Named understanding fusion source and target states must align"
                )
            for bank in depth_config.understanding_named_delta_banks:
                bank_name = str(bank["name"])
                fusion = (
                    named_fusions[bank_name]
                    if bank_name in named_fusions
                    else None
                )
                if not isinstance(fusion, UnderstandingDepthFusion):
                    continue
                bank_positions = _route_token_positions_for_patterns(
                    request,
                    bank.get("task_patterns", ()),
                    token_indexes=token_indexes,
                    sequence_length=target_hidden_states.shape[0],
                )
                if bank_positions.numel() == 0:
                    continue
                updated = output.clone()
                updated[bank_positions] = output[bank_positions] + fusion(
                    source_hidden_states[bank_positions],
                    output[bank_positions],
                )
                output = updated
        return output

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: list[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: torch.LongTensor | None = None,
        packed_gen_token_indexes: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        request = current_depth_request()
        if request is not None:
            packed_und_token_indexes = (
                packed_und_token_indexes
                if packed_und_token_indexes is not None
                else request.und_token_indexes
            )
            packed_gen_token_indexes = (
                packed_gen_token_indexes
                if packed_gen_token_indexes is not None
                else request.gen_token_indexes
            )

        if self.use_moe:
            if packed_und_token_indexes is None:
                raise ValueError("MoT training requires understanding token indexes")
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_und_token_indexes.new_empty((0,))

        has_understanding = (
            request.has_understanding
            if request is not None and request.has_understanding is not None
            else _nonempty(packed_und_token_indexes)
        )
        has_generation = (
            request.has_generation
            if request is not None and request.has_generation is not None
            else _nonempty(packed_gen_token_indexes)
        )
        kind = (
            request.kind
            if request is not None and request.kind is not None
            else (
                "mixed"
                if has_understanding and has_generation
                else "generation"
                if has_generation
                else "understanding"
            )
        )
        plan = self.dynamic_depth_controller.resolve(
            kind=kind,
            task=request.task if request is not None else None,
            timestep=request.timestep if request is not None else None,
            depth_override=request.depth_override if request is not None else None,
            has_understanding=has_understanding,
            has_generation=has_generation,
        )
        depth_config = self.dynamic_depth_controller.config
        fusion_sealing_kind = (
            kind if self._fusion_sealing_active(kind) else None
        )
        fusion_sealing_candidates = (
            sorted(
                set(
                    [
                        *self._fusion_sealing_candidates(kind),
                        plan.execution_depth,
                    ]
                )
            )
            if fusion_sealing_kind is not None
            else []
        )
        collect_router_exits = (
            depth_config.router_enabled
            and depth_config.collect_multi_exit_training
            and has_understanding
            and depth_config.router_candidates_understanding
            and request is not None
            and _nonempty(request.ce_loss_indexes)
        )
        collect_generation_router_exits = (
            depth_config.router_enabled
            and depth_config.collect_multi_exit_training
            and has_generation
            and depth_config.router_candidates_generation
            and request is not None
            and _nonempty(request.mse_loss_indexes)
        )
        understanding_candidates = (
            depth_config.router_candidates_understanding
            if collect_router_exits
            else []
        )
        generation_candidates = (
            depth_config.router_candidates_generation
            if collect_generation_router_exits
            else []
        )
        active_router_sample_ids = None
        label_router_sample_ids = None
        router_prefix_positions = None
        router_sample_tasks = None
        if collect_router_exits:
            assert request is not None
            assert request.ce_loss_indexes is not None
            label_positions = _positions_from_indexes(
                request.ce_loss_indexes,
                sequence_length=packed_sequence.shape[0],
            )
            packed_label_sample_ids = _sample_ids_for_positions(
                label_positions, sample_lens
            )
            active_router_sample_ids = torch.unique(
                packed_label_sample_ids, sorted=True
            )
            label_router_sample_ids = torch.searchsorted(
                active_router_sample_ids, packed_label_sample_ids
            )
            router_prefix_positions = _router_prefix_positions(
                packed_und_token_indexes,
                label_positions,
                packed_label_sample_ids,
                sample_lens,
                active_router_sample_ids,
            )
            if request.sample_tasks is not None:
                if len(request.sample_tasks) != len(sample_lens):
                    raise ValueError(
                        "DepthRequest.sample_tasks must align with sample_lens"
                    )
                router_sample_tasks = tuple(
                    request.sample_tasks[int(sample_id)]
                    for sample_id in active_router_sample_ids.detach().cpu().tolist()
                )
        active_generation_router_sample_ids = None
        label_generation_router_sample_ids = None
        generation_router_positions = None
        generation_router_sample_tasks = None
        if collect_generation_router_exits:
            assert request is not None
            assert request.mse_loss_indexes is not None
            generation_label_positions = _positions_from_indexes(
                request.mse_loss_indexes,
                sequence_length=packed_sequence.shape[0],
            )
            packed_generation_sample_ids = _sample_ids_for_positions(
                generation_label_positions, sample_lens
            )
            active_generation_router_sample_ids = torch.unique(
                packed_generation_sample_ids, sorted=True
            )
            label_generation_router_sample_ids = torch.searchsorted(
                active_generation_router_sample_ids,
                packed_generation_sample_ids,
            )
            generation_router_positions = (
                packed_gen_token_indexes
                if packed_gen_token_indexes is not None
                else generation_label_positions
            )
            if request.sample_tasks is not None:
                if len(request.sample_tasks) != len(sample_lens):
                    raise ValueError(
                        "DepthRequest.sample_tasks must align with sample_lens"
                    )
                generation_router_sample_tasks = tuple(
                    request.sample_tasks[int(sample_id)]
                    for sample_id in active_generation_router_sample_ids.detach()
                    .cpu()
                    .tolist()
                )
        execution_depth = max(
            [
                plan.execution_depth,
                *understanding_candidates,
                *generation_candidates,
                *fusion_sealing_candidates,
            ]
        )
        self._flexumm_aux = (
            {
                "candidate_depths": [],
                "exit_hidden_states": [],
                "router_logits": [],
                "router_risk_means": [],
                "router_risk_scales": [],
                "active_sample_ids": active_router_sample_ids,
                "label_sample_ids": label_router_sample_ids,
                "sample_tasks": router_sample_tasks,
            }
            if collect_router_exits
            else None
        )
        self._flexumm_generation_aux = (
            {
                "candidate_depths": [],
                "exit_hidden_states": [],
                "router_logits": [],
                "router_risk_means": [],
                "router_risk_scales": [],
                "active_sample_ids": active_generation_router_sample_ids,
                "label_sample_ids": label_generation_router_sample_ids,
                "sample_tasks": generation_router_sample_tasks,
            }
            if collect_generation_router_exits
            else None
        )
        fusion_token_indexes = (
            packed_und_token_indexes
            if fusion_sealing_kind == "understanding"
            else packed_gen_token_indexes
            if fusion_sealing_kind == "generation"
            else None
        )
        # During teacher-forced training, answer/label positions are present
        # in the packed sequence but are unavailable at inference time.  Use
        # only the prefix visible before the first supervised label to build
        # the capsule/controller features, while still applying the learned
        # realization to the full understanding token set for the task loss.
        fusion_realization_token_indexes = fusion_token_indexes
        fusion_condition_token_indexes = fusion_token_indexes
        if (
            self.training
            and fusion_sealing_kind == "understanding"
            and request is not None
            and _nonempty(request.ce_loss_indexes)
        ):
            label_positions = _positions_from_indexes(
                request.ce_loss_indexes,
                sequence_length=packed_sequence.shape[0],
            )
            label_sample_ids = _sample_ids_for_positions(
                label_positions, sample_lens
            )
            fusion_active_sample_ids = torch.unique(
                label_sample_ids, sorted=True
            )
            fusion_condition_token_indexes = _router_prefix_positions(
                packed_und_token_indexes,
                label_positions,
                label_sample_ids,
                sample_lens,
                fusion_active_sample_ids,
            )
        fusion_active_sample_ids = (
            self._fusion_active_sample_ids(
                packed_sequence,
                token_indexes=fusion_condition_token_indexes,
                sample_lens=sample_lens,
            )
            if fusion_sealing_kind is not None
            else None
        )
        self._flexumm_fusion_aux = (
            {
                "candidate_depths": [],
                "controller_logits": [],
                "realization_hidden_states": [],
                "baseline_hidden_states": [],
                "active_sample_ids": fusion_active_sample_ids,
                "label_sample_ids": (
                    _sample_ids_for_positions(
                        _positions_from_indexes(
                            request.ce_loss_indexes,
                            sequence_length=packed_sequence.shape[0],
                        ),
                        sample_lens,
                    )
                    if request is not None
                    and _nonempty(request.ce_loss_indexes)
                    else None
                ),
                "mse_sample_ids": (
                    _sample_ids_for_positions(
                        _positions_from_indexes(
                            request.mse_loss_indexes,
                            sequence_length=packed_sequence.shape[0],
                        ),
                        sample_lens,
                    )
                    if request is not None
                    and _nonempty(request.mse_loss_indexes)
                    else None
                ),
                "kind": fusion_sealing_kind,
                "sample_lens": tuple(int(length) for length in sample_lens),
                "sample_tasks": (
                    tuple(request.sample_tasks)
                    if request is not None and request.sample_tasks is not None
                    else tuple(
                        request.task if request is not None else fusion_sealing_kind
                        for _ in sample_lens
                    )
                ),
            }
            if fusion_sealing_kind is not None
            else None
        )

        if self.config.freeze_und and packed_und_token_indexes is not None:
            packed_sequence[packed_und_token_indexes] = packed_sequence[
                packed_und_token_indexes
            ].detach()

        cos, sin = self.rotary_emb(
            packed_sequence, packed_position_ids.unsqueeze(0)
        )
        packed_position_embeddings = (cos.squeeze(0), sin.squeeze(0))

        extra_inputs: dict[str, Any] = {}
        if self.use_moe:
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )

        understanding_output: torch.Tensor | None = None
        generation_output: torch.Tensor | None = None
        previous_router_hidden_states: torch.Tensor | None = None
        previous_generation_router_hidden_states: torch.Tensor | None = None
        fusion_source_hidden_states: torch.Tensor | None = None
        for layer_number, decoder_layer in enumerate(
            self.layers[:execution_depth], start=1
        ):
            if depth_config.tafe_enabled or depth_config.tafe_attention_enabled:
                self._reset_tafe_route_state()
            packed_sequence = decoder_layer(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                **extra_inputs,
            )
            if (
                depth_config.tafe_enabled
                and self._tafe_all_active_routes_exited(
                    has_understanding=has_understanding,
                    has_generation=has_generation,
                )
            ):
                break
            if has_understanding:
                packed_sequence = self._apply_understanding_adapter(
                    packed_sequence,
                    layer_number,
                    packed_und_token_indexes,
                    sample_lens,
                )
                if (
                    layer_number
                    == depth_config.understanding_fusion_source_depth
                ):
                    fusion_source_hidden_states = packed_sequence
                if (
                    layer_number
                    == depth_config.understanding_fusion_target_depth
                ):
                    packed_sequence = self._apply_understanding_fusion(
                        fusion_source_hidden_states,
                        packed_sequence,
                        packed_und_token_indexes,
                        sample_lens,
                    )
            if (
                self._flexumm_fusion_aux is not None
                and layer_number in fusion_sealing_candidates
            ):
                assert fusion_sealing_kind is not None
                capsule = self._compile_fusion_capsule(
                    packed_sequence,
                    kind=fusion_sealing_kind,
                    task=(
                        request.sample_tasks
                        if request is not None and request.sample_tasks is not None
                        else request.task
                        if request is not None
                        else fusion_sealing_kind
                    ),
                    token_indexes=fusion_condition_token_indexes,
                    sample_lens=sample_lens,
                    active_sample_ids=fusion_active_sample_ids,
                )
                controller = getattr(
                    self, "fusion_release_controller", None
                )
                if not isinstance(controller, FusionReleaseController):
                    raise RuntimeError(
                        "Fusion sealing controller is not configured"
                    )
                controller_hidden_states = packed_sequence
                controller_capsule = capsule
                if depth_config.fusion_sealing_detach_controller_features:
                    controller_hidden_states = packed_sequence.detach()
                    controller_capsule = capsule.detach()
                realized_sequence = self._apply_fusion_realization(
                    packed_sequence,
                    capsule,
                    kind=fusion_sealing_kind,
                    token_indexes=fusion_realization_token_indexes,
                    sample_lens=sample_lens,
                    active_sample_ids=fusion_active_sample_ids,
                )
                controller_realization_delta = (
                    realized_sequence - packed_sequence
                    if depth_config.fusion_sealing_condition_on_realization_delta
                    else None
                )
                controller_logits = controller(
                    controller_hidden_states,
                    controller_capsule,
                    depth=layer_number,
                    kind=fusion_sealing_kind,
                    task=(
                        request.sample_tasks
                        if request is not None and request.sample_tasks is not None
                        else request.task
                        if request is not None
                        else fusion_sealing_kind
                    ),
                    timestep=(
                        plan.timestep
                        if fusion_sealing_kind == "generation"
                        else None
                    ),
                    token_indexes=fusion_condition_token_indexes,
                    sample_lens=sample_lens,
                    active_sample_ids=fusion_active_sample_ids,
                    realization_delta=controller_realization_delta,
                )
                if fusion_sealing_kind == "understanding":
                    if request is None or not _nonempty(request.ce_loss_indexes):
                        raise RuntimeError(
                            "Understanding fusion sealing training requires CE indexes"
                        )
                    realization_hidden = self.norm(
                        realized_sequence[request.ce_loss_indexes]
                    )
                    baseline_hidden = self.norm(
                        packed_sequence[request.ce_loss_indexes]
                    )
                else:
                    if request is None or not _nonempty(request.mse_loss_indexes):
                        raise RuntimeError(
                            "Generation fusion sealing training requires MSE indexes"
                        )
                    realization_hidden = (
                        self.norm_moe_gen(
                            realized_sequence[request.mse_loss_indexes]
                        )
                        if self.use_moe
                        else self.norm(
                            realized_sequence[request.mse_loss_indexes]
                        )
                    )
                    baseline_hidden = (
                        self.norm_moe_gen(
                            packed_sequence[request.mse_loss_indexes]
                        )
                        if self.use_moe
                        else self.norm(
                            packed_sequence[request.mse_loss_indexes]
                        )
                    )
                self._flexumm_fusion_aux["candidate_depths"].append(
                    layer_number
                )
                self._flexumm_fusion_aux["controller_logits"].append(
                    controller_logits
                )
                self._flexumm_fusion_aux["realization_hidden_states"].append(
                    realization_hidden
                )
                self._flexumm_fusion_aux["baseline_hidden_states"].append(
                    baseline_hidden
                )
            if (
                has_understanding
                and plan.understanding_depth == layer_number
                and packed_und_token_indexes is not None
            ):
                understanding_output = self.norm(
                    packed_sequence[packed_und_token_indexes]
                )
                if self.config.freeze_und:
                    understanding_output = understanding_output.detach()
            if (
                has_generation
                and plan.generation_depth == layer_number
                and packed_gen_token_indexes is not None
            ):
                generation_output = (
                    self.norm_moe_gen(
                        packed_sequence[packed_gen_token_indexes]
                    )
                    if self.use_moe
                    else self.norm(
                        packed_sequence[packed_gen_token_indexes]
                    )
                )
            if collect_router_exits and layer_number in understanding_candidates:
                assert request is not None
                assert request.ce_loss_indexes is not None
                exit_hidden = self.norm(
                    packed_sequence[request.ce_loss_indexes]
                )
                route_logit, risk_mean, risk_scale = self.depth_router(
                    packed_sequence,
                    task=router_sample_tasks if router_sample_tasks is not None else request.task,
                    depth=layer_number,
                    kind="understanding",
                    timestep=None,
                    token_indexes=router_prefix_positions,
                    detach_features=depth_config.router_detach_features,
                    sample_lens=sample_lens,
                    active_sample_ids=active_router_sample_ids,
                    previous_hidden_states=previous_router_hidden_states,
                    return_risk=True,
                )
                self._flexumm_aux["candidate_depths"].append(layer_number)
                self._flexumm_aux["exit_hidden_states"].append(exit_hidden)
                self._flexumm_aux["router_logits"].append(route_logit)
                self._flexumm_aux["router_risk_means"].append(risk_mean)
                self._flexumm_aux["router_risk_scales"].append(risk_scale)
                previous_router_hidden_states = packed_sequence.detach()
            if (
                collect_generation_router_exits
                and layer_number in generation_candidates
            ):
                assert request is not None
                assert request.mse_loss_indexes is not None
                exit_hidden = (
                    self.norm_moe_gen(packed_sequence[request.mse_loss_indexes])
                    if self.use_moe
                    else self.norm(packed_sequence[request.mse_loss_indexes])
                )
                route_logit, risk_mean, risk_scale = self.generation_depth_router(
                    packed_sequence,
                    task=(
                        generation_router_sample_tasks
                        if generation_router_sample_tasks is not None
                        else request.task
                    ),
                    depth=layer_number,
                    kind="generation",
                    timestep=plan.timestep,
                    token_indexes=generation_router_positions,
                    detach_features=depth_config.router_detach_features,
                    sample_lens=sample_lens,
                    active_sample_ids=active_generation_router_sample_ids,
                    previous_hidden_states=previous_generation_router_hidden_states,
                    return_risk=True,
                )
                self._flexumm_generation_aux["candidate_depths"].append(layer_number)
                self._flexumm_generation_aux["exit_hidden_states"].append(
                    exit_hidden
                )
                self._flexumm_generation_aux["router_logits"].append(route_logit)
                self._flexumm_generation_aux["router_risk_means"].append(risk_mean)
                self._flexumm_generation_aux["router_risk_scales"].append(
                    risk_scale
                )
                previous_generation_router_hidden_states = packed_sequence.detach()

        # Start from the normal final-depth representation, then overwrite only
        # the positions consumed by the understanding head with its early exit.
        if self.use_moe:
            output = torch.zeros_like(packed_sequence)
            if packed_und_token_indexes is not None:
                output[packed_und_token_indexes] = self.norm(
                    packed_sequence[packed_und_token_indexes]
                )
            if packed_gen_token_indexes is not None:
                output[packed_gen_token_indexes] = self.norm_moe_gen(
                    packed_sequence[packed_gen_token_indexes]
                )
        else:
            output = self.norm(packed_sequence)

        if understanding_output is not None:
            output[packed_und_token_indexes] = understanding_output
        if generation_output is not None:
            output[packed_gen_token_indexes] = generation_output
        return output

    def _prediction_entropy(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized next-token entropy for each packed sample.

        This is deliberately an evaluation-only policy signal. The language
        model head is held through a weak reference so attaching it to the
        dynamic backbone does not create a duplicate checkpoint namespace.
        Entropy is computed from the last currently available causal state of
        each sample, before any answer tokens are generated.
        """

        head_ref = getattr(self, "_flexumm_lm_head_ref", None)
        if head_ref is None or head_ref() is None:
            raise RuntimeError(
                "Prediction-entropy routing requires the language-model head"
            )
        lm_head = head_ref()
        assert lm_head is not None
        lengths = [int(value) for value in query_lens.detach().cpu().tolist()]
        if not lengths or any(length < 1 for length in lengths):
            raise ValueError("query_lens must contain positive sample lengths")
        if sum(lengths) != int(packed_query_sequence.shape[0]):
            raise ValueError(
                "query_lens must sum to the packed query sequence length"
            )
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        last_states = torch.stack(
            [packed_query_sequence[offsets[i + 1] - 1] for i in range(len(lengths))],
            dim=0,
        )
        norm_weight = self.norm.weight
        last_states = last_states.to(
            device=norm_weight.device,
            dtype=norm_weight.dtype,
        )
        normalized = self.norm(last_states)
        head_weight = lm_head.weight
        logits = lm_head(
            normalized.to(device=head_weight.device, dtype=head_weight.dtype)
        )
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = entropy / torch.log(
            logits.new_tensor(float(logits.shape[-1])).float()
        )
        return entropy.to(device=packed_query_sequence.device)

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values=None,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:
        request = current_depth_request()
        # BAGEL deliberately encodes a text-to-image prompt through the MoT
        # ``und`` expert stream before its image latents use ``gen``. Keep the
        # expert stream (`mode`) separate from the capability route (`kind`)
        # so understanding-only adapters cannot leak into generation prompt
        # conditioning.
        kind = (
            request.kind
            if request is not None and request.kind is not None
            else "generation"
            if mode == "gen"
            else "understanding"
        )
        locked_understanding_depth = (
            getattr(
                past_key_values,
                "_flexumm_locked_understanding_depth",
                None,
            )
            if kind == "understanding" and past_key_values is not None
            else None
        )
        requested_override = (
            request.depth_override if request is not None else None
        )
        locked_fusion_depth = (
            getattr(past_key_values, "_flexumm_sealed_depth", None)
            if past_key_values is not None
            else None
        )
        locked_fusion_capsule = (
            getattr(past_key_values, "_flexumm_sealed_capsule", None)
            if past_key_values is not None
            else None
        )
        fusion_locked = (
            locked_fusion_depth is not None
            and locked_fusion_capsule is not None
            and getattr(past_key_values, "_flexumm_sealed_kind", None) == kind
        )
        plan = self.dynamic_depth_controller.resolve(
            kind=kind,
            task=request.task if request is not None else None,
            timestep=request.timestep if request is not None else None,
            depth_override=(
                requested_override
                if requested_override is not None
                else locked_understanding_depth
                if locked_understanding_depth is not None
                else locked_fusion_depth
                if fusion_locked
                else None
            ),
            has_understanding=kind == "understanding",
            has_generation=kind == "generation",
        )

        enable_taylorseer = getattr(self, "enable_taylorseer", False)
        if enable_taylorseer:
            # TaylorSeer assumes a stable full layer stream.
            plan = self.dynamic_depth_controller.resolve(
                kind=kind,
                task=request.task if request is not None else None,
                timestep=request.timestep if request is not None else None,
                depth_override=len(self.layers),
                has_understanding=kind == "understanding",
                has_generation=kind == "generation",
            )

        depth_config = self.dynamic_depth_controller.config
        fusion_release_active = (
            self._fusion_sealing_active(kind)
            and not enable_taylorseer
            and not self.training
            and requested_override is None
            and not fusion_locked
        )
        fusion_candidates = (
            sorted(set([*self._fusion_sealing_candidates(kind), plan.execution_depth]))
            if fusion_release_active
            else []
        )
        understanding_router_active = (
            depth_config.router_enabled
            and kind == "understanding"
            and bool(depth_config.router_candidates_understanding)
            and not enable_taylorseer
            and not self.training
            and requested_override is None
            and locked_understanding_depth is None
            and not fusion_release_active
            and not fusion_locked
        )
        generation_router_active = (
            depth_config.router_enabled
            and kind == "generation"
            and mode == "gen"
            and bool(depth_config.router_candidates_generation)
            and not enable_taylorseer
            and not self.training
            and requested_override is None
            and not fusion_release_active
            and not fusion_locked
        )
        router_active = (
            understanding_router_active or generation_router_active
        )
        router_conditioning = (
            depth_config.router_conditioning_generation
            if kind == "generation"
            and depth_config.router_conditioning_generation is not None
            else depth_config.router_conditioning
        )
        router_module = (
            self.generation_depth_router
            if generation_router_active
            else self.depth_router
            if understanding_router_active
            else None
        )
        router_candidates = (
            self.dynamic_depth_controller.router_candidates(
                kind=kind,
                task=request.task if request is not None else kind,
            )
            if router_active
            else []
        )
        execution_depth = (
            fusion_candidates[-1]
            if fusion_release_active
            else router_candidates[-1]
            if router_active
            else plan.execution_depth
        )
        has_prior_cache = (
            past_key_values is not None and past_key_values.seq_lens > 0
        )

        if (
            past_key_values is not None
            and not router_active
            and not fusion_release_active
        ):
            self._materialize_cache_to_depth(past_key_values, plan.execution_depth)

        if enable_taylorseer:
            from modeling.cache_utils.taylorseer import cal_type

            cal_type(self.cache_dic, self.current)
            self.current["stream"] = "layers_stream"

        cos, sin = self.rotary_emb(
            packed_query_sequence, packed_query_position_ids.unsqueeze(0)
        )
        packed_query_position_embeddings = (cos.squeeze(0), sin.squeeze(0))

        extra_inputs: dict[str, Any] = {}
        if self.use_moe:
            extra_inputs["mode"] = mode
            if mode == "gen":
                if packed_vae_token_indexes is None or packed_text_indexes is None:
                    raise ValueError(
                        "MoT generation requires VAE-token and text-token indexes"
                    )
                extra_inputs.update(
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_text_indexes=packed_text_indexes,
                )

        selected_depth = execution_depth
        router_trace: list[dict[str, float | int]] = []
        fusion_trace: list[dict[str, float | int | str]] = []
        selected_fusion_capsule = (
            locked_fusion_capsule if fusion_locked else None
        )
        selected_fusion_sample_ids = None
        selected_fusion_action = "sealed" if fusion_locked else "continue"
        previous_router_hidden_states: torch.Tensor | None = None
        fusion_source_hidden_states: torch.Tensor | None = None
        fusion_token_indexes = (
            packed_vae_token_indexes
            if kind == "generation" and mode == "gen"
            else None
        )
        fusion_sample_lens = [int(length) for length in query_lens.detach().cpu()]
        fusion_active_sample_ids = (
            self._fusion_active_sample_ids(
                packed_query_sequence,
                token_indexes=fusion_token_indexes,
                sample_lens=fusion_sample_lens,
            )
            if fusion_release_active
            else None
        )
        if fusion_locked:
            selected_fusion_sample_ids = fusion_active_sample_ids
        router_threshold = self.dynamic_depth_controller.router_threshold(
            request.task if request is not None else kind
        )
        for layer_idx, decoder_layer in enumerate(
            self.layers[:execution_depth]
        ):
            if depth_config.tafe_enabled or depth_config.tafe_attention_enabled:
                self._reset_tafe_route_state()
            if has_prior_cache and (router_active or fusion_release_active):
                self._materialize_cache_to_depth(past_key_values, layer_idx + 1)
            if enable_taylorseer:
                decoder_layer.current = self.current
                decoder_layer.cache_dic = self.cache_dic
                decoder_layer.enable_taylorseer = True
                self.current["layer"] = layer_idx
            packed_query_sequence, past_key_values = decoder_layer(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                **extra_inputs,
            )
            layer_number = layer_idx + 1
            if (
                depth_config.tafe_enabled
                and self._tafe_all_active_routes_exited(
                    has_understanding=kind == "understanding",
                    has_generation=kind == "generation",
                )
            ):
                selected_depth = layer_number
                break
            if kind == "understanding":
                packed_query_sequence = self._apply_understanding_adapter(
                    packed_query_sequence,
                    layer_number,
                    sample_lens=query_lens,
                )
                if (
                    layer_number
                    == depth_config.understanding_fusion_source_depth
                ):
                    fusion_source_hidden_states = packed_query_sequence
                if (
                    layer_number
                    == depth_config.understanding_fusion_target_depth
                ):
                    packed_query_sequence = self._apply_understanding_fusion(
                        fusion_source_hidden_states,
                        packed_query_sequence,
                        sample_lens=query_lens,
                    )
            if (
                fusion_release_active
                and layer_number in fusion_candidates
            ):
                capsule = self._compile_fusion_capsule(
                    packed_query_sequence,
                    kind=kind,
                    task=(
                        request.sample_tasks
                        if request is not None and request.sample_tasks is not None
                        else request.task
                        if request is not None
                        else kind
                    ),
                    token_indexes=fusion_token_indexes,
                    sample_lens=fusion_sample_lens,
                    active_sample_ids=fusion_active_sample_ids,
                )
                controller = getattr(
                    self, "fusion_release_controller", None
                )
                if not isinstance(controller, FusionReleaseController):
                    raise RuntimeError(
                        "Fusion sealing controller is not configured"
                    )
                controller_hidden_states = packed_query_sequence
                controller_capsule = capsule
                if depth_config.fusion_sealing_detach_controller_features:
                    controller_hidden_states = packed_query_sequence.detach()
                    controller_capsule = capsule.detach()
                realized_sequence = self._apply_fusion_realization(
                    packed_query_sequence,
                    capsule,
                    kind=kind,
                    token_indexes=fusion_token_indexes,
                    sample_lens=fusion_sample_lens,
                    active_sample_ids=fusion_active_sample_ids,
                )
                controller_realization_delta = (
                    realized_sequence - packed_query_sequence
                    if depth_config.fusion_sealing_condition_on_realization_delta
                    else None
                )
                controller_logits = controller(
                    controller_hidden_states,
                    controller_capsule,
                    depth=layer_number,
                    kind=kind,
                    task=request.task if request is not None else kind,
                    timestep=request.timestep if request is not None else None,
                    token_indexes=fusion_token_indexes,
                    sample_lens=fusion_sample_lens,
                    active_sample_ids=fusion_active_sample_ids,
                    realization_delta=controller_realization_delta,
                )
                seal_probability = torch.softmax(
                    controller_logits, dim=-1
                )[..., 1]
                is_final_candidate = layer_number == fusion_candidates[-1]
                early_thresholds = self._fusion_sealing_threshold_tensor(
                    depth_config.fusion_sealing_thresholds,
                    depth_config.fusion_sealing_threshold,
                    request=request,
                    active_sample_ids=fusion_active_sample_ids,
                    sample_lens=fusion_sample_lens,
                    batch_size=seal_probability.shape[0],
                    device=seal_probability.device,
                    dtype=seal_probability.dtype,
                )
                should_seal = (
                    not is_final_candidate
                    and bool(
                        torch.all(
                            seal_probability
                            >= early_thresholds
                        ).item()
                    )
                )
                if (
                    is_final_candidate
                    and depth_config.fusion_sealing_apply_final_realization
                ):
                    # The final candidate is normally only a teacher/reference
                    # state.  In the accuracy-path variant, retain its capsule
                    # so the lightweight realizer is applied once after the
                    # shared backbone finishes.  The optional quality gate
                    # makes this decision independently for each packed sample
                    # instead of perturbing every benchmark example.
                    if depth_config.fusion_sealing_gate_final_realization:
                        assert fusion_active_sample_ids is not None
                        final_default_threshold = (
                            depth_config.fusion_sealing_final_threshold
                            if depth_config.fusion_sealing_final_threshold
                            is not None
                            else depth_config.fusion_sealing_threshold
                        )
                        final_thresholds = self._fusion_sealing_threshold_tensor(
                            depth_config.fusion_sealing_final_thresholds,
                            final_default_threshold,
                            request=request,
                            active_sample_ids=fusion_active_sample_ids,
                            sample_lens=fusion_sample_lens,
                            batch_size=seal_probability.shape[0],
                            device=seal_probability.device,
                            dtype=seal_probability.dtype,
                        )
                        selected_mask = (
                            seal_probability
                            >= final_thresholds
                        )
                        if bool(selected_mask.any().item()):
                            selected_fusion_capsule = capsule[selected_mask]
                            selected_fusion_sample_ids = (
                                fusion_active_sample_ids[selected_mask]
                            )
                            selected_fusion_action = "final_gated"
                        else:
                            selected_fusion_capsule = None
                            selected_fusion_sample_ids = None
                            selected_fusion_action = "continue"
                    else:
                        selected_fusion_capsule = capsule
                        selected_fusion_sample_ids = fusion_active_sample_ids
                        selected_fusion_action = "final_realized"
                fusion_trace.append(
                    {
                        "depth": layer_number,
                        "seal_probability": float(
                            seal_probability.detach().float().mean().item()
                        ),
                        "action": "seal" if should_seal else "continue",
                    }
                )
                if should_seal:
                    selected_depth = layer_number
                    selected_fusion_capsule = capsule
                    selected_fusion_sample_ids = fusion_active_sample_ids
                    selected_fusion_action = "sealed"
                    break
            if router_active and layer_number in router_candidates:
                router_hidden_states = packed_query_sequence
                replay_state = getattr(
                    past_key_values, "_flexumm_replay_state", None
                )
                replay_hidden_states = (
                    [
                        segment.hidden
                        for segment in replay_state.segments
                        if segment.hidden is not None
                        and segment.depth == layer_number
                    ]
                    if replay_state is not None
                    else []
                )
                if replay_hidden_states:
                    router_hidden_states = torch.cat(
                        [*replay_hidden_states, packed_query_sequence],
                        dim=0,
                    )
                has_change_signal = previous_router_hidden_states is not None
                if router_conditioning == "prediction_entropy":
                    if kind != "understanding":
                        raise ValueError(
                            "Prediction-entropy routing is only defined for "
                            "the understanding/token route"
                        )
                    prediction_entropy = self._prediction_entropy(
                        packed_query_sequence,
                        query_lens,
                    )
                    entropy_value = prediction_entropy.float().mean()
                    router_trace.append(
                        {
                            "depth": layer_number,
                            "prediction_entropy": float(entropy_value.item()),
                            "entropy_threshold": float(
                                depth_config.router_prediction_entropy_threshold
                            ),
                        }
                    )
                    should_exit = (
                        depth_config.router_min_exit_depth is None
                        or layer_number >= depth_config.router_min_exit_depth
                    ) and entropy_value.item() <= (
                        depth_config.router_prediction_entropy_threshold
                    )
                else:
                    assert router_module is not None
                    route_logit, risk_mean, risk_scale = router_module(
                        router_hidden_states,
                        task=request.task if request is not None else kind,
                        depth=layer_number,
                        kind=kind,
                        timestep=request.timestep if request is not None else None,
                        token_indexes=None,
                        detach_features=True,
                        previous_hidden_states=previous_router_hidden_states,
                        return_risk=True,
                    )
                    halt_probability = torch.sigmoid(
                        route_logit / depth_config.router_temperature
                    )
                    risk_upper_bound = (
                        risk_mean
                        + depth_config.router_risk_confidence * risk_scale
                    )
                    router_trace.append(
                        {
                            "depth": layer_number,
                            "halt_probability": float(
                                halt_probability.detach().float().item()
                            ),
                            "predicted_harmful_regret": float(
                                risk_mean.detach().float().item()
                            ),
                            "risk_scale": float(
                                risk_scale.detach().float().item()
                            ),
                            "risk_upper_bound": float(
                                risk_upper_bound.detach().float().item()
                            ),
                        }
                    )
                    should_exit = (
                        (
                            depth_config.router_min_exit_depth is None
                            or layer_number
                            >= depth_config.router_min_exit_depth
                        )
                        and halt_probability.item() >= router_threshold
                        and (
                            depth_config.router_quality_prediction_weight == 0
                            or risk_upper_bound.item()
                            <= depth_config.router_risk_threshold
                        )
                    )
                # A representation-change-only policy cannot make a genuine
                # layer-to-layer decision at its first candidate. Always
                # continue there unless it is the final fallback candidate.
                if router_conditioning == "representation_change":
                    should_exit = should_exit and has_change_signal
                if should_exit or layer_number == router_candidates[-1]:
                    selected_depth = layer_number
                    break
                previous_router_hidden_states = router_hidden_states.detach()

        selected_fusion_token_indexes = fusion_token_indexes
        if (
            selected_fusion_capsule is not None
            and depth_config.fusion_sealing_gate_final_realization
            and selected_fusion_sample_ids is not None
            and fusion_active_sample_ids is not None
            and selected_fusion_sample_ids.numel()
            < fusion_active_sample_ids.numel()
        ):
            selected_fusion_token_indexes = (
                self._fusion_token_indexes_for_samples(
                    packed_query_sequence,
                    token_indexes=fusion_token_indexes,
                    sample_lens=fusion_sample_lens,
                    selected_sample_ids=selected_fusion_sample_ids,
                )
            )
        if selected_fusion_capsule is not None:
            packed_query_sequence = self._apply_fusion_realization(
                packed_query_sequence,
                selected_fusion_capsule,
                kind=kind,
                token_indexes=selected_fusion_token_indexes,
                sample_lens=fusion_sample_lens,
                active_sample_ids=selected_fusion_sample_ids,
            )
        raw_query_sequence = packed_query_sequence
        self.last_router_decision = {
            "task": request.task if request is not None else kind,
            "selected_depth": selected_depth,
            "trace": router_trace,
            "locked": locked_understanding_depth is not None,
            "threshold": router_threshold,
            "conditioning": router_conditioning,
        }
        self.last_fusion_sealing_decision = {
            "task": request.task if request is not None else kind,
            "selected_depth": selected_depth,
            "action": selected_fusion_action,
            "trace": [dict(item) for item in fusion_trace],
            "locked": fusion_locked,
            "threshold": depth_config.fusion_sealing_threshold,
            "final_threshold": depth_config.fusion_sealing_final_threshold,
            "threshold_overrides": dict(depth_config.fusion_sealing_thresholds),
            "final_threshold_overrides": dict(
                depth_config.fusion_sealing_final_thresholds
            ),
        }
        # Preserve the prompt-time fusion decision across answer-token
        # forwards, mirroring ``last_prompt_router_decision``.  Benchmark
        # evaluators can therefore audit the final realization gate per item.
        self.last_prompt_fusion_sealing_decision = dict(
            self.last_fusion_sealing_decision
        )
        task_name = str(self.last_router_decision["task"])
        if router_active:
            # Preserve the prompt-time trace after answer-token forwards start
            # reporting the locked decision. Benchmark runners use this for
            # per-example calibration without changing model outputs.
            self.last_prompt_router_decision = {
                "task": self.last_router_decision["task"],
                "selected_depth": selected_depth,
                "trace": [dict(item) for item in router_trace],
                "locked": False,
                "threshold": router_threshold,
                "conditioning": router_conditioning,
            }
            # Keep compact runtime statistics for calibration audits.  The
            # selected-depth histogram proves what happened; these summaries
            # additionally show whether the configured threshold was near
            # any candidate's halt-probability distribution.
            trace_stats = getattr(self, "_router_trace_stats", None)
            if trace_stats is None:
                trace_stats = {}
                self._router_trace_stats = trace_stats
            for trace_item in router_trace:
                probability = trace_item.get("halt_probability")
                depth = trace_item.get("depth")
                if probability is None or depth is None:
                    continue
                key = (task_name, int(depth))
                summary = trace_stats.setdefault(
                    key,
                    {"count": 0, "sum": 0.0, "min": 1.0, "max": 0.0},
                )
                value = float(probability)
                summary["count"] += 1
                summary["sum"] += value
                summary["min"] = min(float(summary["min"]), value)
                summary["max"] = max(float(summary["max"]), value)
        if not hasattr(self, "_router_depth_counts"):
            self._router_depth_counts = Counter()
        if router_active:
            # Count one exit decision per prompt.  Cached answer-token forwards
            # reuse this decision and must not inflate the sample histogram.
            self._router_depth_counts[(task_name, selected_depth)] += 1
        if kind == "understanding" and (
            router_active or locked_understanding_depth is not None
        ):
            if not hasattr(self, "_router_layer_token_counts"):
                self._router_layer_token_counts = Counter()
            query_token_count = int(packed_query_sequence.shape[0])
            if router_active:
                replay_state = getattr(
                    past_key_values, "_flexumm_replay_state", None
                )
                if replay_state is not None:
                    query_token_count += sum(
                        int(segment.query_lens.sum().detach().cpu().item())
                        for segment in replay_state.segments
                    )
            self._router_layer_token_counts[
                (task_name, selected_depth)
            ] += query_token_count
        if (
            understanding_router_active
            and update_past_key_values
            and past_key_values is not None
        ):
            past_key_values._flexumm_locked_understanding_depth = selected_depth
        if (
            selected_fusion_action == "sealed"
            and selected_fusion_capsule is not None
            and update_past_key_values
            and past_key_values is not None
        ):
            past_key_values._flexumm_sealed_depth = int(selected_depth)
            past_key_values._flexumm_sealed_kind = kind
            past_key_values._flexumm_sealed_capsule = (
                selected_fusion_capsule.detach()
            )
        if update_past_key_values and past_key_values is not None:
            self._append_replay_segment(
                past_key_values=past_key_values,
                hidden=raw_query_sequence,
                depth=selected_depth,
                query_lens=query_lens,
                position_ids=packed_query_position_ids,
                query_indexes=packed_query_indexes,
                key_values_lens=key_values_lens,
                key_value_indexes=packed_key_value_indexes,
                is_causal=is_causal,
                mode=mode,
                vae_token_indexes=packed_vae_token_indexes,
                text_indexes=packed_text_indexes,
            )

        if self.use_moe:
            if mode == "und":
                packed_query_sequence = self.norm(raw_query_sequence)
            else:
                packed_query_sequence = torch.zeros_like(raw_query_sequence)
                packed_query_sequence[packed_text_indexes] = self.norm(
                    raw_query_sequence[packed_text_indexes]
                )
                packed_query_sequence[packed_vae_token_indexes] = self.norm_moe_gen(
                    raw_query_sequence[packed_vae_token_indexes]
                )
        else:
            packed_query_sequence = self.norm(raw_query_sequence)

        if enable_taylorseer:
            self.current["step"] += 1

        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )

    def _append_replay_segment(
        self,
        *,
        past_key_values,
        hidden: torch.Tensor,
        depth: int,
        query_lens: torch.Tensor,
        position_ids: torch.Tensor,
        query_indexes: torch.Tensor,
        key_values_lens: torch.Tensor | None,
        key_value_indexes: torch.Tensor | None,
        is_causal: bool,
        mode: str,
        vae_token_indexes: torch.Tensor | None,
        text_indexes: torch.Tensor | None,
    ) -> None:
        state = getattr(past_key_values, "_flexumm_replay_state", None)
        if state is None:
            state = _ReplayState()
            past_key_values._flexumm_replay_state = state

        empty_int = query_lens.new_empty((0,))
        segment_hidden = None if depth == len(self.layers) else hidden.detach()
        state.segments.append(
            _ReplaySegment(
                hidden=segment_hidden,
                depth=depth,
                query_lens=query_lens.detach().clone(),
                position_ids=position_ids.detach().clone(),
                query_indexes=query_indexes.detach().clone(),
                key_values_lens=(
                    key_values_lens.detach().clone()
                    if key_values_lens is not None
                    else torch.zeros_like(query_lens)
                ),
                key_value_indexes=(
                    key_value_indexes.detach().clone()
                    if key_value_indexes is not None
                    else empty_int
                ),
                is_causal=is_causal,
                mode=mode,
                vae_token_indexes=_clone_optional(vae_token_indexes),
                text_indexes=_clone_optional(text_indexes),
            )
        )

    def _materialize_cache_to_depth(self, past_key_values, target_depth: int) -> None:
        state: _ReplayState | None = getattr(
            past_key_values, "_flexumm_replay_state", None
        )
        missing_layers = [
            layer_idx
            for layer_idx in range(target_depth)
            if past_key_values.key_cache[layer_idx] is None
        ]
        has_shallow_segments = state is not None and any(
            segment.depth < target_depth for segment in state.segments
        )
        if not missing_layers and not has_shallow_segments:
            return

        if state is None:
            if past_key_values.seq_lens == 0:
                return
            raise RuntimeError(
                "The KV cache predates FlexUMM and has missing deeper layers. "
                "Start a fresh inference context so replay metadata can be recorded."
            )
        if not self.dynamic_depth_controller.config.replay_deeper_kv:
            raise RuntimeError(
                "A deeper route was selected after an early route, but "
                "replay_deeper_kv is disabled."
            )

        for layer_idx in range(target_depth):
            for segment in state.segments:
                if segment.depth > layer_idx:
                    continue
                if segment.depth < layer_idx:
                    raise RuntimeError(
                        "Replay state is not contiguous; cannot safely deepen KV cache"
                    )
                if segment.hidden is None:
                    raise RuntimeError("Replay hidden state was released too early")

                cached = past_key_values.key_cache[layer_idx]
                actual_prefix = 0 if cached is None else int(cached.shape[0])
                expected_prefix = int(segment.key_values_lens.sum().item())
                if (
                    self.dynamic_depth_controller.config.strict_cache_validation
                    and actual_prefix != expected_prefix
                ):
                    raise RuntimeError(
                        "KV replay prefix mismatch at layer "
                        f"{layer_idx}: cache has {actual_prefix} tokens, "
                        f"segment expects {expected_prefix}."
                    )

                cos, sin = self.rotary_emb(
                    segment.hidden, segment.position_ids.unsqueeze(0)
                )
                position_embeddings = (cos.squeeze(0), sin.squeeze(0))
                extra_inputs: dict[str, Any] = {}
                if self.use_moe:
                    extra_inputs["mode"] = segment.mode
                    if segment.mode == "gen":
                        extra_inputs.update(
                            packed_vae_token_indexes=segment.vae_token_indexes,
                            packed_text_indexes=segment.text_indexes,
                        )

                decoder_layer = self.layers[layer_idx]
                had_taylorseer = hasattr(decoder_layer, "enable_taylorseer")
                old_taylorseer = getattr(
                    decoder_layer, "enable_taylorseer", False
                )
                decoder_layer.enable_taylorseer = False
                try:
                    segment.hidden, past_key_values = decoder_layer(
                        packed_query_sequence=segment.hidden,
                        query_lens=segment.query_lens,
                        packed_query_position_embeddings=position_embeddings,
                        packed_query_indexes=segment.query_indexes,
                        past_key_values=past_key_values,
                        key_values_lens=segment.key_values_lens,
                        packed_key_value_indexes=segment.key_value_indexes,
                        update_past_key_values=True,
                        is_causal=segment.is_causal,
                        **extra_inputs,
                    )
                    if segment.mode == "und":
                        segment.hidden = self._apply_understanding_adapter(
                            segment.hidden,
                            layer_idx + 1,
                        )
                finally:
                    if had_taylorseer:
                        decoder_layer.enable_taylorseer = old_taylorseer
                    else:
                        delattr(decoder_layer, "enable_taylorseer")

                segment.hidden = segment.hidden.detach()
                segment.depth += 1
                if segment.depth == len(self.layers):
                    segment.hidden = None


class DynamicBagel(_Bagel):
    """Thin context-propagating wrapper around the unmodified BAGEL model."""

    dynamic_depth_controller: DynamicDepthController

    def forward(
        self,
        *args,
        depth_task: str | None = None,
        depth_sample_tasks: list[str] | tuple[str, ...] | None = None,
        depth_override: int | None = None,
        **kwargs,
    ):
        bound = _BAGEL_FORWARD_SIGNATURE.bind_partial(self, *args, **kwargs)
        values = bound.arguments
        und_indexes = values.get("packed_text_indexes")
        vit_indexes = values.get("packed_vit_token_indexes")
        if _nonempty(vit_indexes):
            und_indexes = (
                torch.cat((und_indexes, vit_indexes))
                if _nonempty(und_indexes)
                else vit_indexes
            )
        gen_indexes = values.get("packed_vae_token_indexes")

        ce_indexes = values.get("ce_loss_indexes")
        mse_indexes = values.get("mse_loss_indexes")
        sample_lens = values.get("sample_lens")
        has_understanding = _nonempty(ce_indexes)
        has_generation = _nonempty(mse_indexes)
        kind = (
            "mixed"
            if has_understanding and has_generation
            else "generation"
            if has_generation
            else "understanding"
        )

        timestep = values.get("packed_timesteps")
        if has_generation and isinstance(timestep, torch.Tensor):
            timestep = torch.sigmoid(timestep.detach())
            shift = float(self.timestep_shift)
            timestep = shift * timestep / (1 + (shift - 1) * timestep)

        parent = current_depth_request()
        task = depth_task or (parent.task if parent is not None else None) or kind
        override = (
            depth_override
            if depth_override is not None
            else parent.depth_override
            if parent is not None
            else None
        )
        with self.dynamic_depth_controller.route(
            task=task,
            sample_tasks=depth_sample_tasks,
            sample_lens=sample_lens,
            kind=kind,
            timestep=timestep,
            depth_override=override,
            has_understanding=has_understanding,
            has_generation=has_generation,
            und_token_indexes=und_indexes,
            gen_token_indexes=gen_indexes,
            ce_loss_indexes=ce_indexes,
            mse_loss_indexes=mse_indexes,
        ):
            output = super().forward(*args, **kwargs)
        return self._add_router_training_losses(
            output=output,
            packed_label_ids=values.get("packed_label_ids"),
        )

    def _add_fusion_sealing_training_loss(
        self,
        *,
        output: dict[str, Any],
        packed_label_ids: torch.Tensor | None,
        mse_target: torch.Tensor | None,
    ) -> dict[str, Any]:
        """Train seal decisions from candidate realization quality.

        The last collected candidate is the full-fusion reference.  Earlier
        candidates receive a ``seal`` target when their task loss is within
        the configured quality margin of that reference; otherwise they are
        trained to ``continue``.  This is the executable oracle-label stage
        described by the FlexUMM method, without requiring benchmark scores in
        the training loop.
        """

        aux = getattr(self.language_model.model, "_flexumm_fusion_aux", None)
        if not self.training or not aux or not aux.get("controller_logits"):
            return output
        kind = aux.get("kind")
        if kind not in {"understanding", "generation"}:
            return output
        hidden_states = aux.get("realization_hidden_states") or []
        candidate_depths = list(aux.get("candidate_depths") or [])
        gate_final_realization = (
            self.dynamic_depth_controller.config.fusion_sealing_gate_final_realization
        )
        if len(hidden_states) != len(candidate_depths):
            raise RuntimeError(
                "Fusion sealing candidate states and depths are misaligned"
            )
        baseline_hidden_states = aux.get("baseline_hidden_states") or []
        if gate_final_realization and len(baseline_hidden_states) != len(candidate_depths):
            raise RuntimeError(
                "Fusion sealing baseline states and depths are misaligned"
            )
        if len(candidate_depths) == 0:
            return output

        if kind == "understanding":
            if packed_label_ids is None:
                return output
            per_candidate = []
            for hidden in hidden_states:
                logits = self.language_model.lm_head(hidden)
                per_candidate.append(
                    F.cross_entropy(
                        logits,
                        packed_label_ids,
                        reduction="none",
                    )
                )
            sample_ids = aux.get("label_sample_ids")
        else:
            if mse_target is None:
                return output
            per_candidate = []
            for hidden in hidden_states:
                prediction = self.llm2vae(hidden)
                per_candidate.append(
                    (prediction - mse_target.to(prediction)).square().mean(dim=-1)
                )
            sample_ids = aux.get("mse_sample_ids")

        if gate_final_realization and kind == "understanding":
            per_baseline = []
            for hidden in baseline_hidden_states:
                logits = self.language_model.lm_head(hidden)
                per_baseline.append(
                    F.cross_entropy(
                        logits,
                        packed_label_ids,
                        reduction="none",
                    )
                )
        elif gate_final_realization:
            per_baseline = []
            for hidden in baseline_hidden_states:
                prediction = self.llm2vae(hidden)
                per_baseline.append(
                    (prediction - mse_target.to(prediction)).square().mean(dim=-1)
                )

        if sample_ids is None:
            raise RuntimeError(
                "Fusion sealing training is missing per-sample loss indexes"
            )
        active_sample_ids = aux.get("active_sample_ids")
        if active_sample_ids is None or active_sample_ids.numel() < 1:
            raise RuntimeError("Fusion sealing training received no samples")
        if any(values.numel() != sample_ids.numel() for values in per_candidate):
            raise RuntimeError(
                "Fusion sealing candidate losses do not align with target indexes"
            )
        if gate_final_realization and any(
            values.numel() != sample_ids.numel() for values in per_baseline
        ):
            raise RuntimeError(
                "Fusion sealing baseline losses do not align with target indexes"
            )
        sample_ids = sample_ids.to(device=per_candidate[0].device)
        active_sample_ids = active_sample_ids.to(device=sample_ids.device)
        per_sample_cost = torch.stack(
            [
                torch.stack(
                    [
                        values[sample_ids == sample_id].mean()
                        for values in per_candidate
                    ]
                )
                for sample_id in active_sample_ids.detach().cpu().tolist()
            ],
            dim=0,
        )
        per_sample_baseline_cost = None
        if gate_final_realization:
            per_sample_baseline_cost = torch.stack(
                [
                    torch.stack(
                        [
                            values[sample_ids == sample_id].mean()
                            for values in per_baseline
                        ]
                    )
                    for sample_id in active_sample_ids.detach().cpu().tolist()
                ],
                dim=0,
            )
        controller_logits = torch.stack(
            aux["controller_logits"], dim=1
        ).to(device=per_sample_cost.device)
        if controller_logits.shape[:2] != per_sample_cost.shape:
            raise RuntimeError(
                "Fusion sealing controller outputs do not align with losses"
            )

        quality_margin = (
            self.dynamic_depth_controller.config.fusion_sealing_quality_margin
        )
        full_cost = per_sample_cost[:, -1:].detach()
        target = torch.zeros(
            per_sample_cost.shape[:2],
            device=per_sample_cost.device,
            dtype=torch.long,
        )
        if per_sample_cost.shape[1] > 1:
            target[:, :-1] = (
                per_sample_cost[:, :-1].detach()
                <= full_cost + quality_margin
            ).long()
        if gate_final_realization:
            # The final controller output is a quality gate, not an early-exit
            # label: select the residual only when it beats the unmodified
            # full-depth hidden state for this sample.
            target[:, -1] = (
                per_sample_cost[:, -1].detach()
                <= per_sample_baseline_cost[:, -1].detach() + quality_margin
            ).long()
        action_loss = F.cross_entropy(
            controller_logits.reshape(-1, 2), target.reshape(-1)
        )
        # Stage 1 must train the capsule and realizer before a useful seal
        # label exists.  The full-fusion candidate is the teacher; earlier
        # candidates learn to preserve its task realization.  Without this
        # term, detached controller features plus an all-zero initial seal
        # target leave the new capsule path with no gradient.
        preservation_terms = []
        if len(hidden_states) > 1:
            if kind == "generation":
                teacher = self.llm2vae(hidden_states[-1]).detach()
                for hidden in hidden_states[:-1]:
                    prediction = self.llm2vae(hidden)
                    preservation_terms.append(
                        (prediction - teacher.to(prediction)).pow(2).mean()
                    )
            else:
                teacher_logits = self.language_model.lm_head(
                    hidden_states[-1]
                ).detach()
                teacher_probs = torch.softmax(teacher_logits.float(), dim=-1)
                for hidden in hidden_states[:-1]:
                    student_logits = self.language_model.lm_head(hidden)
                    preservation_terms.append(
                        F.kl_div(
                            F.log_softmax(student_logits.float(), dim=-1),
                            teacher_probs,
                            reduction="batchmean",
                        )
                    )
        preservation_loss = (
            torch.stack(preservation_terms).mean()
            if preservation_terms
            else action_loss.new_zeros(())
        )
        # Optional task weights apply to the shared realization objective.
        # They are resolved per packed sample and normalized to mean one, so
        # changing the mix cannot silently change the overall loss scale.
        task_sample_weights = None
        task_loss_weights = (
            self.dynamic_depth_controller.config.fusion_sealing_task_loss_weights
        )
        aux_sample_tasks = aux.get("sample_tasks")
        if task_loss_weights and aux_sample_tasks is not None:
            all_tasks = list(aux_sample_tasks)
            active_ids = active_sample_ids.detach().cpu().tolist()
            if len(all_tasks) == len(aux.get("sample_lens") or ()):
                active_tasks = [all_tasks[int(index)] for index in active_ids]
            elif len(all_tasks) == len(active_ids):
                active_tasks = all_tasks
            else:
                active_tasks = [kind] * len(active_ids)
            weights = []
            for task in active_tasks:
                task_name = _normalize_delta_task(str(task))
                value = 1.0
                for pattern, weight in task_loss_weights.items():
                    if fnmatch.fnmatchcase(
                        task_name, _normalize_delta_task(str(pattern))
                    ):
                        value = float(weight)
                        break
                weights.append(value)
            task_sample_weights = per_sample_cost.new_tensor(weights)
            task_sample_weights = task_sample_weights / task_sample_weights.mean().clamp_min(1e-6)

        def weighted_sample_mean(values: torch.Tensor) -> torch.Tensor:
            if task_sample_weights is None:
                return values.mean()
            return (values * task_sample_weights.to(values)).mean()

        # The full-depth realization is the path used by inference when the
        # controller continues through the final candidate.  A preservation
        # teacher is detached by design, so it cannot improve that path when
        # the shared backbone is frozen.  Add an optional direct task loss to
        # train the lightweight compiler/realizer toward the ground-truth
        # objective while retaining the oracle action and preservation terms.
        quality_loss = weighted_sample_mean(per_sample_cost[:, -1])
        improvement_weight = (
            self.dynamic_depth_controller.config.fusion_sealing_improvement_weight
        )
        improvement_margin = (
            self.dynamic_depth_controller.config.fusion_sealing_improvement_margin
        )
        improvement_loss = quality_loss.new_zeros(())
        if (
            improvement_weight > 0.0
            and gate_final_realization
            and per_sample_baseline_cost is not None
        ):
            improvement_loss = weighted_sample_mean(
                F.relu(
                    per_sample_cost[:, -1]
                    - per_sample_baseline_cost[:, -1].detach()
                    + improvement_margin
                )
            )
        preservation_weight = (
            self.dynamic_depth_controller.config.fusion_sealing_preservation_weight
        )
        quality_weight = (
            self.dynamic_depth_controller.config.fusion_sealing_quality_weight
        )
        probabilities = torch.softmax(controller_logits, dim=-1)
        depths = per_sample_cost.new_tensor(candidate_depths)
        full_depth = depths[-1]
        expected_depth = (
            probabilities[..., 1] * depths
            + probabilities[..., 0] * full_depth
        ).mean() / len(self.language_model.model.layers)
        compute_loss = (
            self.dynamic_depth_controller.config.fusion_sealing_compute_weight
            * expected_depth
        )
        fusion_loss = (
            action_loss
            + compute_loss
            + preservation_weight * preservation_loss
            + quality_weight * quality_loss
            + improvement_weight * improvement_loss
        )
        output["fusion_sealing_loss"] = fusion_loss
        output["fusion_sealing_action_loss"] = action_loss.detach()
        output["fusion_sealing_compute_loss"] = compute_loss.detach()
        output["fusion_sealing_preservation_loss"] = preservation_loss.detach()
        output["fusion_sealing_quality_loss"] = quality_loss.detach()
        output["fusion_sealing_improvement_loss"] = improvement_loss.detach()
        output["fusion_sealing_seal_target_rate"] = target.float().mean().detach()
        output["fusion_sealing_expected_depth"] = (
            expected_depth.detach() * len(self.language_model.model.layers)
        )
        for index, depth in enumerate(candidate_depths):
            output[f"fusion_sealing_target_seal_depth_{depth}"] = (
                target[:, index].float().mean().detach()
            )
        return output

    def _add_router_training_losses(
        self,
        *,
        output: dict[str, Any],
        packed_label_ids: torch.Tensor | None,
    ) -> dict[str, Any]:
        aux = getattr(self.language_model.model, "_flexumm_aux", None)
        generation_aux = getattr(
            self.language_model.model,
            "_flexumm_generation_aux",
            None,
        )
        mse_target = output.pop("mse_target", None)
        output = self._add_fusion_sealing_training_loss(
            output=output,
            packed_label_ids=packed_label_ids,
            mse_target=mse_target,
        )
        if self.training and self.dynamic_depth_controller.config.tafe_enabled:
            qwen_model = self.language_model.model
            entropies = []
            action_probability_batches = []
            for gate_name in (
                "tafe_gate_understanding",
                "tafe_gate_generation",
            ):
                gate = getattr(qwen_model, gate_name, None)
                entropy = getattr(gate, "last_training_action_entropy", None)
                if entropy is not None:
                    entropies.append(entropy)
                probabilities = getattr(
                    gate, "last_training_action_probabilities", None
                )
                if probabilities is not None and probabilities.numel() > 0:
                    action_probability_batches.append(probabilities.float())
            tafe_terms = []
            if entropies:
                tafe_entropy = torch.stack(entropies).mean()
                output["tafe_entropy"] = tafe_entropy.detach()
                tafe_terms.append(
                    -self.dynamic_depth_controller.config.tafe_entropy_weight
                    * tafe_entropy
                )
            if action_probability_batches:
                probabilities = torch.cat(action_probability_batches, dim=0)
                config = self.dynamic_depth_controller.config
                target = probabilities.new_tensor(config.tafe_action_target)
                target = target / target.sum().clamp_min(1e-8)
                mean_probabilities = probabilities.mean(dim=0)
                balance_loss = (mean_probabilities - target).square().sum()
                # Minimize H(token action) - H(batch action).  This is the
                # negative mutual information between token features and the
                # selected FFN action, so it encourages confident but varied
                # token-level routing instead of one global action.
                token_entropy = -(
                    probabilities
                    * torch.log(probabilities.clamp_min(1e-8))
                ).sum(dim=-1).mean()
                batch_entropy = -(
                    mean_probabilities
                    * torch.log(mean_probabilities.clamp_min(1e-8))
                ).sum()
                diversity_loss = token_entropy - batch_entropy
                output["tafe_action_balance"] = balance_loss.detach()
                output["tafe_action_diversity"] = diversity_loss.detach()
                tafe_terms.extend(
                    [
                        config.tafe_action_balance_weight * balance_loss,
                        config.tafe_action_diversity_weight * diversity_loss,
                    ]
                )
            if tafe_terms:
                output["tafe_loss"] = torch.stack(tafe_terms).sum()
        if not self.training:
            return output
        if (
            not aux
            or packed_label_ids is None
            or not aux["exit_hidden_states"]
        ):
            return self._add_generation_router_training_losses(
                output=output,
                aux=generation_aux,
                mse_target=mse_target,
            )

        exit_ce = []
        exit_token_predictions = []
        for exit_hidden in aux["exit_hidden_states"]:
            logits = self.language_model.lm_head(exit_hidden)
            exit_token_predictions.append(logits.detach().argmax(dim=-1))
            exit_ce.append(
                torch.nn.functional.cross_entropy(
                    logits, packed_label_ids, reduction="none"
                )
            )
        per_depth_ce = torch.stack(exit_ce, dim=0)
        # The final candidate is the full-depth teacher.  Distilling its
        # normalized label-token representation gives an earlier exit a much
        # denser target than answer CE alone, while detach() prevents the
        # shortcut objective from pulling the teacher backward.
        if len(aux["exit_hidden_states"]) > 1:
            teacher_hidden = aux["exit_hidden_states"][-1].detach().float()
            exit_hidden_distill = torch.stack(
                [
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        student_hidden.float(),
                        teacher_hidden,
                        dim=-1,
                    ).mean()
                    for student_hidden in aux["exit_hidden_states"][:-1]
                ]
            ).mean()
        else:
            exit_hidden_distill = per_depth_ce.new_zeros(())
        label_sample_ids = aux["label_sample_ids"]
        active_sample_ids = aux["active_sample_ids"]
        if label_sample_ids is None or active_sample_ids is None:
            raise RuntimeError("Missing per-sample router metadata")
        num_active_samples = int(active_sample_ids.numel())
        per_sample_cost = torch.stack(
            [
                per_depth_ce[:, label_sample_ids == sample_index]
                .float()
                .mean(dim=1)
                for sample_index in range(num_active_samples)
            ],
            dim=0,
        )
        token_predictions = torch.stack(exit_token_predictions, dim=0)
        teacher_predictions = token_predictions[-1:].expand_as(
            token_predictions
        )
        token_agreement = token_predictions.eq(teacher_predictions).float()
        labels = packed_label_ids.unsqueeze(0).expand_as(token_predictions)
        token_harmful_regret = CandidateDepthRouter.harmful_teacher_regret(
            token_predictions,
            teacher_predictions,
            labels,
        )
        per_sample_agreement = torch.stack(
            [
                token_agreement[:, label_sample_ids == sample_index]
                .mean(dim=1)
                for sample_index in range(num_active_samples)
            ],
            dim=0,
        )
        per_sample_harmful_regret = torch.stack(
            [
                token_harmful_regret[:, label_sample_ids == sample_index]
                .mean(dim=1)
                for sample_index in range(num_active_samples)
            ],
            dim=0,
        )
        per_sample_answer_regret = CandidateDepthRouter.sample_sequence_risk(
            token_harmful_regret,
            label_sample_ids,
            num_active_samples,
        )
        per_sample_answer_agreement = (
            CandidateDepthRouter.sample_teacher_exact_match(
                token_predictions,
                teacher_predictions,
                label_sample_ids,
                num_active_samples,
            )
        )
        per_sample_label_exact_match = (
            CandidateDepthRouter.sample_label_exact_match(
                token_predictions,
                labels,
                label_sample_ids,
                num_active_samples,
            )
        )
        router_logits = torch.stack(aux["router_logits"], dim=-1)
        router_risk_means = torch.stack(aux["router_risk_means"], dim=-1)
        router_risk_scales = torch.stack(aux["router_risk_scales"], dim=-1)
        depth_config = self.dynamic_depth_controller.config
        probabilities = CandidateDepthRouter.halting_distribution(
            router_logits, temperature=depth_config.router_temperature
        )
        per_depth_cost = per_sample_cost.mean(dim=0)
        depths = probabilities.new_tensor(aux["candidate_depths"])
        expected_depth_per_sample = torch.sum(
            probabilities * depths, dim=-1
        )
        expected_depth = expected_depth_per_sample.mean()
        entropy = (
            -torch.sum(
                probabilities * torch.log(probabilities.clamp_min(1e-8)),
                dim=-1,
            )
        ).mean()
        acceptable_mask = None

        def merge_acceptable(mask: torch.Tensor) -> None:
            nonlocal acceptable_mask
            mask = mask.to(device=per_sample_cost.device, dtype=torch.bool)
            acceptable_mask = (
                mask if acceptable_mask is None else acceptable_mask & mask
            )

        if depth_config.router_teacher_agreement > 0:
            # Backward-compatible symmetric teacher matching.
            merge_acceptable(
                per_sample_agreement
                >= depth_config.router_teacher_agreement
            )
        if depth_config.router_answer_agreement > 0:
            merge_acceptable(
                per_sample_answer_agreement
                >= depth_config.router_answer_agreement
            )
        if depth_config.router_max_teacher_regret < 1.0:
            # Directional safety: allow a shallow candidate to correct a
            # full-depth mistake, but not to lose a correct teacher token.
            merge_acceptable(
                per_sample_harmful_regret
                <= depth_config.router_max_teacher_regret
            )
        if depth_config.router_max_answer_regret < 1.0:
            # Answer-level safety: one harmful answer-token change disqualifies
            # the exit, even if token-average regret is small.
            merge_acceptable(
                per_sample_answer_regret
                <= depth_config.router_max_answer_regret
            )
        if depth_config.router_min_exit_depth is not None:
            depth_eligible = (
                depths >= depth_config.router_min_exit_depth
            ).unsqueeze(0).expand_as(per_sample_cost)
            acceptable_mask = (
                depth_eligible
                if acceptable_mask is None
                else acceptable_mask & depth_eligible
            )
        excess_cost = per_sample_cost.detach() - per_sample_cost[:, -1:].detach()
        harmful_regret = per_sample_harmful_regret.detach()
        answer_regret = per_sample_answer_regret.detach()
        task_sample_weights = None
        aux_sample_tasks = aux.get("sample_tasks")
        if depth_config.router_task_loss_weights and aux_sample_tasks is not None:
            weights = []
            for task in aux_sample_tasks:
                normalized = CandidateDepthRouter.normalize_task(task)
                weights.append(
                    float(depth_config.router_task_loss_weights.get(normalized, 1.0))
                )
            task_sample_weights = per_sample_cost.new_tensor(weights)
        hard_sample_weights = None
        if (
            depth_config.router_hard_sample_weight > 0
            and excess_cost.shape[1] > 1
        ):
            hard_excess = excess_cost[:, :-1].clamp_min(0).max(dim=-1).values
            hard_regret = harmful_regret[:, :-1].max(dim=-1).values
            hard_answer_regret = answer_regret[:, :-1].max(dim=-1).values
            hardness = (
                hard_excess
                + hard_regret
                + depth_config.router_sequence_regret_weight
                * hard_answer_regret
            )
            scaled_hardness = torch.softmax(
                hardness.float()
                / depth_config.router_hard_sample_temperature,
                dim=0,
            ) * max(1, int(hardness.numel()))
            hard_sample_weights = (
                1.0
                + depth_config.router_hard_sample_weight
                * scaled_hardness.to(device=per_sample_cost.device)
            )
        sample_weights = hard_sample_weights
        if task_sample_weights is not None:
            sample_weights = (
                task_sample_weights
                if sample_weights is None
                else sample_weights * task_sample_weights
            )
            sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-6)
        expected_task_loss_per_sample = torch.sum(
            probabilities * per_sample_cost.detach(), dim=-1
        )
        if sample_weights is None:
            expected_task_loss = expected_task_loss_per_sample.mean()
        else:
            normalized_sample_weights = (
                sample_weights.float()
                / sample_weights.float().mean().clamp_min(1e-6)
            ).to(device=expected_task_loss_per_sample.device)
            expected_task_loss = (
                expected_task_loss_per_sample * normalized_sample_weights
            ).mean()
        supervised_router_loss, target_indexes = (
            CandidateDepthRouter.supervised_halting_loss(
                router_logits,
                per_sample_cost.detach(),
                quality_margin=depth_config.router_quality_margin,
                temperature=depth_config.router_temperature,
                acceptable_mask=acceptable_mask,
                sample_weights=sample_weights,
                target_strategy=depth_config.router_target_strategy,
                min_exit_gain=depth_config.router_min_exit_gain,
                positive_weight=depth_config.router_supervised_positive_weight,
                label_correct=per_sample_label_exact_match.detach(),
            )
        )
        exit_calibration_loss = CandidateDepthRouter.exit_confidence_loss(
            router_logits,
            per_sample_cost.detach(),
            quality_margin=depth_config.router_quality_margin,
            temperature=depth_config.router_temperature,
            acceptable_mask=acceptable_mask,
            sample_weights=sample_weights,
            target_indexes=target_indexes,
            positive_weight=depth_config.router_supervised_positive_weight,
        )
        risk_target = CandidateDepthRouter.risk_targets(
            per_sample_cost.detach(),
            strategy=depth_config.router_risk_target_strategy,
            quality_margin=depth_config.router_quality_margin,
            harmful_regret=harmful_regret,
            answer_regret=answer_regret,
            sequence_regret_weight=depth_config.router_sequence_regret_weight,
            target_indexes=target_indexes,
            acceptable_mask=acceptable_mask,
            label_correct=per_sample_label_exact_match.detach(),
        )
        quality_prediction_loss = CandidateDepthRouter.quality_prediction_loss(
            router_risk_means,
            router_risk_scales,
            risk_target,
            sample_weights=sample_weights,
        )
        router_loss = (
            depth_config.router_expected_loss_weight * expected_task_loss
            + depth_config.router_compute_weight
            * expected_depth
            / len(self.language_model.model.layers)
            - depth_config.router_entropy_weight * entropy
            + depth_config.router_supervised_weight * supervised_router_loss
            + depth_config.router_exit_calibration_weight
            * exit_calibration_loss
            + depth_config.router_quality_prediction_weight
            * quality_prediction_loss
        )

        output["multi_exit_ce"] = per_depth_ce.mean(dim=0)
        output["exit_hidden_distill"] = exit_hidden_distill
        output["router_loss"] = router_loss
        output["router_expected_task_loss"] = expected_task_loss.detach()
        output["router_supervised_loss"] = supervised_router_loss.detach()
        output["router_exit_calibration_loss"] = (
            exit_calibration_loss.detach()
        )
        output["router_quality_prediction_loss"] = (
            quality_prediction_loss.detach()
        )
        output["router_predicted_harmful_regret"] = (
            router_risk_means.mean().detach()
        )
        output["router_predicted_risk_scale"] = (
            router_risk_scales.mean().detach()
        )
        output["router_observed_excess_ce"] = excess_cost.mean().detach()
        output["router_observed_harmful_regret"] = (
            harmful_regret.mean().detach()
        )
        output["router_observed_answer_regret"] = (
            answer_regret.mean().detach()
        )
        output["router_observed_answer_agreement"] = (
            per_sample_answer_agreement.mean().detach()
        )
        output["router_observed_label_exact_match"] = (
            per_sample_label_exact_match.mean().detach()
        )
        if hard_sample_weights is not None:
            output["router_hard_sample_weight_mean"] = (
                hard_sample_weights.mean().detach()
            )
            output["router_hard_sample_weight_max"] = (
                hard_sample_weights.max().detach()
            )
        output["router_observed_teacher_agreement"] = (
            per_sample_agreement.mean().detach()
        )
        output["router_expected_depth"] = expected_depth.detach()
        output["router_target_depth"] = depths[target_indexes].float().mean().detach()
        output["router_entropy"] = entropy.detach()
        for depth, probability, cost in zip(
            aux["candidate_depths"], probabilities.mean(dim=0), per_depth_cost
        ):
            output[f"router_p_depth_{depth}"] = probability.detach()
            output[f"exit_ce_depth_{depth}"] = cost.detach()
        for candidate_index, depth in enumerate(aux["candidate_depths"]):
            output[f"router_target_rate_depth_{depth}"] = (
                target_indexes.eq(candidate_index).float().mean().detach()
            )
        for candidate_index, depth in enumerate(aux["candidate_depths"]):
            output[f"router_predicted_harmful_regret_depth_{depth}"] = (
                router_risk_means[:, candidate_index].mean().detach()
            )
            output[f"router_risk_scale_depth_{depth}"] = (
                router_risk_scales[:, candidate_index].mean().detach()
            )
            output[f"router_observed_excess_ce_depth_{depth}"] = (
                excess_cost[:, candidate_index].mean().detach()
            )
            output[f"router_observed_harmful_regret_depth_{depth}"] = (
                harmful_regret[:, candidate_index].mean().detach()
            )
            output[f"router_observed_answer_regret_depth_{depth}"] = (
                answer_regret[:, candidate_index].mean().detach()
            )
            output[f"router_observed_answer_agreement_depth_{depth}"] = (
                per_sample_answer_agreement[:, candidate_index]
                .mean()
                .detach()
            )
            output[f"router_observed_label_exact_match_depth_{depth}"] = (
                per_sample_label_exact_match[:, candidate_index]
                .mean()
                .detach()
            )
        mean_hazards = torch.sigmoid(
            router_logits / depth_config.router_temperature
        ).mean(dim=0)
        for depth, hazard in zip(aux["candidate_depths"][:-1], mean_hazards[:-1]):
            output[f"router_halt_depth_{depth}"] = hazard.detach()
        self._add_generation_router_training_losses(
            output=output,
            aux=generation_aux,
            mse_target=mse_target,
        )
        return output

    def _add_generation_router_training_losses(
        self,
        *,
        output: dict[str, Any],
        aux: dict[str, Any] | None,
        mse_target: torch.Tensor | None,
    ) -> dict[str, Any]:
        if not aux or mse_target is None or not aux["exit_hidden_states"]:
            return output

        label_sample_ids = aux["label_sample_ids"]
        active_sample_ids = aux["active_sample_ids"]
        if label_sample_ids is None or active_sample_ids is None:
            raise RuntimeError("Missing per-sample generation router metadata")
        num_active_samples = int(active_sample_ids.numel())
        target = mse_target.to(
            device=aux["exit_hidden_states"][0].device,
            dtype=aux["exit_hidden_states"][0].dtype,
        )
        exit_mse = []
        for exit_hidden in aux["exit_hidden_states"]:
            predictions = self.llm2vae(exit_hidden)
            if predictions.shape != target.shape:
                raise RuntimeError(
                    "Generation multi-exit predictions and flow targets do not "
                    f"align: {tuple(predictions.shape)} != {tuple(target.shape)}"
                )
            exit_mse.append((predictions - target).pow(2).mean(dim=-1))
        per_depth_token_mse = torch.stack(exit_mse, dim=0)
        per_sample_cost = torch.stack(
            [
                per_depth_token_mse[:, label_sample_ids == sample_index]
                .float()
                .mean(dim=1)
                for sample_index in range(num_active_samples)
            ],
            dim=0,
        )
        router_logits = torch.stack(aux["router_logits"], dim=-1)
        router_risk_means = torch.stack(aux["router_risk_means"], dim=-1)
        router_risk_scales = torch.stack(aux["router_risk_scales"], dim=-1)
        depth_config = self.dynamic_depth_controller.config
        probabilities = CandidateDepthRouter.halting_distribution(
            router_logits,
            temperature=depth_config.router_temperature,
        )
        depths = probabilities.new_tensor(aux["candidate_depths"])
        expected_depth_per_sample = torch.sum(probabilities * depths, dim=-1)
        expected_depth = expected_depth_per_sample.mean()
        entropy = (
            -torch.sum(
                probabilities * torch.log(probabilities.clamp_min(1e-8)),
                dim=-1,
            )
        ).mean()
        acceptable_mask = None
        if depth_config.router_min_exit_depth is not None:
            acceptable_mask = (
                depths >= depth_config.router_min_exit_depth
            ).unsqueeze(0).expand_as(per_sample_cost)

        sample_weights = None
        aux_sample_tasks = aux.get("sample_tasks")
        if depth_config.router_task_loss_weights and aux_sample_tasks is not None:
            weights = []
            for task in aux_sample_tasks:
                normalized = CandidateDepthRouter.normalize_task(task)
                weights.append(
                    float(depth_config.router_task_loss_weights.get(normalized, 1.0))
                )
            sample_weights = per_sample_cost.new_tensor(weights)
            sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-6)

        expected_task_loss_per_sample = torch.sum(
            probabilities * per_sample_cost.detach(),
            dim=-1,
        )
        if sample_weights is None:
            expected_task_loss = expected_task_loss_per_sample.mean()
        else:
            expected_task_loss = (
                expected_task_loss_per_sample * sample_weights.to(
                    device=expected_task_loss_per_sample.device
                )
            ).mean()

        target_strategy = depth_config.router_target_strategy
        if target_strategy == "label_correct":
            target_strategy = "best_label_ce"
        supervised_router_loss, target_indexes = (
            CandidateDepthRouter.supervised_halting_loss(
                router_logits,
                per_sample_cost.detach(),
                quality_margin=depth_config.router_quality_margin,
                temperature=depth_config.router_temperature,
                acceptable_mask=acceptable_mask,
                sample_weights=sample_weights,
                target_strategy=target_strategy,
                min_exit_gain=depth_config.router_min_exit_gain,
                positive_weight=depth_config.router_supervised_positive_weight,
            )
        )
        exit_calibration_loss = CandidateDepthRouter.exit_confidence_loss(
            router_logits,
            per_sample_cost.detach(),
            quality_margin=depth_config.router_quality_margin,
            temperature=depth_config.router_temperature,
            acceptable_mask=acceptable_mask,
            sample_weights=sample_weights,
            target_indexes=target_indexes,
            positive_weight=depth_config.router_supervised_positive_weight,
        )
        excess_mse = per_sample_cost.detach() - per_sample_cost[:, -1:].detach()
        normalized_excess_mse = (
            excess_mse.clamp_min(0)
            / (
                per_sample_cost[:, -1:].detach().abs()
                + depth_config.router_quality_margin
                + 1e-6
            )
        ).clamp(max=1.0)
        risk_strategy = depth_config.router_risk_target_strategy
        if risk_strategy == "label_error":
            risk_strategy = "best_label_ce"
        risk_target = CandidateDepthRouter.risk_targets(
            per_sample_cost.detach(),
            strategy=risk_strategy,
            quality_margin=depth_config.router_quality_margin,
            harmful_regret=normalized_excess_mse,
            answer_regret=normalized_excess_mse,
            sequence_regret_weight=0.0,
            target_indexes=target_indexes,
            acceptable_mask=acceptable_mask,
        )
        quality_prediction_loss = CandidateDepthRouter.quality_prediction_loss(
            router_risk_means,
            router_risk_scales,
            risk_target,
            sample_weights=sample_weights,
        )
        generation_router_loss = (
            depth_config.router_expected_loss_weight * expected_task_loss
            + depth_config.router_compute_weight
            * expected_depth
            / len(self.language_model.model.layers)
            - depth_config.router_entropy_weight * entropy
            + depth_config.router_supervised_weight * supervised_router_loss
            + depth_config.router_exit_calibration_weight
            * exit_calibration_loss
            + depth_config.router_quality_prediction_weight
            * quality_prediction_loss
        )
        if output.get("router_loss") is None:
            output["router_loss"] = generation_router_loss
        else:
            output["router_loss"] = output["router_loss"] + generation_router_loss
        output["generation_router_loss"] = generation_router_loss.detach()
        output["generation_router_expected_task_loss"] = (
            expected_task_loss.detach()
        )
        output["generation_router_supervised_loss"] = (
            supervised_router_loss.detach()
        )
        output["generation_router_exit_calibration_loss"] = (
            exit_calibration_loss.detach()
        )
        output["generation_router_quality_prediction_loss"] = (
            quality_prediction_loss.detach()
        )
        output["generation_router_observed_excess_mse"] = (
            excess_mse.mean().detach()
        )
        output["generation_router_expected_depth"] = expected_depth.detach()
        output["generation_router_target_depth"] = (
            depths[target_indexes].float().mean().detach()
        )
        output["generation_router_entropy"] = entropy.detach()
        for depth, probability, cost in zip(
            aux["candidate_depths"],
            probabilities.mean(dim=0),
            per_sample_cost.mean(dim=0),
        ):
            output[f"generation_router_p_depth_{depth}"] = probability.detach()
            output[f"exit_mse_depth_{depth}"] = cost.detach()
        mean_hazards = torch.sigmoid(
            router_logits / depth_config.router_temperature
        ).mean(dim=0)
        for depth, hazard in zip(
            aux["candidate_depths"][:-1],
            mean_hazards[:-1],
        ):
            output[f"generation_router_halt_depth_{depth}"] = hazard.detach()
        return output

    def _forward_flow(self, *args, **kwargs):
        bound = _BAGEL_FLOW_SIGNATURE.bind_partial(self, *args, **kwargs)
        timestep = bound.arguments.get("timestep")
        parent = current_depth_request()
        task = (
            parent.task
            if parent is not None and parent.task is not None
            else "generation"
        )
        override = parent.depth_override if parent is not None else None
        with self.dynamic_depth_controller.route(
            task=task,
            kind="generation",
            timestep=timestep,
            depth_override=override,
            has_understanding=False,
            has_generation=True,
        ):
            return super()._forward_flow(*args, **kwargs)

    @contextmanager
    def depth_route(
        self,
        task: str,
        *,
        kind: str | None = None,
        depth_sample_tasks: list[str] | tuple[str, ...] | None = None,
        depth_override: int | None = None,
        timestep: torch.Tensor | float | None = None,
    ) -> Iterator[None]:
        """Apply a named task route around an inference or training operation."""

        with self.dynamic_depth_controller.route(
            task=task,
            sample_tasks=depth_sample_tasks,
            kind=kind,
            timestep=timestep,
            depth_override=depth_override,
        ):
            yield

    def dynamic_depth_stats(self) -> list[dict[str, Any]]:
        return self.dynamic_depth_controller.stats()

    def router_diagnostics(self) -> dict[str, Any] | None:
        qwen_model = self.language_model.model
        last = (
            getattr(qwen_model, "last_prompt_router_decision", None)
            or getattr(qwen_model, "last_router_decision", None)
        )
        counts = getattr(qwen_model, "_router_depth_counts", None)
        token_counts = getattr(qwen_model, "_router_layer_token_counts", None)
        if last is None and not counts and not token_counts:
            return None
        histogram = [
            {"task": task, "depth": depth, "count": count}
            for (task, depth), count in sorted((counts or {}).items())
        ]
        total = sum(item["count"] for item in histogram)
        expected_depth = (
            sum(item["depth"] * item["count"] for item in histogram) / total
            if total
            else None
        )
        token_histogram = [
            {"task": task, "depth": depth, "tokens": count}
            for (task, depth), count in sorted((token_counts or {}).items())
        ]
        total_tokens = sum(item["tokens"] for item in token_histogram)
        mean_token_depth = (
            sum(
                item["depth"] * item["tokens"]
                for item in token_histogram
            )
            / total_tokens
            if total_tokens
            else None
        )
        trace_summary = []
        for (task, depth), summary in sorted(
            getattr(qwen_model, "_router_trace_stats", {}).items()
        ):
            count = int(summary["count"])
            trace_summary.append(
                {
                    "task": task,
                    "depth": depth,
                    "count": count,
                    "mean_halt_probability": (
                        float(summary["sum"]) / count if count else None
                    ),
                    "min_halt_probability": float(summary["min"]),
                    "max_halt_probability": float(summary["max"]),
                }
            )
        return {
            "last": last,
            "histogram": histogram,
            "num_routes": total,
            "mean_depth": expected_depth,
            "layer_token_histogram": token_histogram,
            "num_routed_tokens": total_tokens,
            "mean_token_depth": mean_token_depth,
            "trace_summary": trace_summary,
        }

    def fusion_sealing_diagnostics(self) -> dict[str, Any] | None:
        """Return the latest continue-versus-seal decision trace."""

        qwen_model = self.language_model.model
        decision = getattr(qwen_model, "last_fusion_sealing_decision", None)
        if decision is None:
            return None
        return dict(decision)

    def tafe_diagnostics(self) -> dict[str, Any] | None:
        """Return separate FFN and attention route summaries."""

        diagnostics: dict[str, Any] = {}
        qwen_model = self.language_model.model
        for namespace, label in (
            ("tafe_gate", "understanding"),
            ("tafe_gate", "generation"),
            ("tafe_attention_gate", "attention_understanding"),
            ("tafe_attention_gate", "attention_generation"),
        ):
            gate_kind = label.removeprefix("attention_")
            gate = getattr(qwen_model, f"{namespace}_{gate_kind}", None)
            result = getattr(gate, "last_result", None)
            if result is None:
                continue
            actions = result.actions.detach().flatten().to(dtype=torch.long)
            counts = torch.bincount(
                actions,
                minlength=len(TAFEAction),
            )
            diagnostics[label] = {
                "actions": {
                    action.name.lower(): int(counts[action].item())
                    for action in TAFEAction
                },
                "inference_actions": {
                    action.name.lower(): int(total)
                    for action, total in zip(
                        TAFEAction,
                        getattr(
                            gate,
                            "_inference_action_counts",
                            [0] * len(TAFEAction),
                        ),
                    )
                },
                "inference_calls": int(
                    getattr(gate, "_inference_calls", 0)
                ),
                "inference_routed_tokens": int(
                    getattr(gate, "_inference_routed_tokens", 0)
                ),
                "inference_subset_counts": [
                    int(value)
                    for value in getattr(
                        gate,
                        "_inference_subset_counts",
                        [],
                    )
                ],
                "mean_utilities": result.utilities.detach()
                .float()
                .mean(dim=0)
                .cpu()
                .tolist(),
                "last_all_exit": bool(
                    getattr(gate, "last_all_exit", False)
                ),
            }
        return diagnostics or None

    def set_flexumm_training_stage(
        self,
        stage: str,
        *,
        train_generation_base_ffn: bool = False,
    ) -> list[str]:
        """Set the staged FlexUMM trainable parameter policy.

        ``understanding`` trains the shared Transformer, the understanding
        route, and the next-token head. ``generation`` freezes those modules
        and trains the generation route plus the rectified-flow projections.
        ``joint`` enables every parameter as an explicit ablation.

        The returned list contains the parameter names left trainable and can
        be recorded alongside the training manifest.
        """

        normalized_stage = str(stage).strip().lower()
        if normalized_stage not in {"understanding", "generation", "joint"}:
            raise ValueError(
                "stage must be one of: understanding, generation, joint"
            )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        if normalized_stage == "joint":
            for parameter in self.parameters():
                parameter.requires_grad_(True)
        elif normalized_stage == "understanding":
            for name, parameter in self.named_parameters():
                if ".mlp_moe_gen." in name:
                    continue
                if (
                    "tafe_gate_generation." in name
                    or "tafe_attention_gate_generation." in name
                    or ".tafe_attention_generation." in name
                ):
                    continue
                if name.startswith(
                    (
                        "language_model.model.",
                        "language_model.lm_head.",
                    )
                ):
                    parameter.requires_grad_(True)
        else:
            generation_prefixes = (
                "time_embedder.",
                "vae2llm.",
                "llm2vae.",
                "latent_pos_embed.",
            )
            for name, parameter in self.named_parameters():
                generation_route = (
                    "tafe_gate_generation." in name
                    or ".mlp_moe_gen.specialized_banks." in name
                    or "tafe_attention_gate_generation." in name
                    or ".tafe_attention_generation.specialized_banks." in name
                    or (
                        train_generation_base_ffn
                        and ".mlp_moe_gen." in name
                    )
                )
                flow_head = name.startswith(generation_prefixes)
                if generation_route or flow_head:
                    parameter.requires_grad_(True)

        return [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def chat(
        self,
        *args,
        depth_task: str | None = None,
        depth_sample_tasks: list[str] | tuple[str, ...] | None = None,
        depth_override: int | None = None,
        **kwargs,
    ):
        task = (
            depth_task
            or getattr(self, "default_depth_task", None)
            or "understanding"
        )
        with self.depth_route(
            task,
            kind="understanding",
            depth_sample_tasks=depth_sample_tasks,
            depth_override=depth_override,
        ):
            return super().chat(*args, **kwargs)


def enable_dynamic_depth(
    bagel_model: _Bagel,
    config: DynamicDepthConfig | dict[str, Any] | None = None,
) -> DynamicBagel:
    """Enable dynamic depth in-place.

    Call this after constructing BAGEL and before starting a new inference
    context. Rule routing keeps the state dict unchanged. When
    ``router_enabled`` is true, a small registered router module is added and
    should be loaded from or saved into a post-training checkpoint.
    """

    if config is None:
        config = DynamicDepthConfig()
    elif isinstance(config, dict):
        config = DynamicDepthConfig.from_dict(config)
    elif not isinstance(config, DynamicDepthConfig):
        raise TypeError("config must be DynamicDepthConfig, dict, or None")

    qwen_model = bagel_model.language_model.model
    if not isinstance(qwen_model, _Qwen2Model):
        raise TypeError(
            "Expected bagel_model.language_model.model to be BAGEL Qwen2Model, "
            f"got {type(qwen_model)!r}"
        )

    if not isinstance(qwen_model, DynamicQwen2Model):
        qwen_model.__class__ = DynamicQwen2Model
    # Keep the LM head available for the evaluation-only prediction-entropy
    # policy without registering a second module path in the checkpoint.
    qwen_model._flexumm_lm_head_ref = weakref.ref(bagel_model.language_model.lm_head)
    qwen_model.configure_dynamic_depth(config)

    if not isinstance(bagel_model, DynamicBagel):
        bagel_model.__class__ = DynamicBagel
    bagel_model.dynamic_depth_controller = qwen_model.dynamic_depth_controller
    return bagel_model


def set_flexumm_training_stage(
    bagel_model: DynamicBagel,
    stage: str,
    *,
    train_generation_base_ffn: bool = False,
) -> list[str]:
    """Apply the staged FlexUMM trainability policy to a BAGEL model."""

    if not isinstance(bagel_model, DynamicBagel):
        raise TypeError(
            "set_flexumm_training_stage expects a DynamicBagel model"
        )
    return bagel_model.set_flexumm_training_stage(
        stage,
        train_generation_base_ffn=train_generation_base_ffn,
    )
