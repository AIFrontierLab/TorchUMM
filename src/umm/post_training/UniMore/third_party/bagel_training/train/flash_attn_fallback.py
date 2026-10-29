"""PyTorch fallback for BAGEL debug runs when flash-attn is unavailable."""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn.functional as F


_WARNED = False


def _warn_once() -> None:
    global _WARNED
    if not _WARNED:
        warnings.warn(
            "Using UniCross PyTorch SDPA fallback for flash_attn_varlen_func. "
            "Install flash-attn for production BAGEL training.",
            RuntimeWarning,
            stacklevel=2,
        )
        _WARNED = True


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    **_: Any,
) -> torch.Tensor:
    del max_seqlen_q, max_seqlen_k
    _warn_once()

    cu_q = cu_seqlens_q.to(device="cpu", dtype=torch.long).tolist()
    cu_k = cu_seqlens_k.to(device="cpu", dtype=torch.long).tolist()
    outputs: list[torch.Tensor] = []

    for idx in range(len(cu_q) - 1):
        q_i = q[cu_q[idx] : cu_q[idx + 1]]
        k_i = k[cu_k[idx] : cu_k[idx + 1]]
        v_i = v[cu_k[idx] : cu_k[idx + 1]]
        if q_i.numel() == 0:
            continue

        if q_i.shape[1] != k_i.shape[1]:
            if q_i.shape[1] % k_i.shape[1] != 0:
                raise ValueError(f"Cannot expand KV heads from {k_i.shape[1]} to {q_i.shape[1]}")
            repeat = q_i.shape[1] // k_i.shape[1]
            k_i = k_i.repeat_interleave(repeat, dim=1)
            v_i = v_i.repeat_interleave(repeat, dim=1)

        q_b = q_i.transpose(0, 1).unsqueeze(0)
        k_b = k_i.transpose(0, 1).unsqueeze(0)
        v_b = v_i.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q_b,
            k_b,
            v_b,
            attn_mask=None,
            dropout_p=float(dropout_p),
            is_causal=bool(causal),
            scale=softmax_scale,
        )
        outputs.append(out.squeeze(0).transpose(0, 1))

    if not outputs:
        return q.new_empty(q.shape)
    return torch.cat(outputs, dim=0).to(dtype=q.dtype)
