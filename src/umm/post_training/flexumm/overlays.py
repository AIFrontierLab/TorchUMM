"""Task-conditioned shared-weight overlays for a FlexUMM checkpoint bundle."""

from __future__ import annotations

from pathlib import Path
from typing import MutableMapping

import torch
from safetensors.torch import load_file


def apply_overlay_to_state_dict(
    state_dict: MutableMapping[str, torch.Tensor],
    overlay_path: str | Path,
) -> int:
    """Replace selected tensors in a CPU state dict with a validated overlay."""

    overlay_path = Path(overlay_path).expanduser().resolve()
    if not overlay_path.is_file():
        raise FileNotFoundError(overlay_path)
    overlay = load_file(overlay_path, device="cpu")
    for name, tensor in overlay.items():
        if name not in state_dict:
            raise KeyError(f"Overlay tensor is absent from model checkpoint: {name}")
        if state_dict[name].shape != tensor.shape:
            raise ValueError(
                f"Overlay shape mismatch for {name}: "
                f"{tuple(tensor.shape)} != {tuple(state_dict[name].shape)}"
            )
        state_dict[name] = tensor
    return len(overlay)


def apply_overlay_to_model(
    model: torch.nn.Module,
    overlay_path: str | Path,
) -> int:
    """Apply an overlay to an already dispatched model without retaining copies."""

    overlay_path = Path(overlay_path).expanduser().resolve()
    if not overlay_path.is_file():
        raise FileNotFoundError(overlay_path)
    overlay = load_file(overlay_path, device="cpu")
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    with torch.no_grad():
        for name, tensor in overlay.items():
            target = parameters.get(name)
            if target is None:
                target = buffers.get(name)
            if target is None:
                raise KeyError(f"Overlay tensor is absent from model: {name}")
            if target.shape != tensor.shape:
                raise ValueError(
                    f"Overlay shape mismatch for {name}: "
                    f"{tuple(tensor.shape)} != {tuple(target.shape)}"
                )
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
    return len(overlay)
