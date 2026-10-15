"""Import-path helpers for the BAGEL checkout carried by TorchUMM."""

from __future__ import annotations

import os
import sys
from pathlib import Path


_TORCHUMM_ROOT = Path(__file__).resolve().parents[4]


def _candidate_bagel_roots() -> tuple[Path, ...]:
    return (
        _TORCHUMM_ROOT / "src" / "umm" / "post_training" / "recA" / "BAGEL",
        _TORCHUMM_ROOT / "Bagel",
        _TORCHUMM_ROOT / "model" / "Bagel",
        _TORCHUMM_ROOT / "src" / "umm" / "backbones" / "bagel" / "Bagel",
        _TORCHUMM_ROOT / "src" / "umm" / "post_training" / "sft" / "bagel" / "Bagel",
    )


def ensure_bagel_importable(bagel_root: str | os.PathLike[str] | None = None) -> Path:
    """Put the local BAGEL root on ``sys.path`` and return the resolved path.

    BAGEL uses top-level imports such as ``modeling`` and ``data``. By default,
    those packages are resolved from the parent of this ``flexumm`` package.
    ``BAGEL_ROOT`` remains available for explicit development overrides.
    """

    configured_root = bagel_root or os.environ.get("BAGEL_ROOT")
    candidates = (
        (Path(configured_root).expanduser(),)
        if configured_root
        else _candidate_bagel_roots()
    )

    for candidate in candidates:
        root = candidate.resolve()
        if not (root / "modeling" / "bagel" / "qwen2_navit.py").is_file():
            continue
        root_string = str(root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        return root

    expected = "modeling/bagel/qwen2_navit.py"
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate BAGEL_ROOT with {expected}; searched {searched}")
