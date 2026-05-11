from __future__ import annotations

import sys
from pathlib import Path


def add_bagel_code_to_sys_path(repo_root: Path, *, require_data: bool = True) -> Path:
    """Add the best available BAGEL code root to sys.path.

    TorchUMM carries more than one BAGEL copy. The post-training copy under
    `recA/BAGEL` includes the original `data` package required by these LoRA
    scripts, while some inference-only copies do not.
    """

    candidates = [
        repo_root / "src" / "umm" / "post_training" / "recA" / "BAGEL",
        repo_root / "Bagel",
        repo_root / "model" / "Bagel",
        repo_root / "src" / "umm" / "backbones" / "bagel" / "Bagel",
        repo_root / "src" / "umm" / "post_training" / "sft" / "bagel" / "Bagel",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue
        if require_data and not (candidate / "data" / "data_utils.py").exists():
            continue
        if not (candidate / "modeling").exists():
            continue
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        return candidate

    expected = "data/data_utils.py and modeling/" if require_data else "modeling/"
    raise FileNotFoundError(f"Could not locate a BAGEL code root with {expected} under {repo_root}")
