"""UniMore paths and strict validation around TorchUMM's config loader."""

from __future__ import annotations

import os
from pathlib import Path
import re

from umm.core.config import load_config as load_umm_config

PACKAGE_ROOT = Path(__file__).resolve().parent
TORCHUMM_ROOT = PACKAGE_ROOT.parents[3]
_ENVIRONMENT_VARIABLE = re.compile(r"\$\{(\w+)\}")


def load_config(path):
    os.environ.setdefault("UNIMORE_ROOT", str(PACKAGE_ROOT))
    os.environ.setdefault("TORCHUMM_ROOT", str(TORCHUMM_ROOT))
    config = load_umm_config(Path(path).expanduser())

    def validate(value):
        if isinstance(value, str):
            missing = _ENVIRONMENT_VARIABLE.findall(value)
            if missing:
                raise ValueError(f"Set {', '.join(sorted(set(missing)))} before loading this configuration")
        elif isinstance(value, dict):
            for item in value.values():
                validate(item)
        elif isinstance(value, list):
            for item in value:
                validate(item)

    validate(config)
    return config
