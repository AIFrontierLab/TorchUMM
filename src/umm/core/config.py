from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")
_CONFIG_OVERRIDES: list[str] = []


def set_config_overrides(overrides: list[str] | None) -> None:
    """Set process-local config overrides in dotted-path form."""
    global _CONFIG_OVERRIDES
    _CONFIG_OVERRIDES = list(overrides or [])


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand ``${VAR}`` patterns in config string values."""
    if isinstance(obj, str):
        return _ENV_VAR_RE.sub(
            lambda m: os.environ.get(m.group(1), m.group(0)), obj
        )
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def _parse_override_value(value: str) -> Any:
    try:
        import yaml
    except ModuleNotFoundError:
        return value
    return yaml.safe_load(value)


def _apply_override(cfg: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Invalid config override `{override}`. Expected key=value.")
    key, raw_value = override.split("=", 1)
    parts = [part for part in key.strip().split(".") if part]
    if not parts:
        raise ValueError(f"Invalid config override `{override}`. Empty key.")

    cursor: dict[str, Any] = cfg
    for part in parts[:-1]:
        value = cursor.get(part)
        if value is None:
            value = {}
            cursor[part] = value
        if not isinstance(value, dict):
            raise ValueError(
                f"Cannot apply override `{override}` because `{part}` is not a mapping."
            )
        cursor = value
    cursor[parts[-1]] = _parse_override_value(raw_value)


def _apply_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    if not _CONFIG_OVERRIDES:
        return cfg
    for override in _CONFIG_OVERRIDES:
        _apply_override(cfg, override)
    return cfg


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. Install with `pip install pyyaml`."
            ) from exc
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return _apply_overrides(_expand_env_vars(data)) if isinstance(data, dict) else {}

    if suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return _apply_overrides(_expand_env_vars(data)) if isinstance(data, dict) else {}

    raise ValueError(f"Unsupported config format: {p}")
