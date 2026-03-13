"""YAML read/write helpers used across the training pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file and return as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict[str, Any], path: Path | str, *, comment: str | None = None) -> None:
    """Save a dict to a YAML file, optionally prepending a comment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        if comment:
            for line in comment.strip().splitlines():
                f.write(f"# {line}\n")
            f.write("\n")
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def merge_yaml(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dict into base dict (overrides win)."""
    result = dict(base)
    for k, v in overrides.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_yaml(result[k], v)
        else:
            result[k] = v
    return result
