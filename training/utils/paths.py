"""Repo-relative path helpers for the training pipeline.

All paths derive from TRAINING_ROOT so the pipeline works regardless of
where the repo is cloned or which VM/machine is running it.
"""
from __future__ import annotations

from pathlib import Path

# Absolute path to training/ — everything else is relative to this.
TRAINING_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = TRAINING_ROOT / "config"
DATASETS_DIR = TRAINING_ROOT / "datasets"
COMBINED_DIR = DATASETS_DIR / "combined"
SCRIPTS_DIR = TRAINING_ROOT / "scripts"
RUNS_DIR = TRAINING_ROOT / "runs"
EXPORTS_DIR = TRAINING_ROOT / "exports"
LOGS_DIR = TRAINING_ROOT / "logs"


def ensure_dirs(*dirs: Path) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def combined_split(split: str) -> Path:
    """Return path to a split directory inside combined dataset.

    Args:
        split: 'train', 'val', or 'test'
    """
    return COMBINED_DIR / split


def run_dir(name: str) -> Path:
    """Return path to a named training run output directory."""
    return RUNS_DIR / name


def export_path(filename: str) -> Path:
    """Return full path for an export artifact."""
    return EXPORTS_DIR / filename
