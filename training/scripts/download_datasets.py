#!/usr/bin/env python3
"""Download training datasets from Roboflow (and optionally Kaggle).

Usage:
    python scripts/download_datasets.py --roboflow-key rf_xxxxx
    python scripts/download_datasets.py --roboflow-key rf_xxxxx --skip-kaggle
    python scripts/download_datasets.py --config config/dataset_config.yaml --roboflow-key rf_xxxxx
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make utils importable when running from training/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import DATASETS_DIR, CONFIG_DIR, ensure_dirs
from utils.yaml_io import load_yaml

log = logging.getLogger(__name__)

RAW_DIR = DATASETS_DIR / "raw"


def download_roboflow(cfg: dict, api_key: str) -> None:
    """Download all Roboflow projects listed in dataset_config.yaml."""
    try:
        from roboflow import Roboflow
    except ImportError:
        log.error("roboflow package not installed. Run: pip install roboflow")
        sys.exit(1)

    rf_cfg = cfg.get("roboflow", {})
    default_workspace = rf_cfg.get("workspace")
    projects = rf_cfg.get("projects", [])

    if not default_workspace:
        log.warning("No roboflow.workspace set in dataset_config.yaml — skipping Roboflow download.")
        return
    if not projects:
        log.warning("No roboflow.projects listed — skipping Roboflow download.")
        return

    rf = Roboflow(api_key=api_key)

    # Also download hard negative sources (treated as background, not defects)
    hard_neg_sources = cfg.get("hard_negative_sources", [])
    all_entries = [(p, "roboflow") for p in projects] + [(p, "negatives") for p in hard_neg_sources]

    for proj_entry, dest_subdir in all_entries:
        project_slug = proj_entry["project"]
        version_num = proj_entry.get("version", 1)
        fmt = proj_entry.get("format", "yolov8")
        workspace_slug = proj_entry.get("workspace", default_workspace)

        out_dir = RAW_DIR / dest_subdir / project_slug
        ensure_dirs(out_dir)

        log.info("Downloading %s/%s v%s (%s) → %s", workspace_slug, project_slug, version_num, fmt, out_dir)
        try:
            workspace = rf.workspace(workspace_slug)
            project = workspace.project(project_slug)
            version = project.version(version_num)
            version.download(fmt, location=str(out_dir), overwrite=False)
            log.info("  Done: %s", out_dir)
        except Exception as exc:  # noqa: BLE001
            log.error("  Failed to download %s/%s: %s", workspace_slug, project_slug, exc)


def check_kaggle_datasets(cfg: dict) -> None:
    """Print instructions for any Kaggle datasets that need manual download."""
    kaggle_entries = cfg.get("kaggle", {}).get("datasets", [])
    if not kaggle_entries:
        return

    kaggle_raw = RAW_DIR / "kaggle"
    missing = []
    for entry in kaggle_entries:
        zip_name = entry.get("expected_zip")
        if zip_name and not (kaggle_raw / zip_name).exists():
            missing.append((entry["name"], zip_name))

    if missing:
        log.warning("The following Kaggle datasets are not present and must be downloaded manually:")
        for name, zip_name in missing:
            log.warning("  %s → place %s in %s", name, zip_name, kaggle_raw)
        log.warning(
            "Download from Kaggle and place ZIPs in %s, then run merge_datasets.py", kaggle_raw
        )
    else:
        log.info("All expected Kaggle ZIPs are present in %s", kaggle_raw)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download IntelFactor training datasets")
    p.add_argument(
        "--config",
        default=str(CONFIG_DIR / "dataset_config.yaml"),
        help="Path to dataset_config.yaml",
    )
    p.add_argument(
        "--roboflow-key",
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    p.add_argument(
        "--skip-roboflow",
        action="store_true",
        help="Skip Roboflow download step",
    )
    p.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Skip Kaggle check step",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    cfg = load_yaml(args.config)
    ensure_dirs(RAW_DIR / "roboflow", RAW_DIR / "kaggle", RAW_DIR / "custom")

    if not args.skip_roboflow:
        if not args.roboflow_key:
            log.error(
                "No Roboflow API key provided. Use --roboflow-key or set ROBOFLOW_API_KEY."
            )
            # TODO: add your Roboflow API key
            sys.exit(1)
        download_roboflow(cfg, args.roboflow_key)

    if not args.skip_kaggle:
        check_kaggle_datasets(cfg)

    log.info("Download step complete. Run merge_datasets.py next.")


if __name__ == "__main__":
    main()
