"""IntelFactor defect taxonomy — single source of truth loader.

Loads class definitions from training/config/defect_taxonomy.yaml.
All scripts and tests import from here; nothing else hardcodes class names or IDs.

Public API (backward-compatible with previous hardcoded version):
  CLASSES           list[str]          — ordered canonical class names (index = YOLO class ID)
  BACKGROUND_CLASS  str                — 'background' sentinel
  NUM_CLASSES       int                — len(CLASSES)
  BILINGUAL         dict[str, dict]    — {name: {en, zh}}
  ALIASES           dict[str, str]     — source dataset name → canonical name
  resolve(name)     → str | None       — resolve any alias to canonical
  class_id(name)    → int
  class_info(name)  → ClassInfo
  yolo_names_block()→ str              — YAML block for data.yaml

Additional functions:
  load_taxonomy(path)        → raw dict from YAML
  get_class_names(path)      → list[str]
  get_severity_map(path)     → dict[str, str]
  get_confidence_thresholds(path) → dict[str, float]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Default path — resolved relative to this file so it works regardless of cwd
_DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "config" / "defect_taxonomy.yaml"

BACKGROUND_CLASS = "background"


# ---------------------------------------------------------------------------
# YAML loaders
# ---------------------------------------------------------------------------

def load_taxonomy(path: Path | str | None = None) -> dict[str, Any]:
    """Load and return the raw taxonomy YAML as a dict."""
    p = Path(path) if path else _DEFAULT_TAXONOMY_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"defect_taxonomy.yaml not found at {p}. "
            "Run from within the training/ directory or provide an explicit path."
        )
    with p.open() as f:
        return yaml.safe_load(f) or {}


def get_class_names(path: Path | str | None = None) -> list[str]:
    """Return ordered list of canonical class names (index = YOLO class ID)."""
    data = load_taxonomy(path)
    entries = sorted(data["classes"], key=lambda c: c["id"])
    return [c["name"] for c in entries]


def get_severity_map(path: Path | str | None = None) -> dict[str, str]:
    """Return {class_name: severity_label} — 'critical' | 'major' | 'minor'."""
    data = load_taxonomy(path)
    return {c["name"]: c["severity"] for c in data["classes"]}


def get_confidence_thresholds(path: Path | str | None = None) -> dict[str, float]:
    """Return {class_name: default_confidence_threshold}."""
    data = load_taxonomy(path)
    return {c["name"]: float(c["confidence_threshold"]) for c in data["classes"]}


def get_bilingual(path: Path | str | None = None) -> dict[str, dict[str, str]]:
    """Return {class_name: {en, zh}} bilingual display names."""
    data = load_taxonomy(path)
    result = {}
    for c in data["classes"]:
        result[c["name"]] = c["bilingual"]
    result[BACKGROUND_CLASS] = {"en": "Background", "zh": "背景"}
    return result


# ---------------------------------------------------------------------------
# Module-level constants (loaded once at import time)
# ---------------------------------------------------------------------------

CLASSES: list[str] = get_class_names()
NUM_CLASSES: int = len(CLASSES)
BILINGUAL: dict[str, dict[str, str]] = get_bilingual()

# ---------------------------------------------------------------------------
# Source-dataset alias maps
# Maps raw dataset class names → IntelFactor canonical names.
# Update this when adding new dataset sources; do NOT put aliases in the YAML
# (aliases are a training-pipeline concern, not a QC spec concern).
# ---------------------------------------------------------------------------
ALIASES: dict[str, str] = {
    # neu-det / NEU Surface Defect Database
    "scratches":           "blade_scratch",
    "scratch":             "blade_scratch",
    "pitted_surface":      "surface_dent",
    "pitted":              "surface_dent",
    "crazing":             "surface_crack",
    "inclusion":           "inclusion",
    "patches":             "surface_discolor",
    "rolled-in_scale":     "surface_discolor",
    "rolled_in_scale":     "surface_discolor",
    "pittedsurface":       "surface_dent",
    # GC10-DET
    "punching_hole":       "surface_dent",
    "welding_line":        "weld_defect",
    "crescent_gap":        "surface_crack",
    "water_spot":          "grinding_mark",
    "oil_spot":            "surface_discolor",
    "silk_spot":           "surface_discolor",
    "rolled_pit":          "surface_dent",
    "crease":              "surface_crack",
    "waist_folding":       "surface_crack",
    "gscratch":            "blade_scratch",
    "g_scratch":           "blade_scratch",
    # Severstal
    "class1":              "surface_discolor",
    "class2":              "surface_crack",
    "class3":              "surface_dent",
    "class4":              "inclusion",
    # Old IntelFactor class names (station.yaml v1, wiko_taxonomy.yaml v1)
    "scratch_surface":     "blade_scratch",
    "scratch_edge":        "blade_scratch",
    "scratch_deep":        "blade_scratch",
    "chip_edge":           "edge_crack",
    "chip_blade":          "blade_scratch",
    "grind_mark":          "grinding_mark",
    "polish_defect":       "grinding_mark",
    "rust_spot":           "surface_discolor",
    "misalignment":        "bolster_gap",
    "stamp_error":         "etching_defect",
    "handle_gap":          "bolster_gap",
    "handle_crack":        "handle_defect",
    "pit_corrosion":       "surface_discolor",
    "discoloration":       "surface_discolor",
    "crack":               "surface_crack",
    "warp":                "overgrind",
    "logo_defect":         "etching_defect",
    "dimension_out_of_spec": "overgrind",
    "foreign_material":    "inclusion",
    # Hard-negative synonyms — always → background
    "no_defect":           "background",
    "negative":            "background",
    "ok":                  "background",
    "good":                "background",
    "normal":              "background",
    "hammer_finish":       "background",
    "satin_finish":        "background",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassInfo:
    id: int
    name: str
    en: str
    zh: str
    severity: str
    confidence_threshold: float


def class_info(name: str) -> ClassInfo:
    """Return ClassInfo for a canonical class name."""
    if name not in BILINGUAL:
        raise KeyError(f"Unknown class: {name!r}. Add it to defect_taxonomy.yaml.")
    idx = CLASSES.index(name) if name in CLASSES else -1
    thresholds = get_confidence_thresholds()
    severities = get_severity_map()
    return ClassInfo(
        id=idx,
        name=name,
        en=BILINGUAL[name]["en"],
        zh=BILINGUAL[name]["zh"],
        severity=severities.get(name, "major"),
        confidence_threshold=thresholds.get(name, 0.5),
    )


def resolve(name: str) -> str | None:
    """Resolve a raw dataset class name to a canonical IntelFactor name.

    Returns None if the name is unknown and should be skipped.
    """
    name = name.strip().lower().replace(" ", "_").replace("-", "_")
    if name in CLASSES:
        return name
    if name == BACKGROUND_CLASS:
        return BACKGROUND_CLASS
    return ALIASES.get(name)


def class_id(name: str) -> int:
    """Return the 0-based YOLO class ID for a canonical class name."""
    return CLASSES.index(name)


def yolo_names_block() -> str:
    """Return the 'names:' YAML block for a data.yaml file."""
    lines = ["names:"]
    for i, cls in enumerate(CLASSES):
        lines.append(f"  {i}: {cls}")
    return "\n".join(lines)
