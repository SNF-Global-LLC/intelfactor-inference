"""Tests for the unified defect taxonomy and its consumers.

Validates:
  0. CORE INVARIANT: wiko_taxonomy.yaml defects key order == station.yaml defect_classes
     (this is the contract the runtime depends on — checked without any other fixture)
  1. defect_taxonomy.yaml schema — 13 classes, unique IDs 0-12, required fields
  2. training/utils/taxonomy.py loads correctly and exports expected API
  3. configs/station.yaml defect_classes — 13 entries, exact order, no duplicates
  4. configs/wiko_taxonomy.yaml defect keys — 13 entries, canonical names
  5. resolver.py VISION_MODELS entries include defect_classes with canonical names
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Resolve repo root regardless of test runner cwd
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_YAML = REPO_ROOT / "training" / "config" / "defect_taxonomy.yaml"
STATION_YAML = REPO_ROOT / "configs" / "station.yaml"
WIKO_YAML = REPO_ROOT / "configs" / "wiko_taxonomy.yaml"

# Make training/ importable
sys.path.insert(0, str(REPO_ROOT / "training"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def taxonomy_data() -> dict:
    assert TAXONOMY_YAML.exists(), f"defect_taxonomy.yaml missing: {TAXONOMY_YAML}"
    with TAXONOMY_YAML.open() as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def canonical_names(taxonomy_data) -> list[str]:
    return [c["name"] for c in sorted(taxonomy_data["classes"], key=lambda c: c["id"])]


@pytest.fixture(scope="module")
def station_data() -> dict:
    assert STATION_YAML.exists(), f"station.yaml missing: {STATION_YAML}"
    with STATION_YAML.open() as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def wiko_data() -> dict:
    assert WIKO_YAML.exists(), f"wiko_taxonomy.yaml missing: {WIKO_YAML}"
    with WIKO_YAML.open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 0. CORE INVARIANT — wiko_taxonomy.yaml ↔ station.yaml
# These tests are the primary contract. They use only the two runtime config
# files and no other fixture — if these fail, nothing else matters.
# ---------------------------------------------------------------------------

class TestCoreInvariant:
    """wiko_taxonomy.yaml defects key order == station.yaml defect_classes, always."""

    def _wiko_keys(self) -> list[str]:
        with WIKO_YAML.open() as f:
            data = yaml.safe_load(f)
        return list(data["defects"].keys())

    def _station_classes(self) -> list[str]:
        with STATION_YAML.open() as f:
            data = yaml.safe_load(f)
        return list(data["defect_classes"])

    def test_wiko_has_exactly_13_classes(self):
        keys = self._wiko_keys()
        assert len(keys) == 13, (
            f"wiko_taxonomy.yaml must have exactly 13 defect classes, got {len(keys)}: {keys}"
        )

    def test_station_has_exactly_13_classes(self):
        classes = self._station_classes()
        assert len(classes) == 13, (
            f"station.yaml must have exactly 13 defect_classes, got {len(classes)}: {classes}"
        )

    def test_no_duplicates_in_wiko(self):
        keys = self._wiko_keys()
        assert len(keys) == len(set(keys)), (
            f"Duplicate keys in wiko_taxonomy.yaml defects: "
            f"{[k for k in keys if keys.count(k) > 1]}"
        )

    def test_no_duplicates_in_station(self):
        classes = self._station_classes()
        assert len(classes) == len(set(classes)), (
            f"Duplicate entries in station.yaml defect_classes: "
            f"{[c for c in classes if classes.count(c) > 1]}"
        )

    def test_station_classes_exact_match_wiko_key_order(self):
        """THE KEY INVARIANT.

        station.yaml defect_classes must be identical to wiko_taxonomy.yaml
        defects keys in the same order. Any difference means class IDs are
        misaligned between config and model.

        Post-audit check equivalent:
            diff <(python3 -c "import yaml; print('\\n'.join(
                yaml.safe_load(open('configs/wiko_taxonomy.yaml'))['defects'].keys()))")
                 <(python3 -c "import yaml; print('\\n'.join(
                yaml.safe_load(open('configs/station.yaml'))['defect_classes']))")
        Should produce zero output.
        """
        wiko_keys = self._wiko_keys()
        station_classes = self._station_classes()
        assert station_classes == wiko_keys, (
            "station.yaml defect_classes does not match wiko_taxonomy.yaml defects key order.\n"
            f"wiko_taxonomy.yaml: {wiko_keys}\n"
            f"station.yaml:       {station_classes}\n"
            "These must be identical — key order defines YOLO class IDs."
        )

    def test_wiko_keys_are_canonical_snake_case(self):
        for key in self._wiko_keys():
            assert key == key.lower().replace(" ", "_"), (
                f"wiko_taxonomy.yaml key {key!r} is not canonical snake_case"
            )


# ---------------------------------------------------------------------------
# 1. defect_taxonomy.yaml schema
# ---------------------------------------------------------------------------

class TestDefectTaxonomyYAML:

    def test_file_exists(self):
        assert TAXONOMY_YAML.exists()

    def test_has_classes_key(self, taxonomy_data):
        assert "classes" in taxonomy_data, "defect_taxonomy.yaml must have a 'classes' list"

    def test_exactly_13_classes(self, taxonomy_data):
        assert len(taxonomy_data["classes"]) == 13, (
            f"Expected 13 classes, got {len(taxonomy_data['classes'])}"
        )

    def test_class_ids_are_0_through_12(self, taxonomy_data):
        ids = sorted(c["id"] for c in taxonomy_data["classes"])
        assert ids == list(range(13)), f"Class IDs must be 0-12 in sequence, got: {ids}"

    def test_class_ids_are_unique(self, taxonomy_data):
        ids = [c["id"] for c in taxonomy_data["classes"]]
        assert len(ids) == len(set(ids)), "Duplicate class IDs found"

    def test_class_names_are_unique(self, taxonomy_data):
        names = [c["name"] for c in taxonomy_data["classes"]]
        assert len(names) == len(set(names)), "Duplicate class names found"

    def test_required_fields_present(self, taxonomy_data):
        required = {"id", "name", "kyoto_ref", "aql", "severity", "confidence_threshold", "bilingual"}
        for cls in taxonomy_data["classes"]:
            missing = required - set(cls.keys())
            assert not missing, f"Class {cls.get('name', '?')} missing fields: {missing}"

    def test_severity_values(self, taxonomy_data):
        valid = {"critical", "major", "minor"}
        for cls in taxonomy_data["classes"]:
            assert cls["severity"] in valid, (
                f"Class {cls['name']} has invalid severity: {cls['severity']!r}"
            )

    def test_confidence_thresholds_in_range(self, taxonomy_data):
        for cls in taxonomy_data["classes"]:
            t = cls["confidence_threshold"]
            assert 0.0 < t < 1.0, (
                f"Class {cls['name']} confidence_threshold {t} must be in (0, 1)"
            )

    def test_critical_classes_have_lower_thresholds_than_minor(self, taxonomy_data):
        """Critical classes must have lower or equal thresholds than minor classes."""
        critical_thresholds = [
            c["confidence_threshold"] for c in taxonomy_data["classes"]
            if c["severity"] == "critical"
        ]
        minor_thresholds = [
            c["confidence_threshold"] for c in taxonomy_data["classes"]
            if c["severity"] == "minor"
        ]
        if critical_thresholds and minor_thresholds:
            assert max(critical_thresholds) <= min(minor_thresholds), (
                "Critical class thresholds should be ≤ minor class thresholds. "
                f"Max critical: {max(critical_thresholds)}, min minor: {min(minor_thresholds)}"
            )

    def test_bilingual_has_en_and_zh(self, taxonomy_data):
        for cls in taxonomy_data["classes"]:
            bil = cls["bilingual"]
            assert "en" in bil and "zh" in bil, (
                f"Class {cls['name']} bilingual must have 'en' and 'zh'"
            )

    def test_hard_negatives_present(self, taxonomy_data):
        assert "hard_negatives" in taxonomy_data, "defect_taxonomy.yaml must list hard_negatives"
        assert len(taxonomy_data["hard_negatives"]) >= 3, (
            "At least 3 hard negative examples required"
        )

    def test_canonical_class_names(self, canonical_names):
        expected = [
            "blade_scratch", "grinding_mark", "surface_dent", "surface_crack",
            "weld_defect", "edge_burr", "edge_crack", "handle_defect",
            "bolster_gap", "etching_defect", "inclusion", "surface_discolor", "overgrind",
        ]
        assert canonical_names == expected, (
            f"Canonical class names don't match expected.\n"
            f"Expected: {expected}\n"
            f"Got:      {canonical_names}"
        )


# ---------------------------------------------------------------------------
# 2. training/utils/taxonomy.py
# ---------------------------------------------------------------------------

class TestTaxonomyModule:

    def test_import(self):
        import utils.taxonomy  # noqa: F401

    def test_classes_is_13(self):
        from utils.taxonomy import CLASSES
        assert len(CLASSES) == 13

    def test_num_classes(self):
        from utils.taxonomy import NUM_CLASSES, CLASSES
        assert NUM_CLASSES == len(CLASSES)

    def test_classes_match_yaml(self, canonical_names):
        from utils.taxonomy import CLASSES
        assert CLASSES == canonical_names

    def test_resolve_canonical_name(self):
        from utils.taxonomy import resolve
        assert resolve("blade_scratch") == "blade_scratch"

    def test_resolve_alias(self):
        from utils.taxonomy import resolve
        assert resolve("scratches") == "blade_scratch"
        assert resolve("crazing") == "surface_crack"
        assert resolve("welding_line") == "weld_defect"

    def test_resolve_old_class_names(self):
        """Old station.yaml v1 names must resolve to canonical equivalents."""
        from utils.taxonomy import resolve
        assert resolve("scratch_surface") == "blade_scratch"
        assert resolve("chip_edge") == "edge_crack"
        assert resolve("rust_spot") == "surface_discolor"
        assert resolve("handle_gap") == "bolster_gap"
        assert resolve("logo_defect") == "etching_defect"
        assert resolve("warp") == "overgrind"

    def test_resolve_background(self):
        from utils.taxonomy import resolve
        assert resolve("no_defect") == "background"
        assert resolve("ok") == "background"

    def test_resolve_unknown_returns_none(self):
        from utils.taxonomy import resolve
        assert resolve("totally_unknown_class_xyz") is None

    def test_class_id(self):
        from utils.taxonomy import class_id
        assert class_id("blade_scratch") == 0
        assert class_id("overgrind") == 12

    def test_yolo_names_block(self):
        from utils.taxonomy import yolo_names_block
        block = yolo_names_block()
        assert "names:" in block
        assert "0: blade_scratch" in block
        assert "12: overgrind" in block

    def test_get_severity_map(self):
        from utils.taxonomy import get_severity_map
        sev = get_severity_map()
        assert sev["surface_crack"] == "critical"
        assert sev["edge_burr"] == "critical"
        assert sev["surface_discolor"] == "minor"
        assert sev["blade_scratch"] == "major"

    def test_get_confidence_thresholds(self):
        from utils.taxonomy import get_confidence_thresholds
        thresholds = get_confidence_thresholds()
        # Critical classes should have lower thresholds
        assert thresholds["edge_crack"] < thresholds["blade_scratch"]
        assert thresholds["surface_crack"] < thresholds["grinding_mark"]


# ---------------------------------------------------------------------------
# 3. configs/station.yaml
# ---------------------------------------------------------------------------

class TestStationYAML:

    def test_has_defect_classes(self, station_data):
        assert "defect_classes" in station_data

    def test_exactly_13_defect_classes(self, station_data):
        classes = station_data["defect_classes"]
        assert len(classes) == 13, (
            f"station.yaml must have exactly 13 defect_classes, got {len(classes)}"
        )

    def test_all_defect_classes_are_canonical(self, station_data, canonical_names):
        for cls in station_data["defect_classes"]:
            assert cls in canonical_names, (
                f"station.yaml defect_classes contains non-canonical name: {cls!r}. "
                f"Valid names: {canonical_names}"
            )

    def test_defect_classes_order_matches_taxonomy(self, station_data, canonical_names):
        """Class order in station.yaml must match taxonomy ID order (= YOLO class ID)."""
        assert station_data["defect_classes"] == canonical_names, (
            "station.yaml defect_classes order must match taxonomy ID order.\n"
            f"Expected: {canonical_names}\n"
            f"Got:      {station_data['defect_classes']}"
        )

    def test_sop_map_defect_sops_use_canonical_names(self, station_data, canonical_names):
        sop_map = station_data.get("sop_map", {})
        defect_sops = sop_map.get("defect_sops", {})
        for cls in defect_sops:
            assert cls in canonical_names, (
                f"station.yaml sop_map.defect_sops contains non-canonical name: {cls!r}"
            )


# ---------------------------------------------------------------------------
# 4. configs/wiko_taxonomy.yaml
# ---------------------------------------------------------------------------

class TestWikoTaxonomyYAML:

    def test_has_defects_key(self, wiko_data):
        assert "defects" in wiko_data

    def test_exactly_13_defect_entries(self, wiko_data):
        assert len(wiko_data["defects"]) == 13, (
            f"wiko_taxonomy.yaml must have exactly 13 defect entries, "
            f"got {len(wiko_data['defects'])}"
        )

    def test_all_defect_keys_are_canonical(self, wiko_data, canonical_names):
        for key in wiko_data["defects"]:
            assert key in canonical_names, (
                f"wiko_taxonomy.yaml defects contains non-canonical key: {key!r}. "
                f"Valid names: {canonical_names}"
            )

    def test_all_canonical_names_present(self, wiko_data, canonical_names):
        for name in canonical_names:
            assert name in wiko_data["defects"], (
                f"Canonical class {name!r} missing from wiko_taxonomy.yaml defects"
            )

    def test_each_defect_has_required_fields(self, wiko_data):
        required = {"description_en", "description_zh", "common_causes", "severity_range", "sop_section"}
        for name, entry in wiko_data["defects"].items():
            missing = required - set(entry.keys())
            assert not missing, f"wiko_taxonomy.yaml defect {name!r} missing fields: {missing}"


# ---------------------------------------------------------------------------
# 5. resolver.py VISION_MODELS
# ---------------------------------------------------------------------------

class TestResolverVisionModels:

    def test_vision_models_have_defect_classes(self):
        from packages.inference.providers.resolver import VISION_MODELS
        for model_key, model_def in VISION_MODELS.items():
            assert "defect_classes" in model_def, (
                f"VISION_MODELS[{model_key!r}] missing 'defect_classes' key"
            )

    def test_vision_models_defect_classes_are_canonical(self, canonical_names):
        from packages.inference.providers.resolver import VISION_MODELS
        for model_key, model_def in VISION_MODELS.items():
            for cls in model_def.get("defect_classes", []):
                assert cls in canonical_names, (
                    f"VISION_MODELS[{model_key!r}] defect_classes contains "
                    f"non-canonical name: {cls!r}"
                )

    def test_vision_models_have_13_defect_classes(self, canonical_names):
        from packages.inference.providers.resolver import VISION_MODELS
        for model_key, model_def in VISION_MODELS.items():
            classes = model_def.get("defect_classes", [])
            assert len(classes) == 13, (
                f"VISION_MODELS[{model_key!r}] has {len(classes)} defect_classes, expected 13"
            )
