"""
Tests for model bundle loader and validator.
"""

import json

import pytest
import yaml

from packages.inference.utils.model_bundle import (
    load_labels,
    load_thresholds,
    load_metadata,
    validate_bundle,
    ModelBundleError,
)


class TestLoadLabels:
    """Test label loading from JSON."""

    def test_load_valid_labels(self, tmp_path):
        labels_file = tmp_path / "labels.json"
        labels_file.write_text(json.dumps({
            "0": "blade_scratch",
            "1": "grinding_mark",
            "2": "surface_dent",
        }))
        
        result = load_labels(labels_file)
        assert result == {0: "blade_scratch", 1: "grinding_mark", 2: "surface_dent"}

    def test_missing_file_raises_error(self, tmp_path):
        with pytest.raises(ModelBundleError, match="Labels file not found"):
            load_labels(tmp_path / "missing.json")

    def test_invalid_json_raises_error(self, tmp_path):
        labels_file = tmp_path / "labels.json"
        labels_file.write_text("not valid json")
        
        with pytest.raises(ModelBundleError, match="Invalid JSON"):
            load_labels(labels_file)

    def test_invalid_class_id_raises_error(self, tmp_path):
        labels_file = tmp_path / "labels.json"
        labels_file.write_text(json.dumps({"not_a_number": "blade_scratch"}))
        
        with pytest.raises(ModelBundleError, match="Invalid class ID"):
            load_labels(labels_file)


class TestLoadThresholds:
    """Test threshold loading from YAML."""

    def test_load_valid_thresholds(self, tmp_path):
        thresholds_file = tmp_path / "thresholds.yaml"
        thresholds_file.write_text(yaml.dump({
            "thresholds": {
                "blade_scratch": 0.45,
                "grinding_mark": 0.30,
            }
        }))
        
        result = load_thresholds(thresholds_file)
        assert result == {"blade_scratch": 0.45, "grinding_mark": 0.30}

    def test_missing_file_raises_error(self, tmp_path):
        with pytest.raises(ModelBundleError, match="Thresholds file not found"):
            load_thresholds(tmp_path / "missing.yaml")

    def test_missing_thresholds_key_raises_error(self, tmp_path):
        thresholds_file = tmp_path / "thresholds.yaml"
        thresholds_file.write_text(yaml.dump({"other_key": {}}))
        
        with pytest.raises(ModelBundleError, match="must contain 'thresholds' key"):
            load_thresholds(thresholds_file)

    def test_invalid_threshold_value_raises_error(self, tmp_path):
        thresholds_file = tmp_path / "thresholds.yaml"
        thresholds_file.write_text(yaml.dump({
            "thresholds": {"blade_scratch": "not_a_number"}
        }))
        
        with pytest.raises(ModelBundleError, match="Invalid threshold"):
            load_thresholds(thresholds_file)


class TestLoadMetadata:
    """Test metadata loading from JSON."""

    def test_load_valid_metadata(self, tmp_path):
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps({
            "model_name": "kyoto-yolo26n-v1",
            "model_version": "1.0.0",
            "classes": 13,
        }))
        
        result = load_metadata(metadata_file)
        assert result["model_name"] == "kyoto-yolo26n-v1"
        assert result["model_version"] == "1.0.0"

    def test_missing_required_field_raises_error(self, tmp_path):
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps({"model_name": "test"}))  # missing version
        
        with pytest.raises(ModelBundleError, match="missing required field"):
            load_metadata(metadata_file)

    def test_missing_file_raises_error(self, tmp_path):
        with pytest.raises(ModelBundleError, match="Metadata file not found"):
            load_metadata(tmp_path / "missing.json")


class TestValidateBundle:
    """Test full bundle validation."""

    def create_valid_bundle(self, tmp_path):
        """Create a valid model bundle in tmp_path."""
        # Engine file (just touch it)
        engine = tmp_path / "model.engine"
        engine.touch()
        
        # Labels
        labels = tmp_path / "labels.json"
        labels.write_text(json.dumps({
            "0": "blade_scratch",
            "1": "grinding_mark",
        }))
        
        # Thresholds
        thresholds = tmp_path / "thresholds.yaml"
        thresholds.write_text(yaml.dump({
            "thresholds": {
                "blade_scratch": 0.45,
                "grinding_mark": 0.30,
            }
        }))
        
        # Metadata
        metadata = tmp_path / "metadata.json"
        metadata.write_text(json.dumps({
            "model_name": "kyoto-yolo26n-v1",
            "model_version": "1.0.0",
            "classes": 2,
        }))
        
        return engine, labels, thresholds, metadata

    def test_valid_bundle(self, tmp_path):
        engine, labels, thresholds, metadata = self.create_valid_bundle(tmp_path)
        
        result = validate_bundle(engine, labels, thresholds, metadata)
        
        assert result["labels"] == {0: "blade_scratch", 1: "grinding_mark"}
        assert result["thresholds"] == {"blade_scratch": 0.45, "grinding_mark": 0.30}
        assert result["metadata"]["model_name"] == "kyoto-yolo26n-v1"
        assert result["engine_path"] == engine

    def test_missing_engine_raises_error(self, tmp_path):
        _, labels, thresholds, metadata = self.create_valid_bundle(tmp_path)
        
        with pytest.raises(ModelBundleError, match="Engine file not found"):
            validate_bundle(tmp_path / "missing.engine", labels, thresholds, metadata)

    def test_missing_thresholds_for_class_raises_error(self, tmp_path):
        engine, labels, thresholds, metadata = self.create_valid_bundle(tmp_path)
        
        # Remove threshold for grinding_mark
        thresholds.write_text(yaml.dump({
            "thresholds": {"blade_scratch": 0.45}  # missing grinding_mark
        }))
        
        with pytest.raises(ModelBundleError, match="Classes missing thresholds"):
            validate_bundle(engine, labels, thresholds, metadata)

    def test_strict_taxonomy_validation(self, tmp_path):
        engine, labels, thresholds, metadata = self.create_valid_bundle(tmp_path)
        
        # Should pass with matching expected classes
        expected = ["blade_scratch", "grinding_mark"]
        result = validate_bundle(engine, labels, thresholds, metadata, expected)
        assert result is not None

    def test_strict_taxonomy_missing_class_raises_error(self, tmp_path):
        engine, labels, thresholds, metadata = self.create_valid_bundle(tmp_path)
        
        # Should fail if expected class is missing
        expected = ["blade_scratch", "grinding_mark", "missing_class"]
        with pytest.raises(ModelBundleError, match="Missing expected classes"):
            validate_bundle(engine, labels, thresholds, metadata, expected)


class TestBundleIntegration:
    """Integration tests for bundle loading in resolver context."""

    def test_bundle_paths_relative_to_model_dir(self, tmp_path):
        """Test that bundle paths are resolved relative to model_dir."""
        # This would be tested in the resolver integration tests
        pass
