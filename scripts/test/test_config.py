"""Exercise configuration failure modes before expensive spatial processing."""

from pathlib import Path

import pytest
import yaml

from ecologyhydro.config import ConfigError, load_config, project_root


@pytest.fixture
def config_data():
    return yaml.safe_load((project_root() / "config.yaml").read_text(encoding="utf-8"))


def write_config(root: Path, data: dict) -> None:
    (root / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def test_paths_are_relative_to_project_not_cwd(tmp_path, monkeypatch, config_data):
    root = tmp_path / "checkout"
    root.mkdir()
    write_config(root, config_data)
    monkeypatch.chdir(tmp_path)
    config = load_config(root=root)
    assert config.paths.hydrology == root / "data" / "Hydrology"
    assert config.paths.outputs == root / "experiments"
    assert not config.paths.outputs.exists()


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("resources", "max_workers", 0),
        ("resources", "max_workers", True),
        ("resources", "timeout_hours", 13),
        ("resources", "memory_budget_gb", 32),
        ("resources", "memory_budget_gb", float("nan")),
        ("resources", "worker_typo", 2),
        ("study", "resolution_m", -1),
        ("study", "station_ids", ["A", "A"]),
    ],
)
def test_invalid_config_fails_early(tmp_path, config_data, section, key, value):
    config_data[section][key] = value
    write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(root=tmp_path)


def test_calibration_validation_overlap_is_rejected(tmp_path, config_data):
    config_data["study"].update(calibration_years=[2000, 2001], validation_years=[2001])
    write_config(tmp_path, config_data)
    with pytest.raises(ConfigError, match="must not overlap"):
        load_config(root=tmp_path)


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    (tmp_path / "config.yaml").write_text("schema_version: 1\nschema_version: 1\n")
    with pytest.raises(ConfigError, match="Duplicate YAML key"):
        load_config(root=tmp_path)


def test_template_loads_but_cannot_start_model():
    config = load_config()
    with pytest.raises(ConfigError, match="inputs.precipitation: missing file"):
        config.validate_inputs()


def test_existing_inputs_are_resolved_and_validated(tmp_path, config_data):
    source = tmp_path / "fixture.dat"
    source.write_text("File existence fixture; not a spatial validity test.")
    for key in config_data["inputs"]:
        if key != "versions":
            config_data["inputs"][key] = source.name
    config_data["model"]["z_parameter"] = 5
    config_data["study"].update(
        station_ids=["fixture"], projected_crs="EPSG:32648", resolution_m=100
    )
    write_config(tmp_path, config_data)
    load_config(root=tmp_path).validate_inputs()
    source.unlink()
    with pytest.raises(ConfigError, match="missing file"):
        load_config(root=tmp_path).validate_inputs()
