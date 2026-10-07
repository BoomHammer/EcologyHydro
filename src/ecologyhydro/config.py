"""Validated project configuration without starting a simulation."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Year = Annotated[int, Field(strict=True, ge=1, le=9999)]


class ConfigError(ValueError):
    """Invalid configuration or unavailable model input."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Paths(StrictModel):
    climate: Path
    dem: Path
    lulc: Path
    soil: Path
    hydrology: Path
    cache: Path
    outputs: Path
    logs: Path


class Study(StrictModel):
    station_ids: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)
    calibration_years: list[Year] = Field(default_factory=list)
    validation_years: list[Year] = Field(default_factory=list)
    projected_crs: str | None = None
    resolution_m: Annotated[float, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def check_partitions(self):
        for name in ("station_ids", "calibration_years", "validation_years"):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} contains duplicates")
        if set(self.calibration_years) & set(self.validation_years):
            raise ValueError("Calibration and validation years must not overlap")
        return self


class Inputs(StrictModel):
    precipitation: Path | None = None
    reference_et: Path | None = None
    root_restricting_depth: Path | None = None
    pawc: Path | None = None
    lulc: Path | None = None
    watersheds: Path | None = None
    biophysical_table: Path | None = None
    versions: dict[str, str] = Field(default_factory=dict)


class Model(StrictModel):
    z_parameter: Annotated[float, Field(gt=0)] | None = None


class Resources(StrictModel):
    max_workers: PositiveInt = 2
    threads_per_worker: PositiveInt = 1
    memory_budget_gb: Annotated[float, Field(gt=0, le=24)] = 24
    timeout_hours: Annotated[float, Field(gt=0, le=12)] = 12


class ProjectConfig(StrictModel):
    schema_version: Literal[1]
    paths: Paths
    study: Study = Field(default_factory=Study)
    inputs: Inputs = Field(default_factory=Inputs)
    model: Model = Field(default_factory=Model)
    resources: Resources = Field(default_factory=Resources)
    random_seed: Annotated[int, Field(strict=True, ge=0)] = 42
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    def validate_inputs(self) -> None:
        """Fail before model execution if required files or parameters are missing.

        Spatial compatibility and InVEST's own validation belong to M2/M3.
        """
        problems = []
        for name in type(self.inputs).model_fields:
            if name == "versions":
                continue
            value = getattr(self.inputs, name)
            if value is None or not value.is_file():
                problems.append(f"inputs.{name}: missing file ({value})")
        if self.model.z_parameter is None:
            problems.append("model.z_parameter: not configured")
        if not self.study.station_ids:
            problems.append("study.station_ids: not configured")
        if self.study.projected_crs is None or self.study.resolution_m is None:
            problems.append("study: projected_crs and resolution_m are required")
        if problems:
            raise ConfigError("\n".join(problems))


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently accepting the last one."""


def _unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigError("YAML mapping keys must be strings")
        if key in mapping:
            raise ConfigError(f"Duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def project_root() -> Path:
    """Resolve the editable source checkout, independent of the working directory."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise ConfigError("Cannot locate checkout; supply --project-root")
    return root


def load_config(path: str | Path = "config.yaml", *, root: Path | None = None) -> ProjectConfig:
    root = (root or project_root()).resolve()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    try:
        raw = yaml.load(config_path.read_text(encoding="utf-8-sig"), Loader=UniqueKeyLoader)
        config = ProjectConfig.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ConfigError(f"Cannot load {config_path}: {error}") from error
    for section in (config.paths, config.inputs):
        for name in type(section).model_fields:
            value = getattr(section, name)
            if isinstance(value, Path):
                resolved = value if value.is_absolute() else root / value
                setattr(section, name, resolved.resolve())
    return config
