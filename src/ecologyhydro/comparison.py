"""YAML-driven temporal protocol, missing-aware accounts and paired metrics."""

from pathlib import Path

import numpy as np
import yaml

from ecologyhydro.config import Resources, UniqueKeyLoader, project_root
from ecologyhydro.long_record import STATIONS, number, table
from ecologyhydro.water_balance import REGIONS

PHASES = ("warmup", "calibration_stage1", "bridge", "calibration_stage2", "validation")


def load_protocol(path, root=None):
    root = (root or project_root()).resolve()
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    config = yaml.load(source.read_text(encoding="utf-8-sig"), Loader=UniqueKeyLoader)
    if config["schema_version"] != 1 or config["model"] != "reach7":
        raise ValueError("Unsupported comparison schema/model")
    periods = config["years"]
    if set(periods) != set(PHASES):
        raise ValueError("Specify all five temporal phases")
    years = [y for phase in PHASES for y in periods[phase]]
    if any(type(y) is not int for y in years) or not years:
        raise ValueError("Years must be integers")
    if years != sorted(set(years)) or years != list(range(years[0], years[-1] + 1)):
        raise ValueError("Phases must be ordered, disjoint and cover every forcing year")
    if any(not periods[p] for p in PHASES):
        raise ValueError("Every phase requires at least one year")
    if config["evaluation"]["managed_stations"] != STATIONS:
        raise ValueError("Managed station order must match the seven-region account model")
    if len(config["evaluation"]["all_stations"]) != 11:
        raise ValueError("Routing grid requires eleven stations")
    if len(config["products"]) != 3:
        raise ValueError("Cross comparison requires exactly three products")
    Resources.model_validate(config["resources"])
    config["root"] = root
    config["config_path"] = source.resolve()
    config["forcing_years"] = years
    return config


def resolve(config, value, **fields):
    path = Path(str(value).format(**fields))
    return path.resolve() if path.is_absolute() else (config["root"] / path).resolve()


def phase_for(config, year):
    return next(phase for phase in PHASES if year in config["years"][phase])


def training_mask(config, stage1=False):
    selected = config["years"]["calibration_stage1"].copy()
    if not stage1:
        selected += config["years"]["calibration_stage2"]
    return np.isin(config["forcing_years"], selected)


def accounts(config):
    years = config["forcing_years"]
    paths = config["hydrology"]
    observed = table(resolve(config, paths["observations"]), "测站")
    consumed = table(resolve(config, paths["consumption"]), "水资源二级区")
    stored = table(resolve(config, paths["storage"]), "水资源二级区")
    for records in (observed, consumed, stored):
        if any(str(y) not in next(iter(records.values())) for y in years):
            raise ValueError("Required year column missing from hydrological input")
    target = np.array([[number(observed[s][str(y)]) for s in STATIONS] for y in years])
    c = np.array([[number(consumed[r][str(y)]) for r in REGIONS[:7]] for y in years])
    s = np.array([[number(stored[r][str(y)]) for r in REGIONS[:7]] for y in years])
    if (c[np.isfinite(c)] < 0).any() or (target[np.isfinite(target)] <= 0).any():
        raise ValueError("Invalid consumption or observed runoff")
    adjustment = (c + s).cumsum(axis=1)
    valid = np.isfinite(target) & np.isfinite(adjustment)
    return target, adjustment, valid, c, s


def metrics(predicted, observed, eligible):
    predicted, observed = np.asarray(predicted), np.asarray(observed)
    eligible = np.asarray(eligible, dtype=bool)
    if not (predicted.shape == observed.shape == eligible.shape):
        raise ValueError("Metric array shapes must match")
    p, o = predicted[eligible], observed[eligible]
    if not np.isfinite(p).all() or not np.isfinite(o).all() or (o <= 0).any():
        raise ValueError("Invalid eligible observation/prediction")
    if not len(o):
        return {"n": 0, "rmse": None, "relative_rmse_pct": None, "bias_pct": None}
    error = p - o
    return {
        "n": len(o),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "relative_rmse_pct": float(np.sqrt(np.mean((error / o) ** 2)) * 100),
        "bias_pct": float(error.sum() / o.sum() * 100),
    }


def prediction_rows(
    config, parameter_source, product, predicted, natural, target, adjustment, valid
):
    rows = []
    for yi, year in enumerate(config["forcing_years"]):
        phase = phase_for(config, year)
        for si, station in enumerate(STATIONS):
            scored = bool(valid[yi, si] and phase not in ("warmup", "bridge"))
            error = float(predicted[yi, si] - target[yi, si]) if scored else None
            rows.append(
                {
                    "parameter_source": parameter_source,
                    "landcover": product,
                    "year": year,
                    "phase": phase,
                    "station": station,
                    "awy_yield": float(natural[yi, si]),
                    "predicted": finite_value(predicted[yi, si]),
                    "observed": finite_value(target[yi, si]),
                    "known_adjustment": finite_value(adjustment[yi, si]),
                    "scored": scored,
                    "exclusion_reason": phase
                    if phase in ("warmup", "bridge")
                    else ("" if scored else "missing_observation_or_upstream_account"),
                    "error": error,
                    # Annual observations: singleton RMSE is exactly absolute error.
                    "station_year_rmse": abs(error) if error is not None else None,
                }
            )
    return rows


def finite_value(value):
    return float(value) if np.isfinite(value) else None


def summarize_rows(rows):
    """Separate year, station-period and pooled-period metrics; no warmup/bridge scores."""
    years = sorted({r["year"] for r in rows})
    stations = list(dict.fromkeys(r["station"] for r in rows))
    selectors = [("year", str(y), [r for r in rows if r["year"] == y]) for y in years]
    for phase in ("calibration_stage1", "calibration_stage2", "calibration", "validation"):
        selected = [
            r
            for r in rows
            if r["phase"] == phase
            or (phase == "calibration" and r["phase"].startswith("calibration"))
        ]
        selectors.append(("period", phase, selected))
        selectors.extend(
            ("station_period", f"{s}/{phase}", [r for r in selected if r["station"] == s])
            for s in stations
        )
    result = []
    for scope, label, selected in selectors:
        result.append(
            {
                "parameter_source": rows[0]["parameter_source"],
                "landcover": rows[0]["landcover"],
                "scope": scope,
                "label": label,
                **metrics(
                    np.array([r["predicted"] for r in selected], dtype=float),
                    np.array([r["observed"] for r in selected], dtype=float),
                    [r["scored"] for r in selected],
                ),
            }
        )
    return result
