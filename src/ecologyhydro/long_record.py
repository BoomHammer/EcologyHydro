"""Missing-aware water accounts and explicit station-proxy regional definitions."""

import numpy as np

from ecologyhydro.water_balance import REGIONS, read_csv

YEARS = list(range(2013, 2023))
ENDPOINTS = np.array([1, 2, 5, 6, 7, 8, 10])
STATIONS = ["贵得", "兰州", "头道拐", "龙门", "三门峡", "花园口", "利津"]
REACH_MAP = np.array([0, 0, 1, 2, 2, 2, 3, 4, 5, 6, 6])
MACRO_MAP = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
REACH_MACRO = np.array([0, 0, 0, 1, 1, 1, 2])


def number(value):
    """Keep explicit missing values unknown, never zero or interpolation."""
    if value is None or str(value).strip().lower() in ("", "na", "nan", "--", "—", "缺测"):
        return np.nan
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("Non-finite nonmissing numeric field")
    return result


def table(path, label):
    rows = read_csv(path)
    result = {row[label]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicated labels in {path}")
    return result


def load_accounts(root, version="legacy", years=YEARS):
    folder = root / "data/Hydrology"
    observations = table(folder / "实测年径流量2013-2023.csv", "测站")
    consumption = table(folder / "地表水耗水2013-2023.csv", "水资源二级区")
    storage = table(folder / "大中型水库年蓄水变化量2013-2023.csv", "水资源二级区")
    target = np.array([[number(observations[s].get(str(y))) for s in STATIONS] for y in years])
    c = np.array([[number(consumption[r].get(str(y))) for r in REGIONS[:7]] for y in years])
    s = np.array([[number(storage[r].get(str(y))) for r in REGIONS[:7]] for y in years])
    if version == "sector_2022":
        alternative = table(folder / "用取水耗水量/2022年黄河流域_地表水耗水量.csv", "水资源二级区")
        if 2022 in years:
            c[years.index(2022)] = [number(alternative[r]["合计（亿立方米）"]) for r in REGIONS[:7]]
    elif version != "legacy":
        raise ValueError("Unknown account version")
    if np.any(c[np.isfinite(c)] < 0) or np.any(target[np.isfinite(target)] <= 0):
        raise ValueError("Invalid consumption or observed flow")
    # NumPy cumsum deliberately propagates an upstream missing account downstream.
    adjustment = np.cumsum(c + s, axis=1)
    eligible = np.isfinite(target) & np.isfinite(adjustment)
    return target, c, s, adjustment, eligible


def scores(prediction, target, eligible):
    prediction, target, eligible = np.asarray(prediction), np.asarray(target), np.asarray(eligible)
    if prediction.shape != target.shape or eligible.shape != target.shape:
        raise ValueError("Score arrays must align")
    if not eligible.any():
        return {"n": 0, "mape_pct": None, "relative_rmse_pct": None, "bias_pct": None}
    if not np.isfinite(prediction[eligible]).all() or not np.isfinite(target[eligible]).all():
        raise ValueError("Cannot silently drop invalid predictions")
    error = prediction[eligible] - target[eligible]
    relative = error / target[eligible]
    return {
        "n": int(eligible.sum()),
        "mape_pct": float(np.abs(relative).mean() * 100),
        "relative_rmse_pct": float(np.sqrt(np.mean(relative**2)) * 100),
        "bias_pct": float(error.sum() / target[eligible].sum() * 100),
        "worst_station_year_pct": float(np.abs(relative).max() * 100),
        "negative_predictions": int((prediction[eligible] < 0).sum()),
    }
