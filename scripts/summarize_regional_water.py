"""Summarize the regional experiment, structural error floor, and reach residuals."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import isotonic_regression

from ecologyhydro.aggregation import write_csv
from ecologyhydro.baseline import read_json
from ecologyhydro.config import load_config, project_root
from ecologyhydro.regional_calibration import ENDPOINTS
from ecologyhydro.simulation import write_json
from ecologyhydro.water_balance import read_csv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def monotone_lower_bound(observed):
    """Exact relative squared-error floor for ANY nonnegative incremental-yield model."""
    values = np.asarray(observed, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Positive finite observations required")
    fitted = isotonic_regression(values, weights=1 / values**2).x
    return fitted, float(np.sqrt(np.mean(((fitted - values) / values) ** 2)) * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="project/calibration/regional_water_v1")
    parser.add_argument("--refinement")
    args = parser.parse_args()
    root = project_root()
    output = root / args.experiment
    summary = read_json(output / "summary.json")
    stations = load_config().study.station_ids
    endpoints = [stations[i] for i in ENDPOINTS]
    observations = {
        r["测站"]: r for r in read_csv(root / "data/Hydrology/实测年径流量2018-2023.csv")
    }
    bounds = []
    for year in range(2019, 2024):
        for scope, selected in (("six_endpoints", endpoints), ("all_eleven", stations)):
            observed = np.array([float(observations[s][str(year)]) for s in selected])
            _, lower = monotone_lower_bound(observed)
            bounds.append(
                {"year": year, "scope": scope, "minimum_possible_relative_rmse_pct": lower}
            )
    write_csv(output / "nonnegative_yield_error_floor.csv", bounds)
    # All old-model comparisons use exactly the same six endpoints as the new candidates.
    old_rows = read_csv(root / "project/calibration/major_bias/official_comparison.csv")
    old_test = read_csv(root / "project/calibration/major_bias/holdout_2023/stations.csv")
    scores, reaches = [], []
    for product in ("fine", "copernicus"):
        for phase, rows, field in (
            ("training", old_rows, "after"),
            ("reused_2023", old_test, "predicted"),
        ):
            chosen = [r for r in rows if r["landcover"] == product and r["station"] in endpoints]
            relative = np.array([float(r[field]) / float(r["observed"]) - 1 for r in chosen])
            scores.append(
                {
                    "landcover": product,
                    "variant": "old_frozen_six_endpoints",
                    "phase": phase,
                    "mape_pct": float(np.mean(np.abs(relative)) * 100),
                    "relative_rmse_pct": float(np.sqrt(np.mean(relative**2)) * 100),
                }
            )
        for variant, phases in summary["products"][product]["metrics"].items():
            for phase in ("training", "development_leave_year_out", "reused_2023"):
                scores.append(
                    {
                        "landcover": product,
                        "variant": variant,
                        "phase": phase,
                        "mape_pct": phases[phase]["mape_pct"],
                        "relative_rmse_pct": phases[phase]["relative_rmse_pct"],
                    }
                )
        rows = read_csv(output / f"stations_{product}.csv")
        for variant in summary["products"][product]["metrics"]:
            for year in range(2019, 2024):
                selected = {
                    r["station"]: r
                    for r in rows
                    if r["variant"] == variant and int(r["year"]) == year
                }
                upstream = dict.fromkeys(
                    (
                        "awy_yield",
                        "predicted",
                        "observed",
                        "consumption_cumulative",
                        "storage_cumulative",
                    ),
                    0.0,
                )
                for station in endpoints:
                    row = selected[station]
                    difference = {k: float(row[k]) - float(upstream[k]) for k in upstream}
                    reaches.append(
                        {
                            "landcover": product,
                            "variant": variant,
                            "year": year,
                            "downstream_station": station,
                            **difference,
                            "increment_residual": difference["predicted"] - difference["observed"],
                        }
                    )
                    upstream = {k: float(row[k]) for k in upstream}
    target_output = output
    refinement = root / args.refinement if args.refinement else None
    if refinement:
        refined = read_json(refinement / "summary.json")
        for product in ("fine", "copernicus"):
            for version, phases in refined["products"][product].items():
                for phase in ("training", "development_leave_year_out", "reused_2023"):
                    scores.append(
                        {
                            "landcover": product,
                            "variant": f"four_regions_{version}",
                            "phase": phase,
                            "mape_pct": phases[phase]["mape_pct"],
                            "relative_rmse_pct": phases[phase]["relative_rmse_pct"],
                        }
                    )
        target_output = refinement
    write_csv(target_output / "comparison_metrics.csv", scores)
    write_csv(output / "reach_residuals.csv", reaches)
    figures(output, refinement)
    write_json(
        target_output / "analysis_summary.json",
        {"comparison": scores, "monotone_lower_bounds": bounds},
    )
    for row in scores:
        print(row)


def figures(output: Path, refinement: Path | None = None):
    """Export shareable static figures; data labels keep observation scope explicit."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )
    stations = ["贵得", "兰州", "头道拐", "龙门", "三门峡", "花园口"]
    variants = {
        "uniform_raw_legacy": ("统一 Z，未加管理", "#b7bcc3"),
        "regional_raw_legacy": ("三分区 Z，未加管理", "#bc7e2d"),
        "uniform_managed_legacy": ("统一 Z，加入管理", "#347ba4"),
        "regional_managed_legacy": ("三分区 Z，加入管理", "#21836b"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, product, title in zip(
        axes, ("fine", "copernicus"), ("精细分类", "Copernicus"), strict=True
    ):
        rows = read_csv(output / f"stations_{product}.csv")
        for variant, (label, color) in variants.items():
            lookup = {
                r["station"]: r for r in rows if r["variant"] == variant and r["year"] == "2023"
            }
            ax.plot(
                stations,
                [float(lookup[s]["predicted"]) for s in stations],
                marker="o",
                label=label,
                color=color,
                linewidth=1.8,
            )
        ax.plot(
            stations,
            [float(lookup[s]["observed"]) for s in stations],
            "k--",
            marker="s",
            label="实测",
        )
        if refinement:
            refined_rows = read_csv(refinement / f"stations_{product}.csv")
            refined_lookup = {
                r["station"]: r
                for r in refined_rows
                if r["variant"] == "legacy" and r["year"] == "2023"
            }
            ax.plot(
                stations,
                [float(refined_lookup[s]["predicted"]) for s in stations],
                marker="D",
                color="#7046a0",
                label="四分区 Z，加入管理",
                linewidth=2,
            )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("年径流量（亿 m³）")
    axes[1].legend(fontsize=9)
    fig.suptitle("2023 复用测试年：相同六个测站，参数仅用 2019—2022 率定")
    fig.tight_layout()
    destination = refinement or output
    fig.savefig(destination / "reused_2023_comparison.png", dpi=180)
    fig.savefig(destination / "reused_2023_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
