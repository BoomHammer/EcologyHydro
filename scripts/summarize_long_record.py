"""Report spatially defined regional models and out-of-time water balance evidence."""

import argparse

import matplotlib
import numpy as np
import pandas as pd

from ecologyhydro.baseline import read_json
from ecologyhydro.config import project_root
from ecologyhydro.long_record import STATIONS
from ecologyhydro.water_balance import REGIONS, read_csv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/calibration/long_record_v1")
    args = parser.parse_args()
    root, output = project_root(), project_root() / args.output
    summary = read_json(output / "summary.json")
    records, annual, parameter_rows, errors = [], [], [], []
    for product, cases in summary["products"].items():
        frame = pd.read_csv(output / f"stations_{product}.csv")
        parameters = read_json(output / f"parameters_{product}.json")
        for name, fit in parameters.items():
            model, account, window = name.split("__")
            for i, value in enumerate(fit["parameters"]):
                count = len(fit["parameters"])
                label = f"Z{i + 1}" if i < count - 2 else "Kc_scale" if i == count - 2 else "L"
                parameter_rows.append(
                    {
                        "landcover": product,
                        "model": model,
                        "account": account,
                        "window": window,
                        "parameter": label,
                        "value": value,
                        "active_bound": fit["active_bounds"][i],
                        "training_observations": fit["observations"],
                    }
                )
        for key, case in cases.items():
            model, account, phase = key.split("__")
            for scope, metrics in (
                ("seven_conditional", case),
                ("common_six", case["common_six_stations"]),
                ("lijin_conditional", case["lower_reach_conditional"]),
            ):
                records.append(
                    {
                        "landcover": product,
                        "model": model,
                        "account": account,
                        "phase": phase,
                        "scope": scope,
                        **{k: v for k, v in metrics.items() if not isinstance(v, dict)},
                    }
                )
            for year, metrics in case["years"].items():
                annual.append(
                    {
                        "landcover": product,
                        "model": model,
                        "account": account,
                        "phase": phase,
                        "year": int(year),
                        **metrics,
                    }
                )
        for (model, account, phase, year), group in frame.groupby(
            ["model", "account", "phase", "year"]
        ):
            group = group.set_index("station").loc[STATIONS]
            pred, obs = group.predicted.to_numpy(), group.observed.to_numpy()
            for si, station in enumerate(STATIONS):
                delta = pred[si] - (pred[si - 1] if si else 0)
                observed_delta = obs[si] - (obs[si - 1] if si else 0)
                errors.append(
                    {
                        "landcover": product,
                        "model": model,
                        "account": account,
                        "phase": phase,
                        "year": year,
                        "downstream": station,
                        "predicted_increment": delta,
                        "observed_increment": observed_delta,
                        "increment_error": delta - observed_delta,
                    }
                )
    metrics_frame, annual_frame = pd.DataFrame(records), pd.DataFrame(annual)
    metrics_frame.to_csv(output / "comparison.csv", index=False)
    annual_frame.to_csv(output / "annual_metrics.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output / "parameter_trajectories.csv", index=False)
    pd.DataFrame(errors).to_csv(output / "reach_errors.csv", index=False)
    climate = pd.read_csv(output / "climate_regions_fine.csv")
    other = pd.read_csv(output / "climate_regions_copernicus.csv")
    np.testing.assert_allclose(
        climate[["area_km2", "precipitation_mm", "pet_mm"]],
        other[["area_km2", "precipitation_mm", "pet_mm"]],
        rtol=1e-10,
    )
    reference = {
        r["水资源二级区"]: r for r in read_csv(root / "data/Hydrology/降水量2013-2023.csv")
    }
    areas = {
        r["水资源二级区"]: float(r["计算面积(万平方千米)"])
        for r in read_csv(root / "data/Hydrology/水资源二级区面积.csv")
    }
    rainfall_rows = []
    for row in climate.to_dict(orient="records"):
        region, year = REGIONS[int(row["region_id"]) - 1], int(row["year"])
        value = reference[region].get(str(year))
        observed = float(value) * 10 / areas[region] if value else np.nan
        rainfall_rows.append(
            {
                **row,
                "region": region,
                "statistical_precipitation_mm": observed,
                "proxy_difference_pct": (row["precipitation_mm"] / observed - 1) * 100,
                "same_boundary": False,
            }
        )
    pd.DataFrame(rainfall_rows).to_csv(output / "rainfall_comparison.csv", index=False)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    labels = {
        "macro3": "上中下游三区代理",
        "reach7": "七个测站区间",
        "pooled7": "七区＋固定温和约束",
    }
    colors = {"macro3": "#b67736", "reach7": "#3d73a2", "pooled7": "#278673"}
    for column, product in enumerate(("fine", "copernicus")):
        for row, phase in enumerate(("fixed_early", "rolling")):
            for model, label in labels.items():
                selected = annual_frame[
                    (annual_frame.landcover == product)
                    & (annual_frame.account == "legacy")
                    & (annual_frame.phase == phase)
                    & (annual_frame.model == model)
                ]
                axes[row, column].plot(
                    selected.year,
                    selected.relative_rmse_pct,
                    "o-",
                    label=label,
                    color=colors[model],
                )
            axes[row, column].set(
                title=f"{'精细地类' if product == 'fine' else 'Copernicus'}："
                + ("2013—2017 固定训练" if phase == "fixed_early" else "逐年扩展早期训练"),
                ylabel="七站相对 RMSE（%）",
                xticks=[2019, 2020, 2021, 2022],
            )
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "时间顺序验证：只用过去拟合参数\n同一测站代理边界；利津账户为条件近似；非全流程独立验证"
    )
    fig.tight_layout()
    fig.savefig(output / "temporal_validation.png", dpi=180)
    fig.savefig(output / "temporal_validation.pdf")
    plt.close(fig)
    selected = metrics_frame[
        (metrics_frame.account == "legacy") & (metrics_frame.scope == "seven_conditional")
    ]
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
