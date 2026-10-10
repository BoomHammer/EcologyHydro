"""Report every domain/storage candidate, selecting only by development relative RMSE."""

import argparse

import matplotlib
import numpy as np
import pandas as pd

from ecologyhydro.baseline import read_json
from ecologyhydro.config import project_root
from ecologyhydro.simulation import write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rainfall", action="store_true")
    args = parser.parse_args()
    output = root / "project/calibration/connected_water_v1"
    rows = []
    folders = ["regional_water_v2", "reach_loss_v1", "connected_water_v1"]
    if args.rainfall:
        folders.append("reference_rainfall_v1")
    for folder in folders:
        summary = read_json(root / f"project/calibration/{folder}/summary.json")
        for product, variants in summary["products"].items():
            for variant, case in variants.items():
                mode, version = variant.split("__") if "__" in variant else (folder, variant)
                if folder == "reference_rainfall_v1":
                    mode = "domain_reference_rainfall_loss"
                for phase in ("training", "development_leave_year_out", "reused_2023"):
                    rows.append(
                        {
                            "landcover": product,
                            "model": mode,
                            "account": version,
                            "phase": phase,
                            **case[phase],
                        }
                    )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "comparison_metrics.csv", index=False)
    selection = comparison[comparison.phase == "development_leave_year_out"].sort_values(
        "relative_rmse_pct"
    )
    selection = selection.groupby(["landcover", "account"]).first().reset_index()
    write_json(
        output / "development_selection.json",
        {
            "rule": "minimum development leave-year-out relative RMSE including previous models",
            "selected": selection.to_dict(orient="records"),
            "production_accepted": False,
            "reused_2023_used_for_selection": False,
        },
    )
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, product, title in zip(
        axes, ("fine", "copernicus"), ("精细地类", "Copernicus"), strict=True
    ):
        for folder, variant, label, color in (
            ("reach_loss_v1", "legacy", "上一轮：原范围＋净损失", "#d79439"),
            ("connected_water_v1", "domain_loss__legacy", "排除共和内流盆地", "#627ca3"),
            ("connected_water_v1", "domain_storage_loss__legacy", "再加土壤／积雪蓄变", "#187e79"),
        ):
            station_rows = pd.read_csv(
                root / f"project/calibration/{folder}/stations_{product}.csv"
            )
            selected = station_rows[(station_rows.variant == variant) & (station_rows.year == 2023)]
            ax.plot(selected.station, selected.predicted, "o-", label=label, color=color)
        ax.plot(selected.station, selected.observed, "s--", label="实测", color="#333333")
        ax.set(title=title, ylabel="年径流量（亿 m³）")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("2023 复用测试年：边界与跨年蓄变对照（非独立验证）")
    fig.tight_layout()
    fig.savefig(output / "reused_2023_comparison.png", dpi=180)
    fig.savefig(output / "reused_2023_comparison.pdf")
    plt.close(fig)
    storage = pd.read_csv(root / "project/diagnostics/storage_change_v1/storage_changes.csv")
    combined = (
        storage[storage.domain == "gonghe_excluded"]
        .groupby(["year", "zone"])
        .delta_storage_1e8_m3.sum()
    )
    matrix = combined.unstack("zone").cumsum(axis=1)
    matrix = matrix[[2, 3, 6, 7, 8, 9]]
    matrix.columns = ["贵得", "兰州", "头道拐", "龙门", "三门峡", "花园口"]
    matrix.to_csv(output / "soil_snow_cumulative.csv", encoding="utf-8-sig")
    residuals = []
    for product in ("fine", "copernicus"):
        stations = pd.read_csv(output / f"stations_{product}.csv")
        for (variant, year), group in stations.groupby(["variant", "year"], sort=False):
            pred = np.diff(np.r_[0, group.predicted.to_numpy()])
            obs = np.diff(np.r_[0, group.observed.to_numpy()])
            for si, name in enumerate(group.station):
                residuals.append(
                    {
                        "landcover": product,
                        "variant": variant,
                        "year": year,
                        "downstream_station": name,
                        "predicted_increment": pred[si],
                        "observed_increment": obs[si],
                        "residual": pred[si] - obs[si],
                    }
                )
    pd.DataFrame(residuals).to_csv(output / "reach_residuals.csv", index=False)
    fold_rows = []
    for folder in folders:
        if folder == "regional_water_v2":
            continue
        for product in ("fine", "copernicus"):
            records = pd.read_csv(root / f"project/calibration/{folder}/cv_stations_{product}.csv")
            for (variant, year), group in records.groupby(["variant", "year"]):
                error = group.predicted / group.observed - 1
                fold_rows.append(
                    {
                        "folder": folder,
                        "landcover": product,
                        "variant": variant,
                        "year": year,
                        "mape_pct": np.abs(error).mean() * 100,
                        "relative_rmse_pct": np.sqrt(np.mean(error**2)) * 100,
                    }
                )
    pd.DataFrame(fold_rows).to_csv(output / "leave_year_diagnostics.csv", index=False)
    print(comparison[comparison.account == "legacy"].to_string(index=False))
    print(selection.to_string(index=False))


if __name__ == "__main__":
    main()
