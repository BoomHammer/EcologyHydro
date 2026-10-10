"""Summarize spatial attribution, fixed-parameter bounds, and reach-loss experiments."""

import argparse

import matplotlib
import numpy as np
import pandas as pd
from calibrate_regional_water import source_paths
from osgeo import gdal

from ecologyhydro.baseline import read_json
from ecologyhydro.config import load_config, project_root
from ecologyhydro.regional_calibration import ENDPOINTS

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def draw_map(root, audit):
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    paths = source_paths(root, index, "fine", 2023)
    bounds = [101, 35, 112, 42]

    def preview(path):
        with gdal.Warp(
            "",
            str(path),
            format="MEM",
            dstSRS="EPSG:4326",
            outputBounds=bounds,
            width=1100,
            height=700,
            resampleAlg="near",
            dstNodata=0,
        ) as ds:
            return ds.ReadAsArray()

    zone, inland, code = (
        preview(p) for p in (paths["zone"], audit / "inland_mask.tif", paths["code"])
    )
    reach = np.isin(zone, [4, 5, 6])
    desert = np.isin(code, [9, 10, 11, 12])
    layer = np.zeros(zone.shape, dtype=np.uint8)
    layer[inland > 0] = 1
    layer[reach] = 2
    layer[reach & desert] = 3
    layer[reach & (inland > 0)] = 4
    colors = ["#ffffff", "#e4d8ef", "#acd9dd", "#dba34c", "#ba3030"]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(
        layer,
        extent=[101, 112, 35, 42],
        origin="upper",
        interpolation="nearest",
        cmap=ListedColormap(colors),
        vmin=0,
        vmax=4,
        aspect=1 / np.cos(np.deg2rad(38.5)),
    )
    stations = pd.read_csv(root / "data/Hydrology/测站控制面积.csv")
    for _, row in stations[
        stations["测站"].isin(["兰州", "下河沿", "石嘴山", "头道拐"])
    ].iterrows():
        ax.plot(row.X, row.Y, "o", color="#172b45", markersize=4)
        ax.annotate(row["测站"], (row.X, row.Y), xytext=(5, 6), textcoords="offset points")
    labels = [
        "HydroBASINS 已标内流面",
        "模型兰州—头道拐增量汇水区",
        "其中精细地类沙漠类别",
        "汇水区与内流面重叠",
    ]
    ax.legend(
        handles=[
            Patch(facecolor=color, label=label)
            for color, label in zip(colors[1:], labels, strict=True)
        ],
        loc="upper left",
        fontsize=9,
        framealpha=0.95,
    )
    ax.set(
        xlabel="经度（°E）",
        ylabel="纬度（°N）",
        title="兰州—头道拐：内流面、模型汇水范围与沙漠地类",
    )
    ax.grid(alpha=0.15)
    fig.text(
        0.5,
        0.015,
        "图示采用降采样；统计使用完整 250 m 网格。HydroBASINS 不是官方水资源二级区划。",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(audit / "connectivity_map.png", dpi=180)
    fig.savefig(audit / "connectivity_map.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="project/diagnostics/dryland_connectivity_v2")
    parser.add_argument("--experiment", default="project/calibration/reach_loss_v1")
    args = parser.parse_args()
    root = project_root()
    audit, experiment = root / args.audit, root / args.experiment
    gdal.UseExceptions()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    raw = pd.read_csv(audit / "yield_by_zone_category.csv")
    rows, inland_rows = [], []
    names = load_config().study.station_ids
    for (product, year), group in raw.groupby(["landcover", "year"]):
        old = pd.read_csv(root / f"project/calibration/regional_water_v2/stations_{product}.csv")
        old = old[(old.variant == "legacy") & (old.year == year)].set_index("station")
        reach = group[group.zone.between(4, 6)].groupby("category").sum(numeric_only=True)
        upstream = group[group.zone <= 6].groupby("category").sum(numeric_only=True)
        q_up, q_down = old.loc["兰州"], old.loc["头道拐"]
        gap = (q_down.predicted - q_up.predicted) - (q_down.observed - q_up.observed)
        cumulative_inland = (
            group[group.category == "mapped_inland"]
            .sort_values("zone")["yield_1e8_m3"]
            .cumsum()
            .to_numpy()
        )
        for station in ENDPOINTS:
            row = old.loc[names[station]]
            inland_rows.append(
                {
                    "landcover": product,
                    "year": year,
                    "station": names[station],
                    "removed_yield": cumulative_inland[station],
                    "predicted": row.predicted - cumulative_inland[station],
                    "observed": row.observed,
                }
            )
        for category in ("all", "mapped_inland", "desert_proxy", "inland_or_desert"):
            item = reach.loc[category]
            rows.append(
                {
                    "landcover": product,
                    "year": year,
                    "category": category,
                    "area_km2": item.area_km2,
                    "area_pct_reach": item.area_km2 / reach.loc["all", "area_km2"] * 100,
                    "yield_1e8_m3": item.yield_1e8_m3,
                    "yield_pct_reach": item.yield_1e8_m3 / reach.loc["all", "yield_1e8_m3"] * 100,
                    "yield_depth_mm": item.yield_1e8_m3 * 1e5 / item.area_km2
                    if item.area_km2
                    else 0,
                    "baseline_reach_residual": gap,
                    "residual_after_local_yield_removal": gap - item.yield_1e8_m3,
                    "baseline_tdg_error": q_down.predicted - q_down.observed,
                    "tdg_error_after_local_yield_removal": q_down.predicted
                    - q_down.observed
                    - item.yield_1e8_m3,
                    "category_yield_all_upstream_of_tdg": upstream.loc[category, "yield_1e8_m3"],
                }
            )
    attribution = pd.DataFrame(rows)
    attribution.to_csv(audit / "reach_attribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inland_rows).to_csv(audit / "inland_removal_sensitivity.csv", index=False)
    metric_rows = []
    for folder in (root / "project/calibration/regional_water_v2", experiment):
        summary = read_json(folder / "summary.json")
        for product, versions in summary["products"].items():
            for version, result in versions.items():
                for phase in ("training", "development_leave_year_out", "reused_2023"):
                    metric_rows.append(
                        {
                            "model": folder.name,
                            "landcover": product,
                            "version": version,
                            "phase": phase,
                            **result[phase],
                        }
                    )
    pd.DataFrame(metric_rows).to_csv(experiment / "comparison_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, product, title in zip(
        axes, ("fine", "copernicus"), ("精细地类", "Copernicus"), strict=True
    ):
        for folder, label, color in (
            (root / "project/calibration/regional_water_v2", "四分区＋已知账户", "#d19142"),
            (experiment, "另加一个净损失参数", "#187f8c"),
        ):
            stations = pd.read_csv(folder / f"stations_{product}.csv")
            selected = stations[(stations.variant == "legacy") & (stations.year == 2023)]
            ax.plot(selected.station, selected.predicted, "o-", label=label, color=color)
        ax.plot(selected.station, selected.observed, "s--", color="#333333", label="实测")
        ax.set(title=title, ylabel="年径流量（亿 m³）")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("2023 复用测试年：六站对照（非独立验证）")
    fig.tight_layout()
    fig.savefig(experiment / "reused_2023_comparison.png", dpi=180)
    fig.savefig(experiment / "reused_2023_comparison.pdf")
    plt.close(fig)
    draw_map(root, audit)
    print(
        attribution[(attribution.landcover == "fine") & (attribution.year == 2023)].to_string(
            index=False
        )
    )
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
