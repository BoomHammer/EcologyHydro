"""Report fixed development-only stability comparisons and parameter movement."""

import argparse

import matplotlib
import numpy as np
import pandas as pd

from ecologyhydro.baseline import read_json
from ecologyhydro.config import project_root

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/calibration/stability_v1")
    parser.add_argument("--full-pixel", action="store_true")
    args = parser.parse_args()
    output = project_root() / args.output
    summary = read_json(
        output / ("full_pixel_cv/summary.json" if args.full_pixel else "summary.json")
    )
    prefix = "official_" if args.full_pixel else ""
    comparisons, yearly, movements = [], [], []
    for product, cases in summary["products"].items():
        parameters = read_json(output / f"parameters_{product}.json")
        for variant, case in cases.items():
            model, account = variant.split("__")
            comparisons.append(
                {
                    "landcover": product,
                    "model": model,
                    "account": account,
                    **{k: v for k, v in case.items() if not isinstance(v, list)},
                }
            )
            for index, year in enumerate(range(2019, 2023)):
                yearly.append(
                    {
                        "landcover": product,
                        "model": model,
                        "account": account,
                        "year": year,
                        "mape_pct": case["year_mape_pct"][index],
                        "relative_rmse_pct": case["year_relative_rmse_pct"][index],
                    }
                )
            if model == "subset_mean_four_region":
                continue  # Ensembles have member parameters, not one averaged physical parameter.
            members = parameters["members"][variant]
            fitted = parameters["fits"][members["full"][0]]
            folds = [parameters["fits"][names[0]] for names in members["folds"]]
            values = np.array([f["parameters"] + [f["annual_loss_1e8_m3"]] for f in folds])
            names = [f"Z{i + 1}" for i in range(len(fitted["parameters"]) - 1)] + ["Kc_scale", "L"]
            final = fitted["parameters"] + [fitted["annual_loss_1e8_m3"]]
            for i, name in enumerate(names):
                movements.append(
                    {
                        "landcover": product,
                        "model": model,
                        "account": account,
                        "parameter": name,
                        "full_fit": final[i],
                        "fold_min": values[:, i].min(),
                        "fold_max": values[:, i].max(),
                        "fold_std": values[:, i].std(),
                        "fold_bound_hits": sum(f["active_bounds"][i] != 0 for f in folds),
                    }
                )
    frame = pd.DataFrame(comparisons)
    frame.to_csv(output / f"{prefix}comparison.csv", index=False)
    annual = pd.DataFrame(yearly)
    annual.to_csv(output / f"{prefix}yearly_metrics.csv", index=False)
    pd.DataFrame(movements).to_csv(output / "parameter_movement.csv", index=False)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    labels = {
        "four_region": "四区单模型",
        "three_region": "三区单模型",
        "subset_mean_four_region": "四区训练子集平均",
    }
    colors = {
        "four_region": "#476e9e",
        "three_region": "#ba7630",
        "subset_mean_four_region": "#288377",
    }
    for column, product in enumerate(("fine", "copernicus")):
        title = "精细地类" if product == "fine" else "Copernicus"
        for model, label in labels.items():
            selected = annual[
                (annual.landcover == product)
                & (annual.account == "legacy")
                & (annual.model == model)
            ]
            axes[0, column].plot(
                selected.year, selected.relative_rmse_pct, "o-", label=label, color=colors[model]
            )
        axes[0, column].set(
            title=title, ylabel="留出年相对 RMSE（%）", xticks=[2019, 2020, 2021, 2022]
        )
        selected = frame[(frame.landcover == product) & (frame.account == "legacy")]
        axes[1, column].bar(
            [labels[m] for m in selected.model],
            selected.delete_year_prediction_rms_change_pct,
            color=[colors[m] for m in selected.model],
        )
        axes[1, column].set(ylabel="删去一年后的预测 RMS 变化／实测（%）")
        for row in range(2):
            axes[row, column].grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "稳定性对照：固定闭流域约束，2019—2022 条件验证\n"
        "旧版水量账户；组合模型每折只使用该折训练年份"
    )
    fig.tight_layout()
    fig.savefig(output / f"{prefix}stability.png", dpi=180)
    fig.savefig(output / f"{prefix}stability.pdf")
    plt.close(fig)
    print(frame[frame.account == "legacy"].to_string(index=False))
    print(summary["decisions"])


if __name__ == "__main__":
    main()
