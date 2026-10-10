"""Separate annual evaporation and storage effects in the headwater diagnostic."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.long_record import scores
from ecologyhydro.simulation import write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/diagnostics/headwater_balance_v1")
    args = parser.parse_args()
    root, output = project_root(), project_root() / args.output
    summary = read_json(output / "summary.json")
    for source in summary["sources"]:
        if fingerprint(Path(source["path"]))["sha256"] != source["sha256"]:
            raise ValueError(f"Changed source {source['path']}")
    monthly = pd.read_csv(output / "monthly_balance.csv")
    annual = pd.read_csv(output / "annual_cumulative_balance.csv")
    decomposition, comparison = [], []
    for product in ("fine", "copernicus"):
        raw = pd.read_csv(root / f"project/calibration/long_record_v1/stations_{product}.csv")
        selected = raw[
            (raw.model == "reach7")
            & (raw.account == "legacy")
            & (raw.phase == "fixed_early")
            & raw.station.isin(["贵得", "兰州"])
        ]
        merged = annual.merge(
            selected, on=["year", "station"], suffixes=("", "_awy"), validate="one_to_one"
        )
        np.testing.assert_allclose(merged.observed, merged.observed_awy, equal_nan=True)
        np.testing.assert_allclose(
            merged.known_adjustment, merged.known_adjustment_awy, equal_nan=True
        )
        merged["landcover"] = product
        # No fitted L is deducted at Guide/Lanzhou, so reconstruct the AWY AET exactly.
        merged["awy_aet"] = merged.ppt - merged.predicted - merged.known_adjustment
        merged["aet_effect_on_q"] = merged.awy_aet - merged.aet
        merged["soil_snow_effect_on_q"] = -merged.delta_soil - merged.delta_snow
        merged["carry_effect_on_q"] = -merged.delta_runoff_carry
        merged["closure_effect_on_q"] = -merged.closure_with_carry
        merged["provider_minus_awy"] = merged.provider_conditional_q - merged.predicted
        reconstructed = (
            merged.aet_effect_on_q
            + merged.soil_snow_effect_on_q
            + merged.carry_effect_on_q
            + merged.closure_effect_on_q
        )
        np.testing.assert_allclose(
            reconstructed, merged.provider_minus_awy, atol=1e-9, equal_nan=True
        )
        merged["awy_plus_external_storage"] = merged.predicted + merged.soil_snow_effect_on_q
        merged["awy_plus_external_storage_and_carry"] = (
            merged.awy_plus_external_storage + merged.carry_effect_on_q
        )
        decomposition.extend(merged.to_dict(orient="records"))
        for station, rows in merged[merged.validation].groupby("station"):
            for label, column in (
                ("awy_fixed_early", "predicted"),
                ("awy_plus_external_storage", "awy_plus_external_storage"),
                ("awy_plus_external_storage_and_carry", "awy_plus_external_storage_and_carry"),
                ("provider_monthly_model", "provider_conditional_q"),
            ):
                comparison.append(
                    dict(
                        landcover=product,
                        station=station,
                        diagnostic=label,
                        **scores(rows[column], rows.observed, rows.eligible),
                    )
                )
    detail = pd.DataFrame(decomposition)
    detail.to_csv(output / "process_decomposition.csv", index=False)
    pd.DataFrame(comparison).to_csv(output / "headwater_comparison.csv", index=False)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, station in enumerate(("贵得", "兰州")):
        reference = annual[annual.station == station]
        axes[0, col].plot(reference.year, reference.observed, "ko-", label="实测径流")
        axes[0, col].plot(
            reference.year,
            reference.provider_conditional_q,
            "o-",
            color="#278673",
            label="月水量模型（同源诊断）",
        )
        for product, color, label in (
            ("fine", "#3d73a2", "精细 AWY"),
            ("copernicus", "#b67736", "Copernicus AWY"),
        ):
            case = detail[(detail.landcover == product) & (detail.station == station)]
            axes[0, col].plot(case.year, case.predicted, "o-", color=color, label=label)
        axes[0, col].axvspan(2018.5, 2022.5, color="gray", alpha=0.08, label="已用于诊断的后期年份")
        axes[0, col].set(title=f"{station}：相同降水下的年度响应", ylabel="年径流（亿 m³）")
        case = detail[
            (detail.landcover == "fine") & (detail.station == station) & detail.validation
        ]
        x = np.arange(len(case))
        for i, (field, label, color) in enumerate(
            (
                ("aet_effect_on_q", "蒸散差异", "#3d73a2"),
                ("soil_snow_effect_on_q", "土壤／积雪蓄变", "#b67736"),
                ("carry_effect_on_q", "月间径流滞蓄", "#278673"),
            )
        ):
            axes[1, col].bar(x + (i - 1) * 0.23, case[field], width=0.23, label=label, color=color)
        axes[1, col].axhline(0, color="gray", linewidth=0.8)
        axes[1, col].set(
            xticks=x,
            xticklabels=case.year.astype(int),
            ylabel="对径流差异的贡献（亿 m³）",
            title="月模型相对精细 AWY：过程差异分解",
        )
        for ax in axes[:, col]:
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(
        "上游主要误差诊断：不率定新参数；同源月模型不是独立观测\n"
        "2018 管理账户缺失，不补值；本图不提供月径流观测验证"
    )
    fig.tight_layout()
    fig.savefig(output / "headwater_process.png", dpi=180)
    fig.savefig(output / "headwater_process.pdf")
    plt.close(fig)
    source = root / "project/diagnostics/error_origin_v1"
    for manifest in (source / "summary.json", output / "precipitation_sensitivity.json"):
        for item in read_json(manifest)["sources"]:
            if fingerprint(Path(item["path"]))["sha256"] != item["sha256"]:
                raise ValueError(f"Changed diagnostic source {item['path']}")
    errors = pd.read_csv(source / "error_propagation.csv")
    totals = pd.read_csv(source / "conditional_downstream_scores.csv")
    accounts = pd.read_csv(source / "water_account_bounds.csv")
    order = ["贵得", "兰州", "头道拐", "龙门", "三门峡", "花园口", "利津"]
    condition_checks = 0
    for (product, account, phase), group in accounts.groupby(["landcover", "account", "phase"]):
        pred = group.pivot(index="year", columns="station", values="predicted")[order].to_numpy()
        obs = group.pivot(index="year", columns="station", values="observed")[order].to_numpy()
        reported = totals[
            (totals.landcover == product) & (totals.account == account) & (totals.phase == phase)
        ].set_index("variant")
        for name, upstream in (("baseline", None), ("known_guide", 0), ("known_lanzhou", 1)):
            conditional = pred[:, 2:].copy()
            if upstream is not None:
                conditional += (obs[:, upstream] - pred[:, upstream])[:, None]
            actual = scores(conditional, obs[:, 2:], np.ones_like(conditional, bool))
            for key, value in actual.items():
                np.testing.assert_allclose(reported.loc[name, key], value)
            condition_checks += 1
    rainfall = pd.read_csv(output / "precipitation_sensitivity.csv")
    np.testing.assert_allclose(
        rainfall.rainfall_only_predicted,
        rainfall.baseline_predicted + rainfall.rainfall_yield_change,
    )
    np.testing.assert_allclose(
        rainfall.rainfall_and_external_storage,
        rainfall.rainfall_only_predicted + rainfall.soil_snow_effect,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = {
        "baseline": "原七区模型",
        "known_guide": "贵得来水已知",
        "known_lanzhou": "兰州来水已知",
    }
    for product, color, label in (
        ("fine", "#3d73a2", "精细"),
        ("copernicus", "#b67736", "Copernicus"),
    ):
        rows = errors[
            (errors.landcover == product)
            & (errors.account == "legacy")
            & (errors.phase == "fixed_early")
            & (errors.year == 2019)
        ]
        axes[0].plot(rows.station, rows.cumulative_error, "o-", color=color, label=label)
        rows = totals[
            (totals.landcover == product)
            & (totals.account == "legacy")
            & (totals.phase == "fixed_early")
        ]
        axes[1].plot(
            [labels[v] for v in rows.variant],
            rows.relative_rmse_pct,
            "o-",
            color=color,
            label=label,
        )
    axes[0].set(title="2019 年误差沿干流传递", ylabel="模拟减实测（亿 m³）")
    axes[1].set(title="2019—2022 下游五站条件诊断", ylabel="相对 RMSE（%）")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("用实测上游来水定位误差来源，不代表全流域预测精度")
    fig.tight_layout()
    fig.savefig(source / "error_origin.png", dpi=180)
    fig.savefig(source / "error_origin.pdf")
    plt.close(fig)
    # Monthly state differences must telescope, not accumulate stored water twice.
    for (_, _), rows in monthly.groupby(["year", "region_id"]):
        if len(rows) != 12:
            raise ValueError("Incomplete monthly accounting")
        np.testing.assert_allclose(
            rows.ppt
            - rows.aet
            - rows.q
            - rows.delta_soil
            - rows.delta_snow
            - rows.delta_runoff_carry,
            rows.closure_with_carry,
            atol=1e-10,
        )
    write_json(
        output / "verification.json",
        {
            "source_fingerprints_checked": len(summary["sources"]),
            "monthly_rows_checked": len(monthly),
            "conditional_score_groups_recomputed": condition_checks,
            "rainfall_diagnostic_rows_checked": len(rainfall),
            "process_decomposition_identity_passed": True,
            "no_fitted_parameters": True,
            "no_2023_rows": not (detail.year == 2023).any().item(),
            "postprocessor": fingerprint(Path(__file__)),
        },
    )
    print(pd.DataFrame(comparison).to_string(index=False))
    print(
        detail[(detail.landcover == "fine") & (detail.year == 2019)][
            [
                "station",
                "observed",
                "predicted",
                "provider_conditional_q",
                "aet_effect_on_q",
                "soil_snow_effect_on_q",
                "carry_effect_on_q",
                "closure_with_carry",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
