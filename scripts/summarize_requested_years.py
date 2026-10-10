"""Verify and present each station in the two user-requested year splits."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.long_record import STATIONS, load_accounts, scores
from ecologyhydro.simulation import write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def formatted(value, digits=2):
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/calibration/requested_years_v1")
    args = parser.parse_args()
    root, output = project_root(), project_root() / args.output
    runtime = read_json(output / "run_summary.json")
    hashes = {}
    for path in [
        output / "manifest.json",
        *[output / f"inputs_{p}.json" for p in ("fine", "copernicus")],
    ]:
        for source in read_json(path)["sources"]:
            hashes[source["path"]] = source["sha256"]
    rotation = read_json(output / "rotation_2023.json")
    hashes[rotation["source"]["path"]] = rotation["source"]["sha256"]
    for path, digest in hashes.items():
        if fingerprint(Path(path))["sha256"] != digest:
            raise ValueError(f"Changed source: {path}")
    rough = read_json(output / "rough_2018_accounts.json")
    _, _, _, earlier_adjustment, _ = load_accounts(root, years=list(range(2013, 2018)))
    for si, row in enumerate(rough):
        valid = np.isfinite(earlier_adjustment[:, si])
        values = earlier_adjustment[valid, si]
        np.testing.assert_allclose(
            [row["mean"], row["minimum"], row["maximum"]],
            [values.mean(), values.min(), values.max()],
        )
        expected_years = np.arange(2013, 2018)[valid].tolist()
        if row["source_years"] != expected_years:
            raise ValueError("Rough accounts used unavailable or future data")
    frames, comparisons, increments, parameter_rows = [], [], [], []
    for product in ("fine", "copernicus"):
        frame = pd.read_csv(output / f"stations_{product}.csv")
        cached = read_json(root / f"project/calibration/long_record_v1/parameters_{product}.json")
        fits = read_json(output / f"parameters_{product}.json")
        if len(frame) != 44 or frame.duplicated(["year", "account", "station"]).any():
            raise ValueError("Incomplete or repeated station records")
        for year, account in frame[["year", "account"]].drop_duplicates().itertuples(index=False):
            fit = fits[f"{year}__{account}"]
            window = "early" if year == 2018 else "all_development"
            original = cached[f"reach7__{account}__{window}"]
            np.testing.assert_array_equal(fit["parameters"], original["parameters"])
            expected_years = (
                list(range(2013, 2018))
                if year == 2018
                else [*range(2013, 2018), *range(2019, 2023)]
            )
            if fit["training_years"] != expected_years or max(expected_years) >= year:
                raise ValueError("Unexpected training year membership")
            selected = (
                frame[(frame.year == year) & (frame.account == account)]
                .set_index("station")
                .loc[STATIONS]
            )
            np.testing.assert_allclose(
                selected.predicted,
                selected.awy_yield - selected.account_adjustment - selected.net_loss,
            )
            if year == 2023:
                target, _, _, adjustment, _ = load_accounts(root, account, years=[2023])
                np.testing.assert_allclose(selected.observed, target[0])
                np.testing.assert_allclose(selected.account_adjustment, adjustment[0])
                expected_n = 7
            else:
                np.testing.assert_allclose(selected.account_adjustment, [r["mean"] for r in rough])
                np.testing.assert_allclose(
                    selected.rough_account_low,
                    selected.awy_yield - selected.net_loss - [r["maximum"] for r in rough],
                )
                np.testing.assert_allclose(
                    selected.rough_account_high,
                    selected.awy_yield - selected.net_loss - [r["minimum"] for r in rough],
                )
                expected_n = 4
            good = selected.score_eligible.to_numpy()
            if good.sum() != expected_n:
                raise ValueError("Wrong number of available station observations")
            computed = scores(selected.predicted.to_numpy(), selected.observed.to_numpy(), good)
            comparisons.append(
                dict(
                    landcover=product,
                    account=account,
                    year=year,
                    interpretation="rough_accounts" if year == 2018 else "reused_year_conditional",
                    **computed,
                )
            )
            np.testing.assert_allclose(
                selected.error[good], (selected.predicted - selected.observed)[good]
            )
            np.testing.assert_allclose(
                selected.relative_error_pct[good],
                ((selected.predicted / selected.observed - 1) * 100)[good],
            )
            for i, station in enumerate(STATIONS):
                row = selected.loc[station]
                previous = selected.iloc[i - 1] if i else None
                predicted_delta = row.predicted - (previous.predicted if i else 0)
                observed_delta = row.observed - (previous.observed if i else 0)
                increments.append(
                    dict(
                        landcover=product,
                        account=account,
                        year=year,
                        upstream=STATIONS[i - 1] if i else "源头",
                        downstream=station,
                        predicted_increment=predicted_delta,
                        observed_increment=observed_delta,
                        incremental_error=predicted_delta - observed_delta,
                        cumulative_error=row.error,
                    )
                )
            for name, value in zip(
                [*[f"Z{i}" for i in range(1, 8)], "Kc_scale", "L"], fit["parameters"], strict=True
            ):
                parameter_rows.append(
                    dict(
                        landcover=product,
                        account=account,
                        validation_year=year,
                        parameter=name,
                        value=value,
                        observations=fit["observations"],
                    )
                )
        unsupported = frame[~frame.station.isin(STATIONS)]
        if unsupported.predicted.notna().any() or unsupported.score_eligible.any():
            raise ValueError("Unsupported managed predictions were invented")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output / "all_station_results.csv", index=False)
    pd.DataFrame(increments).to_csv(output / "reach_increment_errors.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output / "parameter_comparison.csv", index=False)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output / "aggregate_scores.csv", index=False)
    tables = {}
    for year in (2018, 2023):
        rows = combined[
            (combined.year == year)
            & (combined.account == "legacy")
            & combined.station.isin(STATIONS)
        ]
        fine, cop = [
            rows[rows.landcover == p].set_index("station").loc[STATIONS]
            for p in ("fine", "copernicus")
        ]
        table = pd.DataFrame(
            {
                "station": STATIONS,
                "observed": fine.observed.to_numpy(),
                "fine_predicted": fine.predicted.to_numpy(),
                "fine_error": fine.error.to_numpy(),
                "fine_relative_error_pct": fine.relative_error_pct.to_numpy(),
                "copernicus_predicted": cop.predicted.to_numpy(),
                "copernicus_error": cop.error.to_numpy(),
                "copernicus_relative_error_pct": cop.relative_error_pct.to_numpy(),
            }
        )
        if year == 2018:
            for product, group in (("fine", fine), ("copernicus", cop)):
                table[f"{product}_low"] = group.rough_account_low.to_numpy()
                table[f"{product}_high"] = group.rough_account_high.to_numpy()
        table.to_csv(output / f"station_comparison_{year}.csv", index=False)
        tables[year] = table
        print(year, table.to_string(index=False))
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(7)
    for col, year in enumerate((2018, 2023)):
        table = tables[year]
        axes[0, col].plot(x, table.observed, "ko", label="实测")
        for product, label, color, offset in (
            ("fine", "精细地类", "#3d73a2", -0.06),
            ("copernicus", "Copernicus", "#b67736", 0.06),
        ):
            pred = table[f"{product}_predicted"]
            if year == 2018:
                err = np.vstack([pred - table[f"{product}_low"], table[f"{product}_high"] - pred])
                axes[0, col].errorbar(
                    x + offset, pred, yerr=err, fmt="o-", color=color, label=label, capsize=3
                )
            else:
                axes[0, col].plot(x + offset, pred, "o-", color=color, label=label)
            axes[1, col].plot(
                x + offset, table[f"{product}_relative_error_pct"], "o-", color=color, label=label
            )
        axes[0, col].set(
            title="2013—2017 率定 → 2018 粗略检查"
            if year == 2018
            else "2013—2022 可用资料率定 → 2023 检验",
            ylabel="年水量（亿 m³）",
        )
        axes[1, col].set(ylabel="模拟相对实测的偏差（%）")
        axes[1, col].axhline(0, color="gray", linewidth=1)
        for ax in axes[:, col]:
            ax.set_xticks(x, STATIONS)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle(
        "七区模型逐站结果：2018 使用历史账户情景，误差棒不是置信区间\n"
        "2023 使用当年条件账户；该年此前已检验过，未按本次结果调参"
    )
    fig.tight_layout()
    fig.savefig(output / "station_validation.png", dpi=180)
    fig.savefig(output / "station_validation.pdf")
    plt.close(fig)
    markdown = [
        "# 用户指定年份方案的逐站结果",
        "",
        "水量单位：亿 m³；正误差表示高估，负误差表示低估。默认原年度账户，行业账户敏感性另存。",
        "",
    ]
    for year in (2018, 2023):
        markdown += [
            f"## {year} 年",
            "",
            "| 测站 | 实测 | 精细模拟 | 精细偏差 | Copernicus 模拟 | Copernicus 偏差 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in tables[year].itertuples():
            markdown.append(
                f"| {row.station} | {formatted(row.observed)} | {formatted(row.fine_predicted)} | "
                f"{formatted(row.fine_relative_error_pct)}% | "
                f"{formatted(row.copernicus_predicted)} | "
                f"{formatted(row.copernicus_relative_error_pct)}% |".replace("—%", "—")
            )
        markdown.append("")
    markdown += [
        "2018 的预测是历史账户均值情景，范围见 CSV 和图，不是补齐后的实测账户。",
        "唐乃亥、下河沿、石嘴山、高村只有 AWY 累计产水，缺乏同边界管理账户，未虚构调整后径流。",
        "all_station_results.csv 保留全部 11 站；"
        "reach_increment_errors.csv 区分累计误差和区间新增误差。",
        "",
        "![逐站结果](station_validation.png)",
    ]
    (output / "STATION_RESULTS.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    write_json(
        output / "summary.json",
        {
            "comparisons": comparisons,
            **runtime,
            "source_fingerprints_checked": len(hashes),
            "rows_checked": len(combined),
            "rough_accounts_past_only": True,
            "parameters_exactly_reused": True,
            "training_2018_not_imputed": True,
            "2023_reused_not_blind": True,
            "postprocessor": fingerprint(Path(__file__)),
        },
    )


if __name__ == "__main__":
    main()
