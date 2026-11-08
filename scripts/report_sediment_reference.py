"""Compare climatological hillslope delivery with gauged sediment, without fitting."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from ecologyhydro.cache import fingerprint
from ecologyhydro.simulation import write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdr-run", type=Path, required=True)
    parser.add_argument(
        "--observations", type=Path, default=Path("data/Hydrology/输沙量2011-2024.csv")
    )
    args = parser.parse_args()
    model_path = args.sdr_run / "station_sediment.csv"
    model = pd.read_csv(model_path).set_index("station")
    observed = pd.read_csv(args.observations, encoding="utf-8-sig")
    years = [str(year) for year in range(2011, 2025)]
    output = args.sdr_run / "reference_comparison"
    output.mkdir(exist_ok=True)
    rows = []
    matched = []
    for _, row in observed.iterrows():
        name = row["测站"]
        values = pd.to_numeric(row[years], errors="raise").to_numpy(dtype=float)
        if np.any(np.isfinite(values) & (values < 0)):
            raise ValueError(f"Negative observed sediment: {name}")
        baseline = (
            model.loc[name, "hillslope_delivery_100million_t_yr"] if name in model.index else None
        )
        rows.append(
            {
                "station": name,
                "sdr_climatological_hillslope_1e8_t_yr": baseline,
                "observed_mean_2011_2024_1e8_t_yr": float(np.nanmean(values)),
                "observed_min_1e8_t_yr": float(np.nanmin(values)),
                "observed_max_1e8_t_yr": float(np.nanmax(values)),
                "observed_valid_years": int(np.isfinite(values).sum()),
                **{f"observed_{year}_1e8_t_yr": row[str(year)] for year in (2022, 2023, 2024)},
                "status": "different_processes_and_period_reference_only"
                if baseline is not None
                else "no_sdr_station_catchment",
            }
        )
        if baseline is not None:
            matched.append((name, values, baseline))
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(4, 2, figsize=(13, 13), layout="constrained")
    for axis, (name, values, baseline) in zip(axes.flat, matched, strict=False):
        axis.plot(range(2011, 2025), values, "o-", color="#1565a2", label="测站实测输沙量")
        axis.axhline(baseline, color="#c16525", linestyle="--", label="SDR 年均坡面入河量")
        axis.set(title=name, ylabel="亿吨/年", xticks=[2011, 2015, 2019, 2024], ylim=(0, None))
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(matched) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=9)
    figure.suptitle(
        "参考对照：实测年输沙量与固定气候基准下的坡面入河泥沙\n"
        "虚线不是测站逐年预测；两者过程及时间基准不同",
        fontsize=14,
    )
    figure.savefig(output / "reference_comparison.png", dpi=160)
    plt.close(figure)
    lines = [
        "# 测站实测输沙量与 SDR 的参考对照",
        "",
        "单位：亿吨/年。实测为 2011—2024；SDR 为 1991—2020 年均 R + 当前精细图。",
        "没有率定或修正 SDR，没有生成测站输沙预测；两列不能用于直接认定模型精度。",
        "",
        "| 测站 | SDR 坡面入河量 | 实测2011—2024均值 | 实测2022 | 实测2023 | 实测2024 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        baseline = row["sdr_climatological_hillslope_1e8_t_yr"]
        label = "未划分控制流域" if baseline is None else f"{baseline:.4f}"
        lines.append(
            f"| {row['station']} | {label} | "
            f"{row['observed_mean_2011_2024_1e8_t_yr']:.4f} | "
            f"{row['observed_2022_1e8_t_yr']:.4f} | "
            f"{row['observed_2023_1e8_t_yr']:.4f} | "
            f"{row['observed_2024_1e8_t_yr']:.4f} |"
        )
    lines += [
        "",
        "![实测年际变化与SDR基准](reference_comparison.png)",
        "",
        "7 站可匹配；潼关、小浪底、艾山不借用其他站控制流域。",
        "下一步所需过程、资料和验证边界见 "
        "[测站输沙建模说明](../../../../docs/STATION_SEDIMENT.md)。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output / "manifest.json",
        {
            "sources": [
                fingerprint(model_path),
                fingerprint(args.observations),
                fingerprint(Path(__file__)),
            ],
            "observation_unit": "100million_t/yr_confirmed_by_user",
            "fitting": False,
            "station_sediment_predictions_created": False,
            "matched_stations": len(matched),
            "observed_stations": len(rows),
        },
    )
    print(output)


if __name__ == "__main__":
    main()
