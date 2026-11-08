"""Publish measured-versus-modeled charts from a completed updated-map workflow."""

import argparse
import html
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.simulation import write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    "fine": "精细分类",
    "fine_level1": "精细分类的大类",
    "copernicus": "Copernicus",
    "worldcover": "WorldCover",
}


def save(figure, output, name):
    for suffix in ("png", "svg"):
        figure.savefig(output / f"{name}.{suffix}", dpi=170, bbox_inches="tight")
    plt.close(figure)


def link(output, path, title):
    relative = Path(os.path.relpath(path, output)).as_posix()
    return f'<a href="{html.escape(relative)}">{html.escape(title)}</a>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--water", type=Path, required=True)
    parser.add_argument("--sediment", type=Path, required=True)
    parser.add_argument("--annual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    docs = Path(os.path.relpath(Path("docs").resolve(), args.water.resolve())).as_posix()
    for name in ("report.md", "analysis.md", "three_tables.html"):
        path = args.water / name
        contents = path.read_text(encoding="utf-8")
        if f"{docs}/" not in contents:
            path.write_text(contents.replace("../../../docs/", f"{docs}/"), encoding="utf-8")
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )
    checks = {
        "water_balance": read_json(args.water / "water_balance_audit.json"),
        "quadrature": read_json(args.water / "quadrature.json"),
        "integrity": read_json(args.water / "integrity_check.json"),
        "annual": read_json(args.annual / "summary.json"),
        "annual_raster": read_json(args.annual / "virtual_product_check.json"),
    }
    if not all(checks[key]["passed"] for key in ("water_balance", "quadrature", "integrity")):
        raise ValueError("Water verification failed")
    if checks["annual"]["status"] != "complete":
        raise ValueError("Annual sediment calculation is incomplete")
    if checks["annual_raster"]["status"] != "passed":
        raise ValueError("Annual sediment raster verification failed")
    parallel = args.water / "parallel_years"
    if parallel.exists():
        records = read_json(parallel / "completed.json")
        for record in records:
            name = f"natural_fine_level1_{record['year']}.json"
            extra, primary = read_json(parallel / name), read_json(args.water / name)
            for source, values in extra.items():
                np.testing.assert_array_equal(values, primary[source])
        checks["parallel_years"] = {"passed": True, "years_verified": len(records)}
    ledger = pd.read_csv(args.water / "water_balance.csv")
    fine = ledger[(ledger["product"] == "fine") & (ledger.parameter_source == "fine")]
    fine.to_csv(args.output / "fine_water_2011_2024.csv", index=False, encoding="utf-8-sig")
    figure, axes = plt.subplots(4, 2, figsize=(13, 14), layout="constrained")
    for axis, (station, data) in zip(axes.flat, fine.groupby("station", sort=False), strict=False):
        axis.axvspan(2021.5, 2024.5, color="#f7ead1", label="回顾性评价期")
        axis.plot(data.year, data.observed, "o-", color="#245c83", label="实测径流")
        axis.plot(data.year, data.predicted, "s--", color="#ba5730", label="核算后测站径流")
        axis.plot(data.year, data.natural, ":", color="#54866a", label="InVEST 产水量")
        axis.set(title=station, ylabel="亿 m³/年", xticks=[2011, 2015, 2019, 2022, 2024])
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    axes.flat[-1].set_visible(False)
    figure.suptitle(
        "新精细分类图：2011—2024 年水量\n历史期率定；2022—2024 未参与本轮选型", fontsize=15
    )
    save(figure, args.output, "water_stations")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    tables = {}
    for axis, branch, title in zip(
        axes, ("independent", "common"), ("各图独立率定", "共用参数换图"), strict=True
    ):
        table = pd.read_csv(args.water / f"{branch}_table1.csv", index_col=0)
        tables[branch] = table
        table.T.plot.bar(
            ax=axis, rot=0, width=0.8, color=["#ba5730", "#d8ad5d", "#54866a", "#245c83"]
        )
        axis.set(title=title, ylabel="相对 RMSE（%）", xlabel="")
        axis.set_xticklabels(["2022", "2023", "2024", "三年整体"])
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("四种地表分类的测站径流误差（7 站；越低越好）", fontsize=15)
    save(figure, args.output, "water_errors")
    comparison = pd.read_csv(args.annual / "observations_comparison.csv")
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained", sharex=True)
    mouth_water = fine[fine.station == "利津"]
    mouth_sediment = comparison[comparison.station == "利津"]
    axes[0].plot(mouth_water.year, mouth_water.observed, "o-", label="实测径流")
    axes[0].plot(mouth_water.year, mouth_water.predicted, "s--", label="核算后测站径流")
    axes[0].axvspan(2021.5, 2024.5, color="#f7ead1", zorder=-1, label="回顾性评价期")
    axes[0].set(ylabel="亿 m³/年", title="水量：测站径流对比")
    axes[1].plot(mouth_sediment.year, mouth_sediment.observed_1e8_t_yr, "o-", label="实测输沙")
    axes[1].plot(
        mouth_sediment.year,
        mouth_sediment.hillslope_delivery_1e8_t_yr,
        "s--",
        label="坡面入河量（未模拟库河过程）",
    )
    axes[1].set(ylabel="亿吨/年", xlabel="年份", title="泥沙：不同过程口径的参考对照")
    for axis in axes:
        axis.legend(fontsize=9)
        axis.grid(alpha=0.2)
        axis.set_xticks([2011, 2013, 2015, 2017, 2019, 2021, 2022, 2023, 2024])
    figure.suptitle("利津｜新精细分类图的水文与泥沙结果", fontsize=16)
    save(figure, args.output, "lijin_water_sediment")
    recent = comparison[comparison.year >= 2022]
    if len(comparison) != 140 or comparison.hillslope_delivery_1e8_t_yr.isna().any():
        raise ValueError("Incomplete ten-station sediment comparison")
    sediment_table = recent.pivot(
        index="station",
        columns="year",
        values=["hillslope_delivery_1e8_t_yr", "observed_1e8_t_yr"],
    )
    sediment_table.columns = [
        f"{year} {'坡面入河' if name.startswith('hillslope') else '实测输沙'}"
        for name, year in sediment_table.columns
    ]
    sediment_table = sediment_table[
        [f"{year} {kind}" for year in (2022, 2023, 2024) for kind in ("坡面入河", "实测输沙")]
    ]
    sediment_table.to_csv(args.output / "sediment_recent.csv", encoding="utf-8-sig")
    water_table = fine[fine.year >= 2022].pivot(
        index="station", columns="year", values=["predicted", "observed"]
    )
    water_table.columns = [
        f"{year} {'模拟径流' if name == 'predicted' else '实测径流'}"
        for name, year in water_table.columns
    ]
    water_table = water_table[
        [f"{year} {kind}" for year in (2022, 2023, 2024) for kind in ("模拟径流", "实测径流")]
    ]
    water_table.to_csv(args.output / "water_recent.csv", encoding="utf-8-sig")
    goals = read_json(args.water / "goals.json")
    preparation = read_json(args.output / "preparation.json")
    inputs = [p for p in preparation["sources"] if Path(p["path"]).suffix != ".py"]
    if inputs != [fingerprint(Path(p["path"])) for p in inputs]:
        raise ValueError("Updated-map inputs changed during execution")
    source = preparation["sources"][0]
    annual_image = Path(os.path.relpath(args.annual / "annual_reference.png", args.output))
    guide = "https://storage.googleapis.com/releases.naturalcapitalproject.org/"
    rows = [
        "<!doctype html><html lang='zh'><meta charset='utf-8'>",
        "<title>更新精细分类图：水文与水沙模拟</title>",
        "<style>body{font-family:Microsoft YaHei,system-ui;max-width:1250px;margin:32px auto;",
        "padding:0 20px;color:#233442;line-height:1.65}img{width:100%;height:auto}",
        "table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #cbd5dc;",
        "padding:6px 10px;text-align:right}.table{overflow:auto}h2{margin-top:36px}</style>",
        "<h1>精细分类图更新后的水文与水沙模拟</h1>",
        f"<p>输入：{html.escape(Path(source['path']).name)}；48 类；250 米计算网格。",
        "模拟期 2011—2024；水量评价 7 站，泥沙汇总 14 个控制流域，实测泥沙对照 10 站。</p>",
        "<p>水量采用 TerraClimate，泥沙年际变化采用现有 CMFD 月降水近似 R；",
        "两者气候来源不同。这是同图下的水量与泥沙并行计算，尚不是河库水沙动力耦合模型。</p>",
        "<p>新图作为固定地类用于所有年份，并非历史逐年地类重建。水量只用历史期重新率定并封存；",
        "过去已查看 2022—2024，故这里称回顾性评价，不称独立盲测。</p>",
        "<img src='lijin_water_sediment.png' alt='利津水文和泥沙概览'>",
        "<h2>水文结果</h2><img src='water_errors.png' alt='四图误差比较'>",
    ]
    for branch, title in (("independent", "各图独立率定"), ("common", "共用参数")):
        result = goals[branch]
        rows += [
            f"<h3>{title}</h3><p>精细分类三年整体相对 RMSE："
            f"{result['fine_overall_error_pct']:.3f}%；"
            f"年度及整体四项中最优 {result['aggregate_best_count']}/4；"
            f"站年最优 {result['station_year_wins']}/21。</p>",
            '<div class="table">'
            + tables[branch].to_html(float_format=lambda x: f"{x:.3f}")
            + "</div>",
        ]
    rows += [
        "<p>误差为各站相对误差的均方根，不是流域总水量的相对误差；平局不算胜出。</p>",
        "<img src='water_stations.png' alt='7站逐年水量'>",
        "<p>折线空缺表示缺少水量账户：2014 年头道拐及其下游受蓄水变化缺测影响；"
        "2018 年缺少管理账户。产水量仍已计算，测站径流不以零值补齐。</p>",
        "<p>产水量经过概念储留、耗水和水库蓄水变化等账户核算后得到测站径流。",
        "有效损失项是残差参数，不是独立观测到的渗漏或耗水；年模型不解析洪峰及日调度。</p>",
        "<h3>2022—2024 站点水量（亿 m³/年）</h3>",
        '<div class="table">' + water_table.to_html(float_format=lambda x: f"{x:.2f}") + "</div>",
        link(
            args.output,
            args.water / "three_tables.html",
            "完整三张表：年度误差、21 站年及 4×4 参数互换",
        ),
        "<h2>水沙结果</h2>",
        "<p><strong>SDR 输出是坡面入河泥沙，"
        "不是包含水库拦沙、沟岸侵蚀和河道冲淤的测站输沙预测。</strong>",
        "实测仅作参考，未用于率定，也未用观测差额补算拦沙。不能据该差值报告预测精度。</p>",
        f'<img src="{annual_image.as_posix()}"',
        " alt='10站坡面入河与实测输沙参考对照'>",
        "<h3>2022—2024 泥沙参考对照（亿吨/年）</h3>",
        '<div class="table">'
        + sediment_table.to_html(float_format=lambda x: f"{x:.4f}")
        + "</div>",
        "<p>年度 R 使用 R基准 × MFI年度 / MFI基准均值 的项目近似，尚未验证年际侵蚀力精度。",
        "新图重算完整官方 SDR；年度结果按固定输入下的 R 线性关系加速，"
        "未重复运行 14 次地形计算。</p>",
        "<h2>核验与数据下载</h2>",
        f"<p>水量闭合最大误差 {checks['water_balance']['max_absolute_closure_1e8_m3']:.3g} 亿 m³；",
        "全像元与抽样误差核验通过，旧结果完整性检查通过。泥沙年度记录 196 条、实测对照 140 条。",
        "数值闭合不是独立精度验证；泥沙无效非河道像元保留，未填零。</p><ul>",
    ]
    for path, title in (
        (args.output / "fine_water_2011_2024.csv", "精细图完整水量账户 CSV"),
        (args.water / "all_full_predictions.csv", "四图与参数互换评价 CSV"),
        (args.sediment / "station_sediment.csv", "SDR 年均基准 CSV"),
        (args.sediment / "invest_sdr/sed_export.tif", "SDR 坡面入河量栅格 TIFF（t/ha/年）"),
        (args.annual / "station_annual_sediment.csv", "2011—2024 年 14 站泥沙 CSV"),
        (args.annual / "rasters/sed_export_2024.vrt", "2024 年坡面入河量 VRT（依赖项目缓存）"),
        (args.annual / "observations_comparison.csv", "泥沙实测参考 CSV"),
        (args.annual / "report.md", "年度泥沙方法与局限"),
        (args.water / "analysis.md", "水文选型及水量审计"),
    ):
        rows.append("<li>" + link(args.output, path, title) + "</li>")
    rows += [
        "</ul><p>PNG 和 SVG 图保存在本报告目录；年度泥沙 VRT 必须连同所引用缓存使用。</p>",
        "<h2>方法来源</h2><ul>",
        f'<li><a href="{guide}invest-userguide/latest/en/annual_water_yield.html">'
        "InVEST Annual Water Yield</a></li>",
        f'<li><a href="{guide}invest-userguide/latest/en/sdr.html">InVEST SDR</a></li>',
        '<li><a href="https://doi.org/10.1038/s41597-023-02246-4">'
        "Li et al. (2023)：C/P 数据参考</a></li>",
        '<li><a href="https://doi.org/10.5194/hess-19-4113-2015">'
        "Yin et al. (2015)：降水分辨率与侵蚀力</a></li>",
        "</ul></html>",
    ]
    (args.output / "report.html").write_text("\n".join(rows), encoding="utf-8")
    write_json(
        args.output / "completed.json",
        {
            "status": "complete",
            "water": str(args.water),
            "sediment": str(args.sediment),
            "annual": str(args.annual),
            "goals": goals,
            "checks": checks,
            "report_sources": [fingerprint(Path(__file__)), source],
        },
    )
    print(args.output / "report.html")


if __name__ == "__main__":
    main()
