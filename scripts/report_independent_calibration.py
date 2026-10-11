"""Build the four-row, three-year station error table from full-pixel predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecologyhydro.baseline import read_json
from ecologyhydro.comparison import load_protocol, metrics, training_mask
from ecologyhydro.long_record import STATIONS
from ecologyhydro.simulation import write_json

LABELS = {
    "fine": "精细分类",
    "fine_level1": "精细分类的大类",
    "copernicus": "Copernicus",
    "worldcover": "WorldCover",
}


def main_table(frame, years, absolute=False):
    """Reject missing/mismatched comparisons; never silently drop an adverse cell."""
    expected = pd.MultiIndex.from_product([years, STATIONS], names=["year", "station"])
    rows, reference = [], None
    if set(frame["product"]) != set(LABELS):
        raise ValueError("Require exactly four landcover representations")
    for product, label in LABELS.items():
        part = frame[frame["product"] == product].set_index(["year", "station"])
        if part.index.has_duplicates or set(part.index) != set(expected):
            raise ValueError("Require identical complete station-year keys")
        part = part.reindex(expected)
        if not part.eligible.all() or not np.isfinite(part[["predicted", "observed"]]).all().all():
            raise ValueError("All main-table cells require valid observations and predictions")
        if (part.observed <= 0).any():
            raise ValueError("Relative errors require positive observations")
        if reference is not None and not np.array_equal(reference, part.observed.to_numpy()):
            raise ValueError("Different observations across landcover rows")
        reference = part.observed.to_numpy()
        error = (part.predicted - part.observed).abs()
        if not absolute:
            error = 100 * error / part.observed
        row = {"地类方案": label}
        for year in years:
            for station in STATIONS:
                row[f"{year}_{station}"] = float(error.loc[(year, station)])
            row[f"{year}_总误差"] = float(np.sqrt(np.mean(error.loc[year] ** 2)))
        rows.append(row)
    return pd.DataFrame(rows).set_index("地类方案")


def markdown_table(table):
    columns = [str(table.index.name or "方案"), *table.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for name, row in table.iterrows():
        lines.append("| " + " | ".join([str(name), *(f"{v:.2f}" for v in row)]) + " |")
    return "\n".join(lines)


def html_table(table, years):
    head = '<tr><th rowspan="2">地类方案</th>'
    head += "".join(f'<th colspan="8">{y} 年</th>' for y in years) + "</tr><tr>"
    head += "".join(f"<th>{s}</th>" for _ in years for s in [*STATIONS, "总误差"]) + "</tr>"
    body = []
    for name, row in table.iterrows():
        cells = []
        for column, value in row.items():
            classes = ["total"] if column.endswith("总误差") else []
            if np.isclose(value, table[column].min(), atol=1e-9, rtol=0):
                classes.append("best")
            cells.append(f'<td class="{" ".join(classes)}">{value:.2f}</td>')
        body.append(f"<tr><th>{name}</th>{''.join(cells)}</tr>")
    return (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>四种地类各自率定：三年误差表</title><style>
body{font-family:system-ui,sans-serif;margin:24px;color:#172b3a;background:#f6f8fa}
.scroll{overflow:auto;background:white;border:1px solid #dce3e8}
table{border-collapse:separate;border-spacing:0;white-space:nowrap;font-variant-numeric:tabular-nums}
th,td{padding:12px;border-right:1px solid #e0e5e8;border-bottom:1px solid #e0e5e8;text-align:right}
thead th{background:#e7edf3;text-align:center}
tbody th{position:sticky;left:0;background:#eef2f6;text-align:left}
.total{border-right:3px solid #9baebd;background:#f2f6fa;font-weight:bold}
.best{color:#096b43;background:#dff3e9;font-weight:bold}p{max-width:1000px;line-height:1.7}
</style><h1>四种地类各自率定：2022—2024 年误差表</h1>
<p>单位：%。各站为绝对相对误差；年度总误差为七站相对误差的均方根。
绿色表示该列实际最低值，越低越好。四行分别率定、训练期内部选型，最终采用官方 AWY 方程全像元计算。
三年已被开发查看，属于探索性复核；不参与本轮率定或选型。</p>
<div class="scroll"><table><thead>"""
        + head
        + "</thead><tbody>"
        + "".join(body)
        + """</tbody></table></div>
<p><a href="main_table.csv">下载主表 CSV</a> ·
<a href="main_table_absolute.csv">亿 m³ 绝对误差表</a> ·
<a href="full_predictions.csv">模拟与实测明细</a> · <a href="report.md">实验报告</a></p></html>"""
    )


def report(output):
    recipe = read_json(output / "experiment.json")
    selection = read_json(output / "selection.json")
    config = load_protocol(output / "execution_protocol.yaml")
    years = recipe["validation_years"]
    frame = pd.concat([pd.read_csv(output / f"full_{p}.csv") for p in LABELS], ignore_index=True)
    table = main_table(frame, years)
    table.to_csv(output / "main_table.csv", encoding="utf-8-sig")
    main_table(frame, years, absolute=True).to_csv(
        output / "main_table_absolute.csv", encoding="utf-8-sig"
    )
    frame["signed_error_1e8m3"] = frame.predicted - frame.observed
    frame["absolute_error_1e8m3"] = frame.signed_error_1e8m3.abs()
    frame["absolute_relative_error_pct"] = 100 * frame.absolute_error_1e8m3 / frame.observed
    frame.to_csv(output / "full_predictions.csv", index=False, encoding="utf-8-sig")
    (output / "main_table.html").write_text(html_table(table, years), encoding="utf-8")
    summaries = []
    training_years = np.asarray(config["forcing_years"])[training_mask(config)]
    for product in LABELS:
        part = frame[frame["product"] == product]
        sample = pd.read_csv(output / f"sample_predictions_{product}.csv")
        training = sample[sample.year.isin(training_years)]
        summaries.append(
            dict(
                product=product,
                selected_model=selection[product]["mode"],
                training_relative_rmse_pct=metrics(
                    training.predicted, training.observed, training.eligible
                )["relative_rmse_pct"],
                cv_relative_rmse_pct=selection[product]["cv_relative_rmse_pct"],
                **metrics(part.predicted, part.observed, part.eligible),
            )
        )
    summary = pd.DataFrame(summaries).set_index("product")
    summary.to_csv(output / "summary_metrics.csv")
    errors = frame.pivot(
        index=["year", "station"], columns="product", values="absolute_relative_error_pct"
    )
    wins = {p: int((errors.fine < errors[p]).sum()) for p in LABELS if p != "fine"}
    wins["all_three"] = int(errors.drop(columns="fine").gt(errors.fine, axis=0).all(axis=1).sum())
    yearly = table[[f"{year}_总误差" for year in years]].copy()
    yearly["三年整体误差"] = summary.loc[list(LABELS), "relative_rmse_pct"].to_numpy()
    checks = {p: read_json(output / f"quadrature_{p}.json") for p in LABELS}
    if not all(check["passed"] for check in checks.values()):
        raise ValueError("Unverified quadrature: cannot label experiment complete")
    outcome = dict(
        fine_lower_overall_than_all=bool(
            (
                summary.relative_rmse_pct.drop("fine") > summary.loc["fine", "relative_rmse_pct"]
            ).all()
        ),
        fine_wins_station_years=wins,
        total_station_years=len(errors),
        full_pixel_checks=checks,
        validation_used_for_selection=False,
    )
    write_json(output / "outcome.json", outcome)
    content = [
        "# 四种地类各自率定：三年误差主表",
        "",
        "四种方案分别率定，按各自训练期分块交叉验证选择模型。三年为探索性复核，未参与本轮率定与选型。",
        "主表由官方 AWY 方程全像元计算；数值越低越好。单位：%。",
        "",
        "## 年度及三年整体误差",
        "",
        markdown_table(yearly),
        "",
        "## 完整四行主表",
        "",
        "各站为绝对相对误差，年度总误差为七站相对误差均方根。",
        "",
        "[横向查看 HTML](main_table.html) · [主表 CSV](main_table.csv) · "
        "[亿 m³ 误差表](main_table_absolute.csv)",
        "",
        markdown_table(table),
        "",
        "## 选型与训练精度",
        "",
        "训练误差来自 32768/分区样本；交叉验证用于选型，不是未经选择偏差影响的最终性能估计。",
    ]
    for product, row in summary.iterrows():
        content.append(
            f"- {LABELS[product]}：{row.selected_model}；"
            f"训练 {row.training_relative_rmse_pct:.3f}%，"
            f"训练期分块交叉验证 {row.cv_relative_rmse_pct:.3f}%。"
        )
    content += [
        "",
        "## 实际优势与限制",
        "",
        f"精细分类三年整体误差是否低于其余三种：{outcome['fine_lower_overall_than_all']}。",
        f"21 个站年中，分别胜大类 {wins['fine_level1']}、Copernicus {wins['copernicus']}、"
        f"WorldCover {wins['worldcover']}；同时胜三者 {wins['all_three']}。",
        "不能用整体优势代替逐站年优势，也不能把精细分类未胜出的格子隐藏。",
        "本轮没有用户指定的绝对精度验收阈值，不自行宣布精度已足够。",
        "一级类采用不依赖径流的全域面积混合；水账与站控流域存在近似，仍影响绝对精度。",
        "",
        "## 可复核产物",
        "",
        "- cv_metrics.csv / cv_predictions.csv：全部预设候选及逐折留出预测。",
        "- selection.json / freeze.json：各行训练期选型与冻结记录。",
        "- fit_*.json：独立参数、边界、收敛和初值记录。",
        "- full_predictions.csv：三年模拟、实测、符号误差和绝对误差。",
        "- quadrature_*.json / compression_*.json：全像元与样本求积核对。",
        "- sources.json / executed_*.py：输入指纹和执行代码。",
        "",
        "方法、参数来源与限制见仓库 docs/INDEPENDENT_CALIBRATION.md。",
    ]
    (output / "report.md").write_text("\n".join(content) + "\n", encoding="utf-8")
    print("MAIN ANNUAL TABLE", yearly.to_string(), flush=True)
    print("OUTCOME", outcome, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    report(parser.parse_args().output)
