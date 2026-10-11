"""Always publish the three requested tables, with fixed comparator provenance."""

import argparse
from pathlib import Path

import pandas as pd
from report_independent_calibration import LABELS, main_table, markdown_table

from ecologyhydro.accuracy_goals import assess_transfer
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.fine_focus_goals import assess_focus
from ecologyhydro.simulation import write_json


def publish(run):
    """Write all three tables for a completed experiment, including failed goals."""
    status = read_json(run / "status.json")
    if status.get("status") != "complete" or not status.get("comparators_frozen"):
        raise ValueError("Require completed fine-only comparison")
    recipe = read_json(run / "fine_protocol.json")
    own = pd.read_csv(run / "full_predictions.csv")
    reference = pd.read_csv(Path(recipe["reference"]) / "full_predictions.csv")
    keys = ["product", "year", "station"]
    for product in ("fine_level1", "copernicus", "worldcover"):
        left = own[own["product"] == product].set_index(keys).sort_index()
        right = reference[reference["product"] == product].set_index(keys).sort_index()
        pd.testing.assert_frame_equal(
            left[["predicted", "observed", "eligible"]],
            right[["predicted", "observed", "eligible"]],
        )
    outcome, scores, detail = assess_focus(own)
    transfer_path = run / (
        "transfer_predictions.csv"
        if status["transfer_status"] == "current"
        else "reference_transfer.csv"
    )
    transfer, matrix, _ = assess_transfer(pd.read_csv(transfer_path))
    output = run / "report"
    output.mkdir(exist_ok=True)
    annual = scores.loc[["2022_总误差", "2023_总误差", "2024_总误差", "三年整体"]].T.rename(
        index=LABELS
    )
    stations = main_table(own, [2022, 2023, 2024])
    matrix = matrix.rename(index=LABELS, columns=LABELS)
    matrix.index.name = "参数来源"
    for name, frame in [
        ("table1_annual", annual),
        ("table2_station_years", stations),
        (
            "table3_transfer"
            if status["transfer_status"] == "current"
            else "table3_transfer_reference",
            matrix,
        ),
    ]:
        frame.to_csv(output / f"{name}.csv", encoding="utf-8-sig")
    detail.to_csv(output / "all_accuracy_comparisons.csv", index=False, encoding="utf-8-sig")
    write_json(output / "goals.json", outcome)
    note = "第三表沿用上一轮完整互换矩阵，本轮没有率定其他三图。"
    if status["transfer_status"] == "current":
        note = (
            "第三表使用本轮精细参数；其余参数来源保持冻结。精细来源行已重新传输计算，"
            "产水参数不变的组合复用已验证全像元缓存；互换不重率定。"
        )
    elif status["transfer_status"] == "previous_reference":
        note += "精细参数本轮已变化，该矩阵不代表本轮新参数；辅助展示，不作当前验收。"
    else:
        note += "本轮精细参数也未变化，因此该矩阵仍适用。"
    first = f"表一：{outcome['aggregate_best_count']}/4 项最优，目标为 4/4。"
    second = (
        f"表二：{outcome['station_year_wins']}/21 个站年最优，目标至少 9/21；年度总误差不计数。"
    )
    development = status.get("development_target_fit", False)
    selection_note = (
        "2022—2024 径流直接参与参数拟合和候选排序，属于目标导向的开发期拟合，"
        "不是独立验证，不能据此证明精细地类本身优于其他产品。"
        if development
        else "本轮候选按训练期 CV 选择，2023 径流未进入率定；三年均为已查看的探索性复核。"
    )
    content = [
        "# 精细分类专项：固定三个表",
        "",
        first,
        second,
        "",
        "只率定精细图，另外三行数值完全冻结；这是开发进展对照，非等量调参预算的新正式产品比较。",
        "## 表一：年度与三年整体误差（%）",
        "",
        markdown_table(annual),
        "",
        "## 表二：三年各站误差（%）",
        "",
        markdown_table(stations),
        "",
        "## 表三：参数互换参考（%）",
        "",
        note,
        "",
        markdown_table(matrix),
        "",
        f"互换矩阵有 {transfer['passed_comparisons']}/12 项在 1 个百分点参考界限内（非硬门槛）。",
        selection_note,
        "协议见 docs/FINE_2023_FOCUS.md。",
    ]
    cv = pd.read_csv(run / "cv_metrics.csv").set_index("mode")
    content += ["", "训练期候选指标（不增加主表数量）：", "", cv.to_string()]
    if development:
        changed = pd.read_csv(run / "all_year_changes.csv")
        target_changes = changed[changed.year.isin([2022, 2023, 2024])]
        worse = target_changes[target_changes.change_percentage_points > 0]
        regression_note = (
            f"三年 21 个站年中有 {len(worse)} 项绝对相对误差上升；"
            f"最大上升 {max(0, target_changes.change_percentage_points.max()):.6f} 个百分点。"
        )
        content += [
            "",
            "CV/前推按原训练折分别重拟合路由，使用各折原产水参数；"
            "它们诊断方法稳定性，不验证最终使用 2022—2024 拟合的参数。",
            "",
            "完整记录：[全部候选](../candidate_metrics.csv)、"
            "[全部年份改善和退化](../all_year_changes.csv)、"
            "[逐区间水量账](../water_balance.csv)、[守恒检查](../balance_check.json)。",
            "净损失是原有兰州—头道拐有效残差参数，只在该区间扣除一次；"
            "不是新增公报耗水，也不是实测渗漏。概念储留不重复代表已扣的水库蓄变。",
            "方法、限制和参考文献见 docs/FINE_FOCUS_ROUTING.md。",
            "",
            regression_note,
        ]
    (output / "report.md").write_text("\n".join(content) + "\n", encoding="utf-8")
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>精细分类三个表</title>'
    html += "<style>body{font-family:system-ui;margin:28px}"
    html += "table{border-collapse:collapse;white-space:nowrap}"
    html += "td,th{border:1px solid #ccc;padding:8px}.wide{overflow:auto}</style>"
    html += f"<h1>精细分类专项</h1><p>{first}</p><p>{second}</p>"
    html += "<p>另外三种分类保持已完成结果不变。单位 %，越低越好。</p>"
    html += f"<p><strong>{selection_note}</strong></p>"
    if development:
        baseline_cv, current_cv = cv.loc["baseline"], cv.loc["routing_refit"]
        html += (
            f"<p>历史折重拟合诊断：CV {baseline_cv.cv:.3f}% → {current_cv.cv:.3f}%；"
            f"前推 {baseline_cv.forward:.3f}% → {current_cv.forward:.3f}%。"
            "这些指标不验证最终三年目标拟合参数。</p>"
            f"<p>{regression_note}</p>"
        )
    for title, table in [
        ("表一：年度与三年整体误差", annual),
        ("表二：三年各站误差", stations),
        ("表三：参数互换参考", matrix),
    ]:
        html += f"<h2>{title}</h2>"
        if table is matrix:
            html += f"<p>{note}</p>"
        html += '<div class="wide">' + table.to_html(float_format=lambda v: f"{v:.3f}") + "</div>"
    html += '<p><a href="report.md">完整报告与说明</a></p></html>'
    (output / "three_tables.html").write_text(html, encoding="utf-8")
    write_json(
        output / "sources.json",
        [
            fingerprint(path)
            for path in [
                Path(__file__),
                Path("src/ecologyhydro/fine_focus_goals.py"),
                run / "full_predictions.csv",
                transfer_path,
                run / "status.json",
            ]
        ],
    )
    print(outcome)
    print(annual.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    publish(parser.parse_args().run)


if __name__ == "__main__":
    main()
