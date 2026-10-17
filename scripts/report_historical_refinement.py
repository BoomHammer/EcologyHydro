"""Audit and explain a sealed historical refinement without selecting new models."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from experiment_separated_selection import check_seal, context
from experiment_shared_parameters import load_samples
from report_independent_calibration import LABELS, markdown_table

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.comparison import accounts
from ecologyhydro.fine_focus_routing import balance_ledger
from ecologyhydro.long_record import REACH_MAP, STATIONS
from ecologyhydro.simulation import write_json


def information_limits(output, recipe):
    """Describe clipping in the sealed model; this diagnostic never changes parameters."""
    models = read_json(output / "models.json")
    rows = []
    for product in recipe["products"]:
        data = load_samples(Path(recipe["samples"]), product)
        vegetation = data["veg"] == 1
        first = (data["groups"] // 11 == 0) & vegetation
        weight = data["weights"]
        for source in (product, "common"):
            x = np.asarray(models[source]["parameters"])
            scale = x[7 + np.clip(data["response_group"], 0, 4)]
            capped_kc = data["kc"] * scale >= 1.3
            capped_omega = 1.25 + x[REACH_MAP[data["zone"]]] * data["awc"] / data["p"] >= 5
            rows.append(
                dict(
                    product=product,
                    parameter_source=source,
                    vegetation_area_kc_capped_fraction=float(
                        weight[first & capped_kc].sum() / weight[first].sum()
                    ),
                    vegetation_area_year_omega_capped_fraction=float(
                        weight[vegetation & capped_omega].sum() / weight[vegetation].sum()
                    ),
                    interpretation="weighted_sample_clipping_diagnosis_not_accuracy_attribution",
                )
            )
    write_json(output / "information_limits.json", rows)


def audit_water(output, recipe, config):
    models = read_json(output / "models.json")
    frame = pd.read_csv(output / "all_full_predictions.csv")
    target, _, _, consumed, stored = accounts(config)
    years = config["forcing_years"]
    ledger_rows, maximum, compared = [], 0.0, 0
    for product in recipe["products"]:
        for source, model in models.items():
            natural = np.array(
                [read_json(output / f"natural_{product}_{year}.json")[source] for year in years]
            )
            parameters = model["parameters"]
            ledger = balance_ledger(
                natural, target, consumed, stored, np.r_[np.full(7, parameters[14]), parameters[13]]
            )
            maximum = max(maximum, float(np.nanmax(np.abs(ledger["closure"]))))
            part = frame[(frame["product"] == product) & (frame.parameter_source == source)]
            for row in part.itertuples():
                yi, si = years.index(row.year), STATIONS.index(row.station)
                np.testing.assert_allclose(row.predicted, ledger["predicted"][yi, si], atol=1e-9)
                np.testing.assert_equal(row.observed, target[yi, si])
                compared += 1
            assert np.all(ledger["loss"][:, [0, 1, 3, 4, 5, 6]] == 0)
            for yi, year in enumerate(years):
                for si, station in enumerate(STATIONS):
                    ledger_rows.append(
                        dict(
                            product=product,
                            parameter_source=source,
                            year=year,
                            station=station,
                            **{key: float(values[yi, si]) for key, values in ledger.items()},
                        )
                    )
    pd.DataFrame(ledger_rows).to_csv(output / "water_balance.csv", index=False)
    write_json(
        output / "water_balance_audit.json",
        dict(
            passed=True,
            compared_target_predictions=compared,
            expected_target_predictions=420,
            max_absolute_closure_1e8_m3=maximum,
            loss_deducted_once=True,
            interpretation="algebraic_closure_not_independent_process_validation",
        ),
    )
    if compared != 420:
        raise ValueError("Incomplete comparison panel")


def report(output):
    output, recipe, config = context(output)
    check_seal(output)
    audit_water(output, recipe, config)
    information_limits(output, recipe)
    records = read_json(output / "training_candidate_records.json")
    cv = []
    for branch in ("independent", "common"):
        for mode in recipe["candidates"]:
            rows = [r for r in records if r["branch"] == branch and r["mode"] == mode]
            success = all(r["success"] for r in rows)
            errors = np.concatenate([r["relative_errors"] for r in rows])
            cv.append(
                dict(
                    branch=branch,
                    candidate=mode,
                    all_converged=success,
                    cv_error_pct=float(100 * np.sqrt(np.mean(errors**2))) if success else np.nan,
                )
            )
    pd.DataFrame(cv).to_csv(output / "candidate_scores.csv", index=False)
    comparison, paired = [], []
    for branch in ("independent", "common"):
        new = pd.read_csv(output / f"{branch}_table1.csv", index_col=0)
        old = pd.read_csv(Path(recipe["baseline"]) / f"{branch}_table1.csv", index_col=0)
        for product in new.index:
            for metric in new.columns:
                comparison.append(
                    dict(
                        branch=branch,
                        product=product,
                        metric=metric,
                        baseline_pct=float(old.loc[product, metric]),
                        current_pct=float(new.loc[product, metric]),
                        change_percentage_points=float(
                            new.loc[product, metric] - old.loc[product, metric]
                        ),
                    )
                )
        predictions = pd.read_csv(output / f"{branch}_predictions.csv")
        errors = predictions.assign(error=(predictions.predicted / predictions.observed - 1).abs())
        errors = errors.pivot(index=["year", "station"], columns="product", values="error")
        for product in recipe["products"][1:]:
            paired.append(
                dict(
                    branch=branch,
                    comparator=product,
                    fine_wins=int((errors.fine < errors[product]).sum()),
                    ties=int((errors.fine == errors[product]).sum()),
                    total=21,
                )
            )
    pd.DataFrame(comparison).to_csv(output / "baseline_comparison.csv", index=False)
    write_json(output / "pairwise_wins.json", paired)
    models = read_json(output / "models.json")
    write_json(
        output / "parameter_diagnostics.json",
        {
            source: dict(
                z_near_lower=int(np.sum(np.array(model["parameters"][:7]) < 1.001)),
                z_near_upper=int(np.sum(np.array(model["parameters"][:7]) > 29.999)),
                loss_1e8_m3_per_year=model["parameters"][13],
                retention=model["parameters"][14],
            )
            for source, model in models.items()
        },
    )
    baseline = pd.read_csv(Path(recipe["baseline"]) / "independent_table1.csv", index_col=0)
    current = pd.read_csv(output / "independent_table1.csv", index_col=0)
    fine = pd.DataFrame(
        {"原结果": baseline.loc[LABELS["fine"]], "新结果": current.loc[LABELS["fine"]]}
    )
    fine["变化（百分点）"] = fine["新结果"] - fine["原结果"]
    goals = read_json(output / "goals.json")
    outcome = goals["independent"]
    branch_summary = pd.DataFrame(
        {
            key: {
                "原整体误差（%）": value["baseline_fine_overall_error_pct"],
                "新整体误差（%）": value["fine_overall_error_pct"],
                "站年严格最优数/21": value["station_year_wins"],
                "汇总严格最优数/4": value["aggregate_best_count"],
            }
            for key, value in goals.items()
        }
    ).T.rename(index={"independent": "四图独立率定", "common": "四图共用参数"})
    passed = all(
        outcome[key] for key in ("overall_improved", "majority_station_years", "majority_aggregate")
    )
    cv_table = pd.DataFrame(cv).pivot(index="candidate", columns="branch", values="cv_error_pct")
    limits = read_json(output / "information_limits.json")
    clipped = next(
        row["vegetation_area_kc_capped_fraction"]
        for row in limits
        if row["product"] == row["parameter_source"] == "fine"
    )
    old_outer = read_json(Path(recipe["baseline"]) / "outer_diagnostic.json")
    new_outer = read_json(output / "outer_diagnostic.json")
    outer_table = pd.DataFrame(
        {
            "原后期前推误差": {key: value["error_pct"] for key, value in old_outer.items()},
            "新后期前推误差": {key: value["error_pct"] for key, value in new_outer.items()},
        }
    )
    conclusion = (
        f"本轮{'达到' if passed else '未达到'}多数最优目标。独立率定精细图 "
        f"{outcome['station_year_wins']}/21 站年严格最优，年度及整体 "
        f"{outcome['aggregate_best_count']}/4 项严格最优。"
        "候选只按历史期选定，目标期结果没有用于改选。"
    )
    lines = [
        "# 历史选型改进结果与审计",
        "",
        conclusion,
        "",
        markdown_table(branch_summary),
        "",
        "共用参数下精细图整体误差降低，但最优站年数从原 7/21 降至 5/21。"
        "独立率定整体误差略升，最优站年数仍为 7/21。两组均未达到多数最优目标。",
        "",
        markdown_table(fine),
        "",
        "## 全部候选的历史留出期相对 RMSE（%）",
        "",
        markdown_table(cv_table),
        "",
        "时间加权仅影响拟合，各候选的历史评价指标均未加权。"
        "本次根据已有结果发起改进，属于回顾性方法开发，不能称独立盲测。",
        "",
        "## 早期选型、后期前推的辅助检查（%）",
        "",
        markdown_table(outer_table),
        "",
        "只有一个后期窗口，历史合并误差改善不保证后期表现改善。",
        "",
        "## 信息表达与参数限制",
        "",
        f"加权样本中，精细图约 {100 * clipped:.2f}% 的植被面积 Kc 达到既有 1.3 上限。"
        "截断会压缩部分原始 Kc 差异，但根深、土壤和空间分布仍可能不同，"
        "不能据此归因全部误差，也不能据此取消物理约束。"
        "详细比例见 information_limits.json，参数边界见 parameter_diagnostics.json。",
        "",
        "## 水量与数值核对",
        "",
        "420 条目标期预测与逐区水量账户复核；净损失只在一个区间扣除。"
        "全像元结果和高分辨率抽样误差见 quadrature.json。代数闭合不代表过程已获独立验证。",
        "",
        "[完整三张表](three_tables.html)；[水量审计](water_balance_audit.json)；"
        "[相对原结果变化](baseline_comparison.csv)；[逐对胜负](pairwise_wins.json)。",
        "",
        "方法与文献见 [实验说明](../../../docs/HISTORICAL_REFINEMENT.md)。",
    ]
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    path = output / "three_tables.html"
    html = path.read_text(encoding="utf-8")
    marker = '<section id="refinement-audit">'
    if marker in html:
        html = html.split(marker)[0]
    html += marker + "<h2>相对原结果的变化与审计</h2><p>" + conclusion + "</p>"
    html += branch_summary.to_html(float_format=lambda value: f"{value:.6f}")
    html += fine.to_html(float_format=lambda value: f"{value:.6f}")
    html += "<h3>全部候选的历史评价（%）</h3>" + cv_table.to_html(float_format=lambda v: f"{v:.6f}")
    html += '<p><a href="analysis.md">详细审计与比较</a></p></section>'
    path.write_text(html, encoding="utf-8")
    write_json(
        output / "report_sources.json",
        [fingerprint(Path(__file__)), fingerprint(Path("src/ecologyhydro/fine_focus_routing.py"))],
    )
    print(conclusion)
    print(fine.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    report(parser.parse_args().run)
