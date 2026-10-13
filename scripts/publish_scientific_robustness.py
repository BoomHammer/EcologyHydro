"""Audit both adaptive searches and publish a simpler goal-preserving development model."""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from experiment_fine_focus_routing import cached_natural, compile_results, parameters, prepare
from experiment_fine_priority import full_year
from experiment_independent_calibration import rows_for, run_tasks
from report_fine_focus import publish
from report_independent_calibration import LABELS, markdown_table
from run_landcover_comparison import worker_setup

from ecologyhydro.accuracy_goals import PRODUCTS
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.comparison import accounts, metrics
from ecologyhydro.fine_focus_goals import assess_focus
from ecologyhydro.routing_robustness import fit, forward
from ecologyhydro.simulation import write_json


def select(output, recipe):
    frames = []
    for screen in recipe["screens"]:
        part = pd.read_csv(Path(screen) / "candidate_metrics.csv")
        part["screen"] = screen
        frames.append(part)
    candidates = pd.concat(frames, ignore_index=True)
    candidates.to_csv(output / "candidate_metrics.csv", index=False)
    fine = candidates[candidates["product"].eq("fine")]
    passed = fine[fine.aggregate_all_best.eq(True) & fine.station_year_40_percent_best.eq(True)]
    if passed.empty:
        raise ValueError("No simpler candidate preserves both primary goals")
    ranked = passed.sort_values(["cv", "n_free", "target_error", "name"])
    ranked.to_csv(output / "accepted_candidates.csv", index=False)
    chosen = ranked.iloc[0]
    fitted = read_json(Path(chosen.screen) / f"fit_fine_{chosen['name']}.json")
    frozen = read_json(Path(recipe["reference"]) / "freeze_fine.json")
    frozen["retention"] = fitted["parameters"][:7]
    frozen["seed"][13] = fitted["parameters"][7]
    frozen["mode"] = chosen["name"]
    write_json(output / "current_parameters.json", frozen)
    write_json(output / "selected_fit.json", fitted)
    write_json(
        output / "selection.json",
        dict(
            selected=chosen["name"],
            screen=chosen.screen,
            spec=fitted["spec"],
            development_target_fit=True,
            adaptive_search=True,
            independent_validation=False,
            selection=recipe["selection"],
        ),
    )
    years = worker_setup(output / "execution_protocol.yaml")["forcing_years"]
    write_json(
        output / "freeze.json",
        dict(
            seed=frozen["seed"],
            lower_z=None,
            head=[frozen["seed"][0], frozen["head_multiplier"], frozen["retention"][0]],
            rainfall_factors=[1.0] * len(years),
        ),
    )
    previous_cv = pd.read_csv(Path(recipe["accepted"]) / "cv_predictions.csv")
    current_cv = pd.read_csv(Path(chosen.screen) / "cv_fine.csv")
    spec = fitted["spec"]
    mode = f"{spec['sharing']}_{spec['ridge']:g}"
    current_cv = current_cv[current_cv["mode"].eq(mode)].copy()
    current_cv["mode"] = "routing_refit"
    cv = pd.concat([previous_cv[previous_cv["mode"].eq("baseline")], current_cv])
    cv.to_csv(output / "cv_predictions.csv", index=False)
    scores = []
    for name, part in cv.groupby("mode"):
        future = part[part.fit.eq("fold2")]
        scores.append(
            dict(
                mode=name,
                cv=metrics(part.predicted, part.observed, part.eligible)["relative_rmse_pct"],
                forward=metrics(future.predicted, future.observed, future.eligible)[
                    "relative_rmse_pct"
                ],
            )
        )
    pd.DataFrame(scores).to_csv(output / "cv_metrics.csv", index=False)
    return frozen, candidates


def equal_budget(output, recipe, candidates):
    predictions = pd.concat(
        [
            pd.read_csv(Path(screen) / f"predictions_{product}.csv").assign(screen=screen)
            for screen in recipe["screens"]
            for product in PRODUCTS
        ],
        ignore_index=True,
    )
    # No objective comparisons across different weights or penalty strengths.
    # All products use the identical raw target-error selection rule.
    selected, selected_models, selections, comparisons = [], {}, [], []
    for product in PRODUCTS:
        ranked = candidates[candidates["product"].eq(product)].sort_values(
            ["target_error", "n_free", "name"]
        )
        chosen = ranked.iloc[0]
        selected.append(
            predictions[
                predictions["product"].eq(product)
                & predictions["mode"].eq(chosen["name"])
                & predictions.screen.eq(chosen.screen)
            ]
        )
        fitted = read_json(Path(chosen.screen) / f"fit_{product}_{chosen['name']}.json")
        model = read_json(Path(recipe["reference"]) / f"freeze_{product}.json")
        model["retention"], model["seed"][13] = fitted["parameters"][:7], fitted["parameters"][7]
        model["mode"] = chosen["name"]
        selected_models[product] = model
        selections.append(dict(product=product, candidate=chosen["name"], screen=chosen.screen))
    for (screen, name), part in predictions.groupby(["screen", "mode"]):
        if set(part["product"]) != set(PRODUCTS):
            raise ValueError("Unequal candidate grid")
        outcome, _, _ = assess_focus(part)
        comparisons.append(dict(screen=screen, candidate=name, **outcome))
    fair = pd.concat(selected)
    fair.to_csv(output / "equal_budget_predictions.csv", index=False)
    write_json(output / "equal_budget_models.json", selected_models)
    write_json(output / "equal_budget_selections.json", selections)
    outcome, scores, detail = assess_focus(fair)
    write_json(output / "equal_budget_goals.json", outcome)
    scores.T.rename(index=LABELS).to_csv(output / "equal_budget_annual.csv", encoding="utf-8-sig")
    detail.to_csv(output / "equal_budget_station_years.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparisons).to_csv(output / "equal_budget_all_candidates.csv", index=False)
    return outcome, scores.T.rename(index=LABELS), pd.DataFrame(comparisons)


def sensitivity(output, recipe, config, frozen):
    years = np.array(config["forcing_years"])
    natural = cached_natural(Path(recipe["reference"]), "fine", "fine", years)
    target, adjustment, eligible, _, _ = accounts(config)
    keep = np.isin(years, recipe["validation_years"])
    spec = read_json(output / "selection.json")["spec"]
    prior = parameters(read_json(Path(recipe["reference"]) / "freeze_fine.json"))
    # Sensitivity is conditional on the selected architecture and frozen AWY parameters.
    # Retain all three year-deletion refits; these are not independent validation.
    weights = keep / keep.sum()
    if spec["training_weight"] != 0:
        raise ValueError("Sensitivity weights must match the selected training objective")
    profile, deleted, prediction_rows = [], [], []
    for loss in sorted(set(np.r_[np.arange(0, 101, 5), frozen["seed"][13]])):
        result = fit(natural, target, adjustment, eligible, weights, prior, spec, fixed_loss=loss)
        profile.append(
            dict(
                loss=loss,
                data_objective=result["data_objective"],
                error_pct=100 * np.sqrt(result["data_objective"]),
                parameters=result["parameters"],
            )
        )
    for year in recipe["validation_years"]:
        fit_weights = weights.copy()
        fit_weights[years == year] = 0
        result = fit(natural, target, adjustment, eligible, fit_weights, prior, spec)
        predicted = forward(natural, result["parameters"]) - adjustment
        error = metrics(predicted, target, eligible & (years == year)[:, None])["relative_rmse_pct"]
        deleted.append(dict(omitted_year=year, omitted_error_pct=error, **result))
        prediction_rows += rows_for(
            config,
            "fine",
            "year_deletion_sensitivity",
            str(year),
            predicted,
            target,
            eligible,
            years == year,
        )
    pd.DataFrame(profile).to_csv(output / "loss_profile.csv", index=False)
    write_json(output / "year_deletion_fits.json", deleted)
    pd.DataFrame(prediction_rows).to_csv(output / "year_deletion_predictions.csv", index=False)
    return deleted


def scientific_report(output, recipe, candidates, outcome, fair, comparisons, deleted):
    fit_result = read_json(output / "selected_fit.json")
    spec = fit_result["spec"]
    cv = pd.read_csv(output / "cv_metrics.csv").set_index("mode")
    old_cv = pd.read_csv(Path(recipe["accepted"]) / "cv_metrics.csv").set_index("mode")
    history = pd.DataFrame(
        [
            dict(model="原历史基线", **cv.loc["baseline"].to_dict()),
            dict(model="上一版八参数", **old_cv.loc["routing_refit"].to_dict()),
            dict(model="本版三参数", **cv.loc["routing_refit"].to_dict()),
        ]
    ).set_index("model")
    lines = [
        "# 科学稳健性诊断（开发期，不是独立验证）",
        "",
        "主目标保持表一 4/4、表二至少 9/21，表三为辅助诊断。旧达标结果完整保留。",
        "本报告记录不能由开发期排名回答的科学问题；不利对照不会替换或隐藏。",
        "",
        f"选中 `{spec['name']}`：路由自由参数从 8 降为 {fit_result['n_free']}；"
        "前三个增量区间共享一个储留系数，其余四个共享另一个，另拟合一个 LTG 有效净损失。",
        "这些是本轮新增调参维度，不代表整个 AWY 模型只有三个参数。分区结构来自两轮探索，"
        "也属于模型选择，不能用 3/21 当作已控制过拟合的证据。",
        "",
        "## 历史折重拟合（%）",
        "",
        markdown_table(history),
        "",
        "全部折分别使用各自原产水参数并重拟合路由；最终模型仍直接拟合 2022—2024。"
        "只有 fold2 为前推；其他折包含留出期之后的训练信息。CV 用于选型后也不是无偏测试。",
        "",
        "## 对等追加路由搜索",
        "",
        f"四图各 {len(candidates) // 4} 个候选、每次三个初值，相同边界、时间权重、"
        "正则强度及区域共享结构。两轮网格完整保留，第二轮为自适应细化。"
        "每图按原始三年误差最小选择；不跨不同正则项比较优化目标值。",
        "这消除了本次追加搜索预算差异，但继承的产水率定历史不同，"
        "仍不等于从零开始的完全对称实验，也不构成独立验证。",
        "",
        markdown_table(fair),
        "",
        f"对等选优后：精细分类表一 {outcome['aggregate_best_count']}/4，"
        f"表二 {outcome['station_year_wins']}/21。所有共同候选中，"
        f"表一最大 {int(comparisons.aggregate_best_count.max())}/4；"
        f"表二最大 {int(comparisons.station_year_wins.max())}/21（两项最大值可能来自不同候选）。",
        "因此不能将冻结对照下达标解释为精细地类本身优越。"
        "完整对照见 `equal_budget_all_candidates.csv`、`equal_budget_station_years.csv`。",
        "",
        "## 可辨识性与稳定性",
        "",
        f"本轮按参数边界缩放的数据雅可比条件数 {fit_result['condition_number']:.6f}。"
        "它只诊断当前位置的线性敏感性；不证明物理可辨识，也不替代过程观测。",
        "`loss_profile.csv` 固定净损失后重拟合储留，覆盖 0—100 亿 m³/年；"
        "这是条件损失剖面，没有给出置信区间。",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    dict(
                        omitted_year=d["omitted_year"],
                        error_pct=d["omitted_error_pct"],
                        loss=d["parameters"][7],
                        upper_r=d["parameters"][0],
                        lower_r=d["parameters"][3],
                    )
                    for d in deleted
                ]
            ).set_index("omitted_year")
        ),
        "",
        "删年拟合只是开发后敏感性检查，结构已看过所有三年，不能改称独立留一年验证。"
        "七站为嵌套流域，同年误差相关，三个目标年份不足以建立稳健显著性结论。",
        "",
        "## 仍待外部数据解决",
        "",
        "2011—2024 已全部参与开发或诊断。本次未取得同口径完整七站的新独立资料、"
        "耗水/蓄变账户及配套气象；已封存模型和待验证协议，尚未完成独立验证。",
        "有效净损失不是实测渗漏/耗水；水量闭合是代数检查。独立蒸散、库容、"
        "引调水或区间水量观测尚未约束其归因；不能因此宣称 CMIP6/换地类外推已验证。",
        "参考文献和复现步骤见 `docs/SCIENTIFIC_ROBUSTNESS.md`。",
        "",
    ]
    (output / "scientific_review.md").write_text("\n".join(lines), encoding="utf-8")
    note = "\n科学稳健性及对等追加搜索：[完整诊断](../scientific_review.md)。\n"
    with (output / "report/report.md").open("a", encoding="utf-8") as stream:
        stream.write(note)
    html_path = output / "report/three_tables.html"
    html = html_path.read_text(encoding="utf-8").replace(
        "</html>",
        '<p><a href="../scientific_review.md">科学稳健性与对等追加搜索诊断</a></p></html>',
    )
    html_path.write_text(html, encoding="utf-8")


def seal(output, config):
    # Necessary checks for a future holdout; this is not an automatic certificate of independence.
    from datetime import UTC, datetime

    paths = [
        output / name
        for name in (
            "current_parameters.json",
            "equal_budget_models.json",
            "execution_protocol.yaml",
            "biophysical.yaml",
            "selection.json",
            "fine_protocol.json",
        )
    ]
    paths += sorted(Path("src/ecologyhydro").glob("*.py"))
    paths += sorted(Path("scripts").glob("*.py"))
    write_json(
        output / "independent_validation_contract.json",
        dict(
            status="pending_external_data",
            registered_at_utc=datetime.now(UTC).isoformat(),
            development_years=config["forcing_years"],
            eligible_years=[2025, 2026, 2027],
            stations=config["evaluation"]["managed_stations"],
            products=list(PRODUCTS),
            required_years="all_three_registered_years_no_outcome_based_stopping",
            model_files=[fingerprint(p) for p in paths],
            required_observation_access="after_registration_with_documented_provenance",
            scoring="relative_RMSE_and_absolute_relative_station_errors_with_frozen_accounts",
            selection="no_refitting_no_ranking_based_exclusions",
            interpretation="eligibility_checks_are_necessary_not_sufficient_for_independence",
            confirmatory_product_comparison="equal_budget_models_all_four_products",
            management_assumptions="effective_residuals_not_validated_physical_processes",
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("config/scientific_robustness_publish.yaml")
    )
    args = parser.parse_args()
    output = prepare(args.config)
    recipe = read_json(output / "fine_protocol.json")
    config = worker_setup(output / "execution_protocol.yaml")
    extra = [
        Path(__file__),
        Path("src/ecologyhydro/routing_robustness.py"),
        Path("src/ecologyhydro/validation_boundary.py"),
    ]
    for screen in recipe["screens"]:
        extra += [p for p in Path(screen).glob("*") if p.suffix in (".json", ".csv", ".py")]
    frozen_sources = read_json(output / "sources.json") + [fingerprint(p) for p in extra]
    write_json(output / "sources.json", frozen_sources)
    for path in extra[:3]:
        shutil.copyfile(path, output / f"executed_{path.name}")
    try:
        frozen, candidates = select(output, recipe)
        result, fair, comparisons = equal_budget(output, recipe, candidates)
        deleted = sensitivity(output, recipe, config, frozen)
        run_tasks(
            full_year, [(str(output), y) for y in config["forcing_years"]], config, output, "full"
        )
        compile_results(output, recipe, config, frozen)
        goals = read_json(output / "goals.json")
        if not (goals["aggregate_all_best"] and goals["station_year_40_percent_best"]):
            raise ValueError("Full-pixel primary goals failed")
        if frozen_sources != [fingerprint(Path(p["path"])) for p in frozen_sources]:
            raise ValueError("Source changed during publication")
        write_json(
            output / "status.json",
            dict(
                status="complete",
                fine_only=True,
                full_pixels=True,
                comparators_frozen=True,
                development_target_fit=True,
                transfer_status="current",
                water_balance_passed=True,
                independent_validation=False,
                equal_budget_diagnostic_complete=True,
            ),
        )
        publish(output)
        scientific_report(output, recipe, candidates, result, fair, comparisons, deleted)
        seal(output, config)
        print("PUBLISHED", output, flush=True)
    except Exception as error:
        write_json(output / "status.json", dict(status="failed", error=repr(error)))
        raise


if __name__ == "__main__":
    main()
