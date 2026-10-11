"""Select targeted fine-only changes on training folds, then verify all pixels."""

import shutil
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from calibrate_regional_water import block_data, open_inputs
from experiment_independent_calibration import rows_for, run_tasks, sample_response
from experiment_shared_parameters import load_samples
from report_independent_calibration import markdown_table
from run_landcover_comparison import input_paths, worker_setup
from verify_class_response import official_depth

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.class_response import group_for
from ecologyhydro.comparison import accounts, metrics, training_mask
from ecologyhydro.fine_priority import (
    fit_head,
    predict_candidate,
    rainfall_factors,
    regional_delivery,
)
from ecologyhydro.long_record import ENDPOINTS, REACH_MAP
from ecologyhydro.shared_biophysical import load_registry, parameter_table
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import windows


def context(output):
    output = Path(output)
    recipe = read_json(output / "fine_protocol.json")
    config = worker_setup(output / "execution_protocol.yaml")
    return output, recipe, config


def candidate_parameters(output, recipe, name, mode):
    local = Path(recipe["local_run"])
    seed = read_json(local / f"local_fit_fine_{name}.json")["parameters"]
    lower = (
        read_json(local / f"reach_fit_fine_{name}.json")["parameters"][6]
        if "lower" in mode
        else None
    )
    head = None
    if mode.startswith("headwater"):
        kind = "rain" if "rain" in mode else "original"
        head = read_json(output / f"head_{kind}_{name}.json")["parameters"]
    return seed, lower, head


def fit_fold(output_string, name, held, final_mode=None):
    output, recipe, config = context(output_string)
    data = load_samples(Path(recipe["baseline"]) / "samples", "fine")
    target, adjustment, eligible, _, _ = accounts(config)
    years = config["forcing_years"]
    keep = training_mask(config) & ~np.isin(years, held)
    climate = pd.read_csv(output / "climate_and_bounds.csv")
    seed = read_json(Path(recipe["local_run"]) / f"local_fit_fine_{name}.json")["parameters"]
    kinds = (
        ["original", "rain"]
        if final_mode is None
        else (
            ["rain" if "rain" in final_mode else "original"]
            if final_mode.startswith("headwater")
            else []
        )
    )
    for kind in kinds:
        factors = rainfall_factors(climate, years, keep, kind == "rain")
        fit = fit_head(data, target, adjustment, eligible, keep, seed, factors, recipe)
        fit["rainfall_factors"] = factors.tolist()
        write_json(output / f"head_{kind}_{name}.json", fit)
        print("HEAD FIT", name, kind, fit["parameters"], flush=True)
    if final_mode is not None:
        return
    rows = []
    for mode in recipe["candidates"]:
        seed, lower, head = candidate_parameters(output, recipe, name, mode)
        factors = rainfall_factors(climate, years, keep, "rain" in mode)
        predicted = predict_candidate(data, seed, lower, head, factors, len(years)) - adjustment
        rows += rows_for(
            config, "fine", mode, name, predicted, target, eligible, np.isin(years, held)
        )
    pd.DataFrame(rows).to_csv(output / f"cv_{name}.csv", index=False)


def full_year(output_string, year):
    output, recipe, config = context(output_string)
    frozen = read_json(output / "freeze.json")
    x = np.asarray(frozen["seed"]).copy()
    head, lower = frozen["head"], frozen["lower_z"]
    if lower is not None:
        x[6] = lower
    if head is not None:
        x[0] = head[0]
    registry = load_registry(output / "biophysical.yaml")
    table = parameter_table(registry, "fine", "detail")
    lut = np.full(256, -1, int)
    for row in table:
        lut[row["lucode"]] = group_for(registry, row["profile"])
    total = np.zeros(11)
    factor = frozen["rainfall_factors"][config["forcing_years"].index(year)]
    with ExitStack() as stack:
        datasets = open_inputs(stack, input_paths(config, "fine", year))
        gt = datasets["zone"].GetGeoTransform()
        volume = abs(gt[1] * gt[5] - gt[2] * gt[4]) / 1e11
        for window in windows(datasets["zone"]):
            data, _ = block_data(datasets, window, table)
            data["response_group"] = lut[data["code"].astype(int)]
            upper = REACH_MAP[data["zone"]] == 0
            if head is not None:
                data["p"][upper] *= factor
                data["kc"][upper & (data["veg"] == 1)] *= head[1]
            depth = official_depth(x, data)
            total += np.bincount(data["zone"], weights=depth * volume, minlength=11)
    write_json(
        output / f"natural_fine_{year}.json", dict(natural=total.cumsum()[ENDPOINTS].tolist())
    )
    print("FULL FINE", year, flush=True)


def publish(output, recipe, config, selected):
    previous = Path(recipe["previous_evaluation"])
    old = pd.read_csv(previous / "full_fine.csv")
    new = pd.read_csv(output / "full_fine.csv")
    frames, summaries, table_rows = [], [], []
    for label, frame in [("原精细图", old), ("本轮精细图", new)]:
        frame = frame.copy()
        frame["方案"] = label
        frame["误差百分比"] = (frame.predicted / frame.observed - 1) * 100
        frames.append(frame)
        summaries.append(
            dict(方案=label, **metrics(frame.predicted, frame.observed, frame.eligible))
        )
        row = {"方案": label}
        for year, part in frame.groupby("year", sort=True):
            for item in part.to_dict("records"):
                row[f"{year}_{item['station']}"] = abs(item["误差百分比"])
            row[f"{year}_总误差"] = float(np.sqrt(np.mean(part["误差百分比"] ** 2)))
        table_rows.append(row)
    summary = pd.DataFrame(summaries).set_index("方案")
    table = pd.DataFrame(table_rows).set_index("方案")
    table.to_csv(output / "comparison_table.csv", encoding="utf-8-sig")
    summary.to_csv(output / "summary.csv", encoding="utf-8-sig")
    pd.concat(frames).to_csv(output / "station_errors.csv", index=False, encoding="utf-8-sig")
    annual = table[[f"{y}_总误差" for y in recipe["validation_years"]]].copy()
    annual["三年整体"] = summary.relative_rmse_pct
    cv = pd.read_csv(output / "cv_metrics.csv").set_index("mode")
    content = [
        "# 精细分类自身精度专项结果",
        "",
        f"训练期选中候选：`{selected}`。其他三种地类未重率定，不作新的产品排名。",
        "所有三年均已用于开发诊断，属于探索性复核；参数和候选按训练期留出误差选择。",
        "",
        "## 年度及整体误差（%）",
        "",
        markdown_table(annual),
        "",
        "## 完整逐站表",
        "",
        markdown_table(table),
        "",
        "## 训练期候选与前推误差",
        "",
        markdown_table(cv),
        "",
        "贵得以上降水代理使用公报统计区均值，边界与贵得汇水区不同；若该候选入选，改善仍是有条件的输入敏感性结果。",
        "地表耗水、水库蓄变和原始径流不变；非植被属性不改成植被。区域储留执行显式水量闭合。",
        "协议及依据见 docs/FINE_PRIORITY.md；freeze.json 保存完整参数和降水因子，"
        "sources.json 保存源文件指纹。",
    ]
    (output / "report.md").write_text("\n".join(content) + "\n", encoding="utf-8")
    html = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        "<title>精细分类误差改进</title><style>body{font-family:system-ui;margin:30px}"
        "table{border-collapse:collapse;white-space:nowrap}"
        "td,th{padding:9px;border:1px solid #ccc}.wide{overflow:auto}</style>"
        "<h1>精细分类自身精度</h1><p>单位 %；站年为绝对相对误差，"
        "年度为七站相对误差均方根。探索性复核，不作四产品新排名。</p>"
    )
    html += annual.to_html(float_format=lambda v: f"{v:.2f}")
    html += (
        '<h2>三年逐站误差</h2><div class="wide">'
        + table.to_html(float_format=lambda v: f"{v:.2f}")
        + '</div><p><a href="comparison_table.csv">CSV</a> · '
        '<a href="report.md">完整报告</a></p></html>'
    )
    (output / "main_table.html").write_text(html, encoding="utf-8")
    print(annual.to_string(), flush=True)


def main():
    recipe_path = Path("config/fine_priority.yaml")
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    base = Path(recipe["baseline"])
    output = Path(recipe["output_root"]) / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True)
    write_json(output / "fine_protocol.json", recipe)
    for name in (
        "experiment.json",
        "execution_protocol.yaml",
        "biophysical.yaml",
        "sample_recipe.json",
        "hierarchy.json",
    ):
        shutil.copyfile(base / name, output / name)
    shutil.copyfile(recipe["diagnosis"], output / "climate_and_bounds.csv")
    config = worker_setup(output / "execution_protocol.yaml")
    sources = [
        recipe_path,
        Path(__file__),
        Path("src/ecologyhydro/fine_priority.py"),
        Path("src/ecologyhydro/class_response.py"),
        Path(recipe["diagnosis"]),
        base / "sources.json",
        base / "samples/samples_fine.npz",
    ]
    sources += [
        Path(recipe["local_run"]) / f"{kind}_fit_fine_{name}.json"
        for kind in ["local", "reach"]
        for name in ["fold0", "fold1", "fold2", "final"]
    ]
    sources += [
        Path(config["hydrology"][key])
        for key in ["observations", "consumption", "storage", "precipitation_reference"]
    ]
    sources += [Path("data/Hydrology/水资源二级区面积.csv")]
    write_json(output / "sources.json", [fingerprint(p) for p in sources])
    for path in sources:
        if path.suffix == ".py":
            shutil.copyfile(path, output / f"executed_{path.name}")
    print("OUTPUT", output, flush=True)
    try:
        folds = read_json(base / "experiment.json")["folds"]
        run_tasks(
            fit_fold,
            [(str(output), f"fold{i}", h) for i, h in enumerate(folds)],
            config,
            output,
            "cv",
        )
        cv = pd.concat([pd.read_csv(output / f"cv_fold{i}.csv") for i in range(3)])
        cv.to_csv(output / "cv_predictions.csv", index=False)
        scores = []
        for mode, part in cv.groupby("mode"):
            if len(part) != 56 or part.duplicated(["year", "station"]).any():
                raise ValueError("Incomplete CV candidate")
            forward = part[part.fit == "fold2"]
            scores.append(
                dict(
                    mode=mode,
                    cv=metrics(part.predicted, part.observed, part.eligible)["relative_rmse_pct"],
                    forward=metrics(forward.predicted, forward.observed, forward.eligible)[
                        "relative_rmse_pct"
                    ],
                )
            )
        scores = pd.DataFrame(scores)
        scores.to_csv(output / "cv_metrics.csv", index=False)
        selected = scores.loc[scores.cv.idxmin(), "mode"]
        write_json(output / "selection.json", dict(selected=selected, rule=recipe["selection"]))
        print("SELECTION", selected, scores.to_string(index=False), flush=True)
        fit_fold(str(output), "final", [], selected)
        seed, lower, head = candidate_parameters(output, recipe, "final", selected)
        years = config["forcing_years"]
        factors = rainfall_factors(
            pd.read_csv(output / "climate_and_bounds.csv"),
            years,
            training_mask(config),
            "rain" in selected,
        )
        write_json(
            output / "freeze.json",
            dict(
                seed=seed,
                lower_z=lower,
                head=head,
                rainfall_factors=factors.tolist(),
                selected=selected,
                status="frozen_before_candidate_review",
            ),
        )
        target, adjustment, eligible, _, _ = accounts(config)
        data = sample_response(output, "fine", compressed=False)
        sampled = predict_candidate(data, seed, lower, head, factors, len(years)) - adjustment
        pd.DataFrame(
            rows_for(
                config,
                "fine",
                selected,
                "final",
                sampled,
                target,
                eligible,
                np.ones(len(years), bool),
            )
        ).to_csv(output / "sample_predictions.csv", index=False)
        del data
        if selected == "baseline":
            shutil.copyfile(
                Path(recipe["previous_evaluation"]) / "full_fine.csv", output / "full_fine.csv"
            )
            write_json(
                output / "quadrature.json",
                read_json(Path(recipe["previous_evaluation"]) / "quadrature_fine.json"),
            )
        else:
            run_tasks(full_year, [(str(output), y) for y in years], config, output, "full")
            natural = np.array(
                [read_json(output / f"natural_fine_{y}.json")["natural"] for y in years]
            )
            retention = np.full(7, seed[14])
            if head is not None:
                retention[0] = head[2]
            delivery, change = regional_delivery(natural, retention)
            prediction = delivery - adjustment
            prediction[:, 2:] -= seed[13]
            review = np.isin(years, recipe["validation_years"])
            rows = rows_for(config, "fine", selected, "final", prediction, target, eligible, review)
            pd.DataFrame(rows).to_csv(output / "full_fine.csv", index=False)
            mask = eligible & review[:, None]
            difference = float(np.max(np.abs(prediction - sampled)[mask] / target[mask]))
            write_json(
                output / "quadrature.json",
                dict(max_fraction_observed=difference, passed=difference <= 0.01),
            )
            if difference > 0.01:
                raise ValueError("Full-pixel accuracy check failed")
            np.savez(
                output / "regional_storage.npz",
                natural=natural,
                delivered=delivery,
                change=change,
                retention=retention,
            )
        publish(output, recipe, config, selected)
        saved = read_json(output / "sources.json")
        if saved != [fingerprint(Path(item["path"])) for item in saved]:
            raise ValueError("Experiment sources changed during execution")
        write_json(output / "status.json", dict(status="complete", fine_only=True))
    except Exception as error:
        write_json(output / "status.json", dict(status="failed", error=repr(error)))
        raise


if __name__ == "__main__":
    main()
