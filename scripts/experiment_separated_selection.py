"""Select and seal on historical folds first; then evaluate both fair controls."""

import argparse
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
from report_independent_calibration import LABELS, main_table, markdown_table
from run_landcover_comparison import input_paths, worker_setup
from verify_class_response import official_depth

from ecologyhydro.accuracy_goals import assess_transfer
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.class_response import collapse_hierarchy, evaluate, group_for
from ecologyhydro.comparison import accounts
from ecologyhydro.fine_focus_goals import assess_focus
from ecologyhydro.long_record import ENDPOINTS
from ecologyhydro.routing_robustness import forward
from ecologyhydro.separated_selection import choose, fit_past
from ecologyhydro.shared_biophysical import load_registry, parameter_table
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import windows


def context(output):
    output = Path(output)
    return (
        output,
        read_json(output / "protocol.json"),
        worker_setup(output / "execution_protocol.yaml"),
    )


def prepare(path):
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    output = Path(recipe["output_root"]) / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True)
    reference = Path(recipe["reference"])
    for name in (
        "experiment.json",
        "execution_protocol.yaml",
        "biophysical.yaml",
        "hierarchy.json",
        "sample_recipe.json",
    ):
        shutil.copyfile(reference / name, output / name)
    write_json(output / "protocol.json", recipe)
    config = worker_setup(output / "execution_protocol.yaml")
    target, adjustment, eligible, _, _ = accounts(config)
    future = np.isin(config["forcing_years"], recipe["target_years"])
    target[future], adjustment[future], eligible[future] = np.nan, np.nan, False
    np.savez(
        output / "training_accounts.npz", target=target, adjustment=adjustment, eligible=eligible
    )
    sources = [path, Path(__file__), Path("src/ecologyhydro/separated_selection.py")]
    sources += [
        Path("src/ecologyhydro") / name
        for name in (
            "class_response.py",
            "regional_transfer.py",
            "routing_robustness.py",
            "shared_biophysical.py",
        )
    ]
    sources += [
        reference / name
        for name in (
            "execution_protocol.yaml",
            "biophysical.yaml",
            "hierarchy.json",
            "sample_recipe.json",
        )
    ]
    sources += [Path(recipe["samples"]) / f"samples_{p}.npz" for p in recipe["products"]]
    sources += [
        Path(config["hydrology"][key]) for key in ("observations", "consumption", "storage")
    ]
    sources += [
        Path(recipe["accepted"]) / name
        for name in (
            "current_parameters.json",
            "full_predictions.csv",
            "goals.json",
        )
    ]
    write_json(output / "sources.json", [fingerprint(p) for p in sources])
    for source in sources:
        if source.suffix == ".py":
            shutil.copyfile(source, output / f"executed_{source.name}")
    print("OUTPUT", output, flush=True)
    return output


def fit_task(output_string, source, mode, fold_name):
    output, recipe, config = context(output_string)
    path = output / f"fit_{source}_{mode}_{fold_name}.json"
    if path.exists():
        return
    years = np.array(config["forcing_years"])
    fold = next((f for f in recipe["inner_folds"] if f["name"] == fold_name), None)
    train = recipe["final_train"] if fold is None else fold["train"]
    products = recipe["products"] if source == "common" else [source]
    data = {p: load_samples(Path(recipe["samples"]), p) for p in products}
    with np.load(output / "training_accounts.npz") as archive:
        target, adjustment, eligible = (
            archive[key] for key in ("target", "adjustment", "eligible")
        )
    try:
        fitted = fit_past(
            data, target, adjustment, eligible, np.isin(years, train), mode, recipe["max_nfev"]
        )
    except RuntimeError as error:
        write_json(path, dict(success=False, mode=mode, error=str(error)))
        print("NONCONVERGED", source, mode, fold_name, flush=True)
        return
    fitted.update(
        source=source, training_years=train, held_years=[] if fold is None else fold["held"]
    )
    write_json(path, fitted)
    if fold is not None:
        held = np.isin(years, fold["held"])
        if max(train) >= min(fold["held"]):
            raise ValueError("Forward fold is not chronological")
        rows = []
        for product, sample in data.items():
            pred = evaluate(fitted["parameters"], sample, len(years)) - adjustment
            rows += rows_for(config, product, mode, fold_name, pred, target, eligible, held)
        pd.DataFrame(rows).assign(source=source).to_csv(
            output / f"cv_{source}_{mode}_{fold_name}.csv", index=False
        )
    print("FIT", source, mode, fold_name, flush=True)


def select_branch(output, recipe, branch, fold_names):
    records = []
    sources = ["common"] if branch == "common" else recipe["products"]
    for mode in recipe["candidates"]:
        for source in sources:
            for fold in fold_names:
                result = read_json(output / f"fit_{source}_{mode}_{fold}.json")
                errors = []
                if result["success"]:
                    frame = pd.read_csv(output / f"cv_{source}_{mode}_{fold}.csv")
                    expected = 7 * len(
                        next(f["held"] for f in recipe["inner_folds"] if f["name"] == fold)
                    )
                    expected *= len(recipe["products"]) if source == "common" else 1
                    if len(frame) != expected or not frame.eligible.all():
                        raise ValueError("Incomplete CV panel")
                    errors = (frame.predicted / frame.observed - 1).tolist()
                records.append(
                    dict(
                        mode=mode,
                        source=source,
                        fold=fold,
                        success=result["success"],
                        relative_errors=errors,
                    )
                )
    return choose(records, recipe["candidates"]), records


def seal_selection(output):
    output, recipe, config = context(output)
    folds = [f["name"] for f in recipe["inner_folds"]]
    tasks = [
        (str(output), source, mode, fold)
        for fold in folds
        for source in ["common", *recipe["products"]]
        for mode in recipe["candidates"]
    ]
    run_tasks(fit_task, tasks, config, output, "training_folds")
    selected, diagnostics, all_records = {}, {}, []
    for branch in ("common", "independent"):
        selected[branch], records = select_branch(output, recipe, branch, folds)
        all_records += [dict(branch=branch, **r) for r in records]
        outer_choice, _ = select_branch(
            output, recipe, branch, recipe["outer_diagnostic"]["select_with"]
        )
        outer_records = [
            r
            for r in records
            if r["mode"] == outer_choice["mode"]
            and r["fold"] == recipe["outer_diagnostic"]["evaluate_with"]
        ]
        if any(not r["success"] for r in outer_records):
            raise ValueError("Outer diagnostic fit failed for the inner-selected candidate")
        error = np.concatenate([r["relative_errors"] for r in outer_records])
        diagnostics[branch] = dict(
            selected=outer_choice,
            error_pct=float(100 * np.sqrt(np.mean(error**2))),
            evaluated_years=[2019, 2020, 2021],
            retrospective=True,
        )
    write_json(output / "training_candidate_records.json", all_records)
    write_json(output / "selection.json", selected)
    write_json(output / "outer_diagnostic.json", diagnostics)
    tasks = [
        (
            str(output),
            source,
            selected["common" if source == "common" else "independent"]["mode"],
            "final",
        )
        for source in ["common", *recipe["products"]]
    ]
    run_tasks(fit_task, tasks, config, output, "final_training")
    models = {}
    for _, source, mode, _ in tasks:
        result = read_json(output / f"fit_{source}_{mode}_final.json")
        if not result["success"]:
            raise ValueError("Selected final fit failed; do not use target errors as a fallback")
        models[source] = result
    write_json(output / "models.json", models)
    sealed = [
        output / name
        for name in (
            "models.json",
            "selection.json",
            "protocol.json",
            "biophysical.yaml",
            "hierarchy.json",
            "execution_protocol.yaml",
            "training_accounts.npz",
        )
    ]
    write_json(
        output / "selection_seal.json",
        dict(
            sealed_at_utc=datetime.now(UTC).isoformat(),
            files=[fingerprint(p) for p in sealed],
            target_years_used_for_new_fits_or_selection=False,
            prior_exposure_acknowledged=True,
        ),
    )
    print("SEALED BEFORE EVALUATION", selected, flush=True)
    write_json(output / "status.json", dict(status="selected_and_sealed", target_evaluated=False))


def check_seal(output):
    for manifest in (
        read_json(output / "sources.json"),
        read_json(output / "selection_seal.json")["files"],
    ):
        if manifest != [fingerprint(Path(item["path"])) for item in manifest]:
            raise ValueError("A sealed parameter, protocol or source changed")


def full_product(output_string, product):
    output, recipe, config = context(output_string)
    models = read_json(output / "models.json")
    registry = load_registry(output / "biophysical.yaml")
    hierarchy = read_json(output / "hierarchy.json")
    labels = {int(k): v for k, v in hierarchy["labels"].items()}
    original = "fine" if product == "fine_level1" else product
    table = parameter_table(registry, original, "detail")
    lut = np.full(256, -1, int)
    for row in table:
        lut[row["lucode"]] = group_for(registry, row["profile"])
    for year in config["forcing_years"]:
        path = output / f"natural_{product}_{year}.json"
        if path.exists():
            continue
        totals = {source: np.zeros(11) for source in models}
        with ExitStack() as stack:
            datasets = open_inputs(stack, input_paths(config, original, year))
            gt = datasets["zone"].GetGeoTransform()
            volume = abs(gt[1] * gt[5] - gt[2] * gt[4]) / 1e11
            for window in windows(datasets["zone"]):
                data, _ = block_data(datasets, window, table)
                data["response_group"] = lut[data["code"].astype(int)]
                data["weights"] = np.full(len(data["p"]), volume)
                if product == "fine_level1":
                    data = collapse_hierarchy(data, labels, hierarchy["components"])
                for source, result in models.items():
                    depth = official_depth(np.array(result["parameters"]), data)
                    totals[source] += np.bincount(
                        data["zone"], weights=depth * data["weights"], minlength=11
                    )
        write_json(
            path, {source: total.cumsum()[ENDPOINTS].tolist() for source, total in totals.items()}
        )
        print("FULL", product, year, flush=True)
    target, adjustment, eligible, _, _ = accounts(config)
    years = config["forcing_years"]
    sample = sample_response(output, product, compressed=False)
    rows, checks = [], []
    for source, model in models.items():
        x = np.array(model["parameters"])
        natural = np.array(
            [read_json(output / f"natural_{product}_{y}.json")[source] for y in years]
        )
        pred = forward(natural, np.r_[np.full(7, x[14]), x[13]]) - adjustment
        approx = evaluate(x, sample, len(years)) - adjustment
        keep = np.isin(years, recipe["target_years"])
        valid = eligible & keep[:, None]
        error = float(np.max(np.abs(approx - pred)[valid] / target[valid]))
        checks.append(
            dict(source=source, product=product, max_fraction_observed=error, passed=error <= 0.01)
        )
        part = pd.DataFrame(
            rows_for(config, product, model["mode"], "final", pred, target, eligible, keep)
        )
        rows.append(part.assign(parameter_source=source))
    pd.concat(rows).to_csv(output / f"full_{product}.csv", index=False)
    write_json(output / f"quadrature_{product}.json", checks)


def publish(output):
    output, recipe, config = context(output)
    check_seal(output)
    frames = pd.concat([pd.read_csv(output / f"full_{p}.csv") for p in recipe["products"]])
    frames.to_csv(output / "all_full_predictions.csv", index=False)
    independent = frames[frames["product"] == frames.parameter_source]
    common = frames[frames.parameter_source == "common"]
    transfer = frames[frames.parameter_source != "common"]
    outcomes, tables = {}, {}
    for name, data in (("independent", independent), ("common", common)):
        outcome, scores, _ = assess_focus(data)
        outcomes[name] = outcome
        annual = scores.loc[
            [f"{y}_总误差" for y in recipe["target_years"]] + ["三年整体"]
        ].T.rename(index=LABELS)
        station = main_table(data, recipe["target_years"])
        annual.to_csv(output / f"{name}_table1.csv", encoding="utf-8-sig")
        station.to_csv(output / f"{name}_table2.csv", encoding="utf-8-sig")
        data.to_csv(output / f"{name}_predictions.csv", index=False)
        tables[name] = (annual, station)
    _, matrix, _ = assess_transfer(transfer)
    matrix = matrix.rename(index=LABELS, columns=LABELS)
    matrix.to_csv(output / "table3_transfer.csv", encoding="utf-8-sig")
    write_json(output / "goals.json", outcomes)
    checks = [c for p in recipe["products"] for c in read_json(output / f"quadrature_{p}.json")]
    write_json(
        output / "quadrature.json", dict(passed=all(c["passed"] for c in checks), products=checks)
    )
    selected = read_json(output / "selection.json")
    lines = [
        "# 先选型封存、再作公平比较",
        "",
        "细类参数库、根深、物候与灌溉处理均未新增。旧达标版本保持不变。",
        "2022—2024 未进入本轮拟合与选型；过去已查看这些年份，结果仍为回顾性评价，不是新盲测。",
        "两种候选均从固定初值率定全部自由参数；没有复用旧的全期拟合参数。",
        "内层按四图平均误差选结构，最后每图一套固定模型；没有按站年拼接。",
        "",
    ]
    html = '<!doctype html><meta charset="utf-8"><title>分离选型与公平比较</title>'
    html += (
        "<style>body{font-family:system-ui;margin:28px}table{border-collapse:collapse}"
        "td,th{border:1px solid #bbb;padding:6px}.wide{overflow:auto}</style>"
    )
    html += (
        "<h1>先选型封存、再作公平比较</h1><p>细类参数库不变。"
        "目标期未参与新拟合；历史已查看，仍为回顾性评价。</p>"
    )
    for name, title in (("independent", "四图对等独立率定"), ("common", "同一参数换图控制")):
        result = outcomes[name]
        note = (
            f"{title}：{result['aggregate_best_count']}/4、{result['station_year_wins']}/21；"
            f"结构 {selected[name]['mode']}。"
        )
        lines += [
            f"## {title}",
            "",
            note,
            "",
            markdown_table(tables[name][0]),
            "",
            markdown_table(tables[name][1]),
            "",
        ]
        html += f"<h2>{title}</h2><p>{note}</p>"
        for table in tables[name]:
            html += (
                '<div class="wide">' + table.to_html(float_format=lambda v: f"{v:.3f}") + "</div>"
            )
    lines += [
        "## 完整互换矩阵",
        "",
        markdown_table(matrix),
        "",
        "全部候选见 training_candidate_records.json；"
        "嵌套外层仅一个前推时段，见 outer_diagnostic.json。",
        "选型封存见 selection_seal.json。抽样与全像元差异见 quadrature.json；"
        "全像元表不拼接旧结果。",
        "同图不同年份、嵌套测站及历史方法开发依赖仍存在，不能把对等预算视为自动证明因果或独立性。",
    ]
    html += "<h2>完整互换矩阵</h2>" + matrix.to_html(float_format=lambda v: f"{v:.3f}")
    html += '<p><a href="report.md">完整说明</a></p>'
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "three_tables.html").write_text(html, encoding="utf-8")
    check_seal(output)
    write_json(
        output / "status.json",
        dict(
            status="complete",
            accepted_baseline_preserved=True,
            fair_goals_passed=outcomes["independent"]["aggregate_all_best"]
            and outcomes["independent"]["station_year_40_percent_best"],
            quadrature_passed=all(c["passed"] for c in checks),
            independent_blind_validation=False,
        ),
    )
    print("FAIR RESULTS", outcomes, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/separated_selection.yaml"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--stage", choices=("select", "evaluate", "all"), default="all")
    args = parser.parse_args()
    output = args.run or prepare(args.config)
    try:
        if args.stage in ("select", "all"):
            if (output / "selection_seal.json").exists():
                check_seal(output)
            else:
                seal_selection(output)
        if args.stage in ("evaluate", "all"):
            check_seal(output)
            _, recipe, config = context(output)
            run_tasks(
                full_product, [(str(output), p) for p in recipe["products"]], config, output, "full"
            )
            publish(output)
    except Exception as error:
        write_json(output / "status.json", dict(status="failed", error=repr(error)))
        raise


if __name__ == "__main__":
    main()
