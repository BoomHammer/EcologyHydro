"""Run a bounded, past-only refinement with equal opportunities for all four maps."""

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from experiment_independent_calibration import rows_for, run_tasks
from experiment_separated_selection import check_seal, context, full_product, prepare, publish
from experiment_shared_parameters import load_samples

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.class_response import evaluate
from ecologyhydro.historical_refinement import choose_history, fit_history
from ecologyhydro.simulation import write_json


def prepare_refinement(path):
    output = prepare(path)
    _, recipe, _ = context(output)
    extra = [
        Path(__file__),
        Path("scripts/experiment_separated_selection.py"),
        Path("src/ecologyhydro/historical_refinement.py"),
        Path("docs/HISTORICAL_REFINEMENT.md"),
    ]
    sources = read_json(output / "sources.json")
    sources += [
        fingerprint(p) for p in extra if str(p.resolve()) not in {s["path"] for s in sources}
    ]
    write_json(output / "sources.json", sources)
    for source in extra:
        shutil.copyfile(source, output / f"executed_{source.name}")
    protected = [Path("AGENTS.md")]
    for name in ("baseline", "accepted"):
        protected += [p for p in Path(recipe[name]).rglob("*") if p.is_file()]
    write_json(output / "protected_integrity.json", [fingerprint(p) for p in protected])
    return output


def fit_task(output_string, source, mode, fold_name):
    output, recipe, config = context(output_string)
    path = output / f"fit_{source}_{mode}_{fold_name}.json"
    if path.exists():
        return
    years = np.array(config["forcing_years"])
    fold = next((f for f in recipe["inner_folds"] if f["name"] == fold_name), None)
    train = recipe["final_train"] if fold is None else fold["train"]
    if set(train) & set(recipe["target_years"]):
        raise ValueError("Target years must not enter historical fits")
    if fold is not None and max(train) >= min(fold["held"]):
        raise ValueError("Nonchronological historical fold")
    products = recipe["products"] if source == "common" else [source]
    data = {p: load_samples(Path(recipe["samples"]), p) for p in products}
    with np.load(output / "training_accounts.npz") as archive:
        target, adjustment, eligible = (archive[k] for k in ("target", "adjustment", "eligible"))
    try:
        fitted = fit_history(
            data,
            target,
            adjustment,
            eligible,
            np.isin(years, train),
            recipe["candidates"][mode],
            recipe["max_nfev"],
        )
    except RuntimeError as error:
        write_json(path, dict(success=False, mode=mode, error=str(error)))
        print("NONCONVERGED", source, mode, fold_name, flush=True)
        return
    fitted.update(
        mode=mode,
        candidate=recipe["candidates"][mode],
        source=source,
        training_years=train,
        held_years=[] if fold is None else fold["held"],
    )
    if fold is not None:
        held = np.isin(years, fold["held"])
        rows = []
        for product, sample in data.items():
            pred = evaluate(fitted["parameters"], sample, len(years)) - adjustment
            rows += rows_for(config, product, mode, fold_name, pred, target, eligible, held)
        pd.DataFrame(rows).assign(source=source).to_csv(
            output / f"cv_{source}_{mode}_{fold_name}.csv", index=False
        )
    write_json(path, fitted)
    print("FIT", source, mode, fold_name, flush=True)


def records_for(output, recipe, branch, folds):
    records = []
    sources = ["common"] if branch == "common" else recipe["products"]
    for mode in recipe["candidates"]:
        for source in sources:
            for fold in folds:
                result = read_json(output / f"fit_{source}_{mode}_{fold}.json")
                errors = []
                if result["success"]:
                    frame = pd.read_csv(output / f"cv_{source}_{mode}_{fold}.csv")
                    expected = 7 * len(
                        next(f["held"] for f in recipe["inner_folds"] if f["name"] == fold)
                    )
                    expected *= len(recipe["products"]) if source == "common" else 1
                    if len(frame) != expected or not frame.eligible.all():
                        raise ValueError("Incomplete historical held panel")
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
    return records


def select_and_seal(output):
    output, recipe, config = context(output)
    folds = [f["name"] for f in recipe["inner_folds"]]
    sources = ["common", *recipe["products"]]
    tasks = [
        (str(output), source, mode, fold)
        for fold in folds
        for source in sources
        for mode in recipe["candidates"]
    ]
    run_tasks(fit_task, tasks, config, output, "training_folds")
    selected, diagnostics, records = {}, {}, []
    for branch in ("common", "independent"):
        current = records_for(output, recipe, branch, folds)
        selected[branch] = choose_history(current, recipe["candidates"])
        records += [dict(branch=branch, **row) for row in current]
        outer = recipe["outer_diagnostic"]
        early = [r for r in current if r["fold"] in outer["select_with"]]
        choice = choose_history(early, recipe["candidates"])
        held = [
            r
            for r in current
            if r["mode"] == choice["mode"] and r["fold"] == outer["evaluate_with"]
        ]
        if any(not r["success"] for r in held):
            raise ValueError("Outer diagnostic fit failed")
        errors = np.concatenate([r["relative_errors"] for r in held])
        diagnostics[branch] = dict(
            selected=choice, error_pct=float(100 * np.sqrt(np.mean(errors**2))), retrospective=True
        )
    write_json(output / "training_candidate_records.json", records)
    write_json(output / "selection.json", selected)
    write_json(output / "outer_diagnostic.json", diagnostics)
    tasks = [
        (
            str(output),
            source,
            selected["common" if source == "common" else "independent"]["mode"],
            "final",
        )
        for source in sources
    ]
    run_tasks(fit_task, tasks, config, output, "final_training")
    models = {
        source: read_json(output / f"fit_{source}_{mode}_final.json")
        for _, source, mode, _ in tasks
    }
    if any(not result["success"] for result in models.values()):
        raise ValueError("A final historical fit failed")
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
            "training_candidate_records.json",
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
    write_json(output / "status.json", dict(status="selected_and_sealed", target_evaluated=False))
    print("SEALED BEFORE EVALUATION", selected, flush=True)


def publish_refinement(output):
    publish(output)
    _, recipe, _ = context(output)
    legacy = read_json(output / "goals.json")
    write_json(output / "legacy_goals.json", legacy)
    goals = {}
    for branch, result in legacy.items():
        current = pd.read_csv(output / f"{branch}_table1.csv", index_col=0)
        baseline = pd.read_csv(Path(recipe["baseline"]) / f"{branch}_table1.csv", index_col=0)
        value, old = float(current.iloc[0, -1]), float(baseline.iloc[0, -1])
        goals[branch] = dict(
            aggregate_best_count=result["aggregate_best_count"],
            aggregate_required=recipe["goal_annual"],
            station_year_wins=result["station_year_wins"],
            station_year_required=recipe["goal_station_years"],
            station_year_total=21,
            fine_overall_error_pct=value,
            baseline_fine_overall_error_pct=old,
            overall_improved=value < old,
            majority_station_years=result["station_year_wins"] >= recipe["goal_station_years"],
            majority_aggregate=result["aggregate_best_count"] >= recipe["goal_annual"],
        )
    write_json(output / "goals.json", goals)
    protected = read_json(output / "protected_integrity.json")
    unchanged = protected == [fingerprint(Path(p["path"])) for p in protected]
    write_json(output / "integrity_check.json", dict(passed=unchanged, files=len(protected)))
    if not unchanged:
        raise ValueError("A protected original artifact changed")
    status = read_json(output / "status.json")
    status["fair_goals_passed"] = all(
        goals["independent"][k]
        for k in ("overall_improved", "majority_station_years", "majority_aggregate")
    )
    status["goal_definition"] = "lower_overall_error_and_11_of_21_and_3_of_4"
    write_json(output / "status.json", status)
    for filename in ("report.md", "three_tables.html"):
        path = output / filename
        content = path.read_text(encoding="utf-8").replace(
            "两种候选均从固定初值率定全部自由参数", "五种预先固定候选均从相同初值率定全部自由参数"
        )
        note = (
            "本轮只用历史期选型；多数目标为至少 11/21 站年、辅助指标至少 3/4。"
            "全部失败候选与不利结果保留。方法和参考文献："
            "../../../docs/HISTORICAL_REFINEMENT.md。"
            "新目标结果见 goals.json，旧目标仅见 legacy_goals.json。"
        )
        content += "\n" + (f"<p>{note}</p>" if filename.endswith("html") else note) + "\n"
        path.write_text(content, encoding="utf-8")
    print("CURRENT GOALS", goals, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/historical_refinement.yaml"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--stage", choices=("select", "evaluate", "all"), default="all")
    args = parser.parse_args()
    output = args.run or prepare_refinement(args.config)
    try:
        if args.stage in ("select", "all"):
            if not (output / "selection_seal.json").exists():
                select_and_seal(output)
            check_seal(output)
        if args.stage in ("evaluate", "all"):
            check_seal(output)
            _, recipe, config = context(output)
            run_tasks(
                full_product, [(str(output), p) for p in recipe["products"]], config, output, "full"
            )
            publish_refinement(output)
    except Exception as error:
        write_json(output / "status.json", dict(status="failed", error=repr(error)))
        raise


if __name__ == "__main__":
    main()
