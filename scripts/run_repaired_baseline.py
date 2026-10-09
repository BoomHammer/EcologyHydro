"""Re-run fixed historical scenarios on repaired M2 inputs, without calibration."""

import json
from pathlib import Path

from ecologyhydro.config import load_config, project_root
from ecologyhydro.runtime import configure_threads


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    config = load_config()
    configure_threads(config.resources)
    from ecologyhydro.aggregation import write_csv
    from ecologyhydro.model_inputs import prepare_inputs
    from ecologyhydro.simulation import valid_completed, write_json
    from ecologyhydro.trials import run_group
    from ecologyhydro.water_balance import read_csv

    root = project_root()
    output = root / "project/repairs/soil_routing"
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "m2_index.json"
    index = read_json(root / "project/cache/m2/latest.json")
    if not index.get("closed_lake_acceptance", {}).get("passed"):
        raise ValueError("Repaired closed-lake acceptance is required")
    if not index.get("inland_channel_acceptance", {}).get("passed"):
        raise ValueError("Mapped inland channels must be excluded from mainstem catchments")
    if not index.get("readiness", {}).get("engineering_ready"):
        raise ValueError("Complete input coverage and routing checks are required")
    write_json(index_path, index)
    fine_jobs = [
        job
        for job in read_json(root / "project/m4/rotation/plan.json")["jobs"]
        if job["variant"] == "rotation"
    ]
    cop_job = next(
        job
        for job in read_json(root / "project/m4/sensitivity/plan.json")["jobs"]
        if job["landcover"] == "copernicus" and job["variant"] == "central"
    )
    jobs = []
    for year in config.study.calibration_years:
        fine_job = next(j for j in fine_jobs if j["year"] == year)
        for prior in (fine_job, {**cop_job, "year": year}):
            job = {k: prior[k] for k in ("year", "landcover", "scope", "z", "scenario", "variant")}
            scenario = read_json(job["scenario"])
            if set(scenario["overrides"]) != {"biophysical_table_path"}:
                raise ValueError("Baseline permits only unchanged historical biophysical tables")
            # Serial preparation prevents simultaneous writers of shared cache artifacts.
            prepare_inputs(index_path, config.paths.cache, year, job["landcover"], "full")
            jobs.append(job)
    write_json(
        output / "plan.json",
        {
            "jobs": jobs,
            "policy": "fixed historical tables and Z=5; no fitting",
            "validation_used": False,
        },
    )
    directory, batch = run_group(
        config,
        root / "config.yaml",
        index_path,
        jobs,
        2,
        "repaired_baseline",
        batch_root=output / "runs",
    )
    old = {}
    for name in ("rotation", "sensitivity"):
        summary = read_json(root / f"project/m4/{name}/summary.json")
        for entry in read_json(summary["batch"])["completed"]:
            job = entry["job"]
            if not (
                (name == "rotation" and job["variant"] == "rotation")
                or (
                    name == "sensitivity"
                    and job["landcover"] == "copernicus"
                    and job["variant"] == "central"
                )
            ):
                continue
            run_dir = Path(entry["directory"])
            state = read_json(run_dir / "run.json")
            if state["status"] == "reused":
                run_dir = Path(state["reused_run_dir"])
                state = read_json(run_dir / "run.json")
            if not valid_completed(state):
                raise ValueError("Historical baseline checksum failed")
            for row in read_csv(run_dir / "stations.csv"):
                old[job["year"], job["landcover"], row["station"]] = float(
                    row["natural_yield_1e8_m3"]
                )
    records = []
    for entry in batch["completed"]:
        run_dir = Path(entry["directory"])
        state = read_json(run_dir / "run.json")
        if state["status"] == "reused":
            run_dir = Path(state["reused_run_dir"])
            state = read_json(run_dir / "run.json")
        if not valid_completed(state):
            raise ValueError("Repaired baseline checksum failed")
        job = entry["job"]
        for row in read_csv(run_dir / "stations.csv"):
            before = old.get((job["year"], job["landcover"], row["station"]))
            after = float(row["natural_yield_1e8_m3"])
            records.append(
                {
                    "year": job["year"],
                    "landcover": job["landcover"],
                    "station": row["station"],
                    "before_1e8_m3": before,
                    "after_1e8_m3": after,
                    "observed_1e8_m3": row["observed_1e8_m3"],
                    "change_percent": (after / before - 1) * 100 if before else None,
                    "run_dir": str(run_dir),
                }
            )
    write_csv(output / "station_comparison.csv", records)
    write_json(
        output / "summary.json",
        {
            "batch": str(directory / "batch.json"),
            "status": batch["status"],
            "tasks": len(jobs),
            "station_rows": len(records),
            "elapsed_seconds": batch["elapsed_seconds"],
            "peak_tree_rss_bytes": batch["peak_tree_rss_bytes"],
            "validation_used": False,
            "calibrated": False,
            "historical_comparisons": "fine rotation 2019-2022; Copernicus central 2019",
        },
    )
    print(
        json.dumps(
            [
                r
                for r in records
                if r["year"] == 2019 and r["station"] in ("唐乃亥", "兰州", "利津")
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
