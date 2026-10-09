"""Summarize executed M3 runs without treating engineering priors as calibration."""

import csv
import json
from pathlib import Path

from ecologyhydro.config import project_root
from ecologyhydro.simulation import valid_completed


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def effective_run(entry):
    directory = Path(entry["directory"])
    request = read_json(directory / "run.json")
    if request["status"] == "reused":
        directory = Path(request["reused_run_dir"])
    run = read_json(directory / "run.json")
    if not valid_completed(run):
        raise ValueError(f"Run integrity check failed: {directory}")
    return directory, run


def main():
    root = project_root()
    suite_path = root / "project/m3/latest_suite.json"
    suite = read_json(suite_path)
    if suite["status"] != "success":
        raise ValueError("M3 suite has not completed successfully")
    performance, full_results, benchmark_values = [], [], {}
    scope_value = full_value = None
    maximum_balance_error = maximum_official_relative_error = 0.0
    for group in suite["groups"]:
        batch = read_json(group["path"])
        if batch["status"] != "success" or len(batch["completed"]) != len(batch["jobs"]):
            raise ValueError("Missing or failed batch tasks")
        performance.append(
            {
                "label": group["label"],
                "workers": batch["workers"],
                "tasks": len(batch["jobs"]),
                "seconds": batch["elapsed_seconds"],
                "peak_tree_rss_mib": batch["peak_tree_rss_bytes"] / 1024**2,
                "peak_work_directory_mib": batch["peak_run_directory_bytes"] / 1024**2,
                "peak_intermediate_mib": batch["peak_intermediate_bytes"] / 1024**2,
                "batch_path": group["path"],
            }
        )
        for entry in batch["completed"]:
            directory, run = effective_run(entry)
            with (directory / "stations.csv").open(encoding="utf-8-sig", newline="") as stream:
                stations = list(csv.DictReader(stream))
            checks = read_json(directory / "checks.json")
            maximum_balance_error = max(
                maximum_balance_error, checks["maximum_water_balance_error_mm"]
            )
            for record in checks["official_zone_comparisons"]:
                error = record["absolute_difference_m3"] / max(abs(record["pixel_m3"]), 1)
                maximum_official_relative_error = max(maximum_official_relative_error, error)
            if group["label"] == "smoke":
                scope_value = float(stations[0]["natural_yield_m3"])
            if group["label"] == "full":
                full_value = float(stations[0]["natural_yield_m3"])
            if group["label"] in {"full", "full_fine"}:
                full_results.extend(stations)
            if group["label"].startswith("benchmark_"):
                benchmark_values.setdefault(run["year"], []).append(
                    float(stations[0]["natural_yield_m3"])
                )
    if scope_value is None or full_value is None:
        raise ValueError("Missing smoke or full-scale validation")
    relative = abs(full_value - scope_value) / max(abs(scope_value), 1)
    if relative > 1e-6:
        raise ValueError("Small-basin/full-basin station sums disagree")
    concurrency_difference = max(
        (max(values) - min(values) for values in benchmark_values.values()), default=0
    )
    if concurrency_difference > 1:
        raise ValueError("Concurrency changed a station volume by more than 1 m3")
    output = root / "project/m3/report"
    output.mkdir(exist_ok=True)
    for name, rows in (("stations_2019.csv", full_results), ("performance.csv", performance)):
        with (output / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    bench = [row for row in performance if row["label"].startswith("benchmark_")]
    fastest = min(bench, key=lambda row: row["seconds"])["workers"] if bench else None
    summary = {
        "engineering_acceptance": True,
        "scientific_accuracy_validated": False,
        "year": 2019,
        "z": suite["z"],
        "station_rows": len(full_results),
        "small_full_relative_volume_difference": relative,
        "concurrency_maximum_absolute_volume_difference_m3": concurrency_difference,
        "maximum_water_balance_error_mm": maximum_balance_error,
        "maximum_official_volume_relative_difference": maximum_official_relative_error,
        "fastest_tested_subbasin_workers": fastest,
        "production_workers": 2,
        "production_reason": "retain two-task cap until full-basin concurrency is benchmarked",
        "performance": performance,
        "limitations": [
            "Fixed uncalibrated Z and provisional biophysical/soil parameters",
            "Natural water yield compared with regulated observed runoff",
            "Hydrography-conditioned catchments and climate uncertainty",
            "No 2023 validation observations used and no accuracy-based tuning",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# M3 工程试算结果",
        "",
        "2019 年、固定 Z=5；结果未经率定，不用于正式精度结论。",
        "",
        "| 测站 | 地类方案 | 自然产水 亿m³ | 实测径流 亿m³ |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in full_results:
        lines.append(
            f"| {row['station']} | {row['landcover']} | "
            f"{float(row['natural_yield_1e8_m3']):.3f} | {row['observed_1e8_m3']} |"
        )
    lines += ["", "| 任务 | 并发 | 总秒数 | 进程树峰值 MiB |", "| --- | ---: | ---: | ---: |"]
    for row in performance:
        lines.append(
            f"| {row['label']} | {row['workers']} | {row['seconds']:.2f} | "
            f"{row['peak_tree_rss_mib']:.1f} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
