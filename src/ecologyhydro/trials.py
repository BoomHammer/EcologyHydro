"""M3 engineering suite with bounded process-level parallelism and telemetry."""

import json
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from uuid import uuid4

import psutil

from ecologyhydro.simulation import write_json


def unique_id(label):
    return f"{label}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"


def directory_size(path):
    size = 0
    for item in path.rglob("*"):
        with suppress(OSError):
            if item.is_file():
                size += item.stat().st_size
    return size


def tree_sample(process):
    memory = cpu = 0
    members = [process]
    with suppress(psutil.NoSuchProcess):
        members.extend(process.children(recursive=True))
    for member in members:
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            memory += member.memory_info().rss
            times = member.cpu_times()
            cpu += times.user + times.system
    return memory, cpu


def stop_process(child):
    with suppress(psutil.NoSuchProcess):
        process = psutil.Process(child.pid)
        members = [process, *process.children(recursive=True)]
        for member in reversed(members):
            with suppress(psutil.NoSuchProcess):
                member.kill()
    child.wait()


def run_group(config, config_path, index_path, jobs, workers, label, batch_root=None):
    if workers not in (1, 2, 4) or workers > len(jobs):
        raise ValueError("M3 concurrency must be 1, 2 or 4 and not exceed the task count")
    batch_dir = (batch_root or config.paths.cache.parent / "m3") / unique_id(label)
    batch_dir.mkdir(parents=True)
    queue = list(enumerate(jobs))
    active, completed = [], []
    started = time.monotonic()
    peak_memory = peak_disk = peak_intermediate = 0
    last_disk = 0
    cache_start = directory_size(config.paths.cache / "m3")
    status = "running"
    failure_reason = None
    try:
        while queue or active:
            while queue and len(active) < workers:
                number, job = queue.pop(0)
                directory = batch_dir / f"run_{number:02d}_{uuid4().hex[:8]}"
                directory.mkdir()
                stream = (directory / "console.log").open("w", encoding="utf-8")
                command = [
                    sys.executable,
                    "-m",
                    "ecologyhydro.simulation",
                    "--config",
                    str(config_path),
                    "--index",
                    str(index_path),
                    "--run-dir",
                    str(directory),
                    "--year",
                    str(job["year"]),
                    "--landcover",
                    job["landcover"],
                    "--scope",
                    job["scope"],
                    "--z",
                    str(job["z"]),
                ]
                if job.get("force", False):
                    command.append("--force")
                if job.get("scenario"):
                    command.extend(["--scenario", str(job["scenario"])])
                if job.get("holdout_manifest"):
                    command.extend(["--holdout-manifest", str(job["holdout_manifest"])])
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                active.append(
                    {
                        "child": child,
                        "stream": stream,
                        "directory": directory,
                        "job": job,
                        "started": time.monotonic(),
                        "peak_rss_bytes": 0,
                        "cpu_seconds": 0.0,
                        "peak_cpu_percent": 0.0,
                        "last_cpu": 0.0,
                        "last_sample": time.monotonic(),
                    }
                )
                print(
                    f"Started {job['scope']} {job['landcover']} {job['year']}: {directory}",
                    flush=True,
                )
            for entry in active[:]:
                now = time.monotonic()
                child = entry["child"]
                with suppress(psutil.NoSuchProcess):
                    rss, cpu = tree_sample(psutil.Process(child.pid))
                    entry["peak_rss_bytes"] = max(entry["peak_rss_bytes"], rss)
                    elapsed = now - entry["last_sample"]
                    if elapsed > 0.1:
                        entry["peak_cpu_percent"] = max(
                            entry["peak_cpu_percent"],
                            max(0, cpu - entry["last_cpu"]) / elapsed * 100,
                        )
                        entry["last_cpu"], entry["last_sample"] = cpu, now
                    entry["cpu_seconds"] = max(entry["cpu_seconds"], cpu)
                if now - entry["started"] > config.resources.timeout_hours * 3600:
                    raise TimeoutError(f"M3 task exceeded time budget: {entry['directory']}")
                if child.poll() is not None:
                    entry["stream"].close()
                    result_path = entry["directory"] / "run.json"
                    result = (
                        json.loads(result_path.read_text(encoding="utf-8"))
                        if result_path.exists()
                        else {"status": "failed"}
                    )
                    metrics = {
                        key: entry[key]
                        for key in ("peak_rss_bytes", "cpu_seconds", "peak_cpu_percent")
                    }
                    metrics.update(
                        elapsed_seconds=now - entry["started"],
                        exit_code=child.returncode,
                        status=result["status"],
                        run_directory_bytes=directory_size(entry["directory"]),
                    )
                    metrics["mean_cpu_percent"] = (
                        metrics["cpu_seconds"] / metrics["elapsed_seconds"] * 100
                    )
                    write_json(entry["directory"] / "resources.json", metrics)
                    completed.append(
                        {
                            "directory": str(entry["directory"]),
                            "job": entry["job"],
                            "result": result,
                            "resources": metrics,
                        }
                    )
                    active.remove(entry)
                    print(
                        f"Finished {result['status']} in {metrics['elapsed_seconds']:.1f} s: "
                        f"{entry['directory'].name}",
                        flush=True,
                    )
                    if child.returncode != 0 or result["status"] not in {"success", "reused"}:
                        raise RuntimeError(
                            f"M3 task failed; see {entry['directory'] / 'console.log'}"
                        )
            total_memory, _ = tree_sample(psutil.Process())
            peak_memory = max(peak_memory, total_memory)
            if total_memory > config.resources.memory_budget_gb * 1024**3:
                raise MemoryError("M3 concurrent process tree exceeded memory budget")
            if time.monotonic() - last_disk >= 2:
                peak_disk = max(peak_disk, directory_size(batch_dir))
                intermediate = sum(
                    directory_size(path) for path in batch_dir.glob("run_*/model/intermediate")
                )
                peak_intermediate = max(peak_intermediate, intermediate)
                last_disk = time.monotonic()
            if active:
                time.sleep(0.5)
        status = "success"
    except BaseException as error:
        status = "failed"
        failure_reason = f"{type(error).__name__}: {error}"
        raise
    finally:
        for entry in active:
            stop_process(entry["child"])
            entry["stream"].close()
            write_json(
                entry["directory"] / "termination.json",
                {"status": "terminated", "reason": failure_reason},
            )
        summary = {
            "status": status,
            "failure_reason": failure_reason,
            "workers": workers,
            "jobs": jobs,
            "completed": completed,
            "elapsed_seconds": time.monotonic() - started,
            "peak_tree_rss_bytes": peak_memory,
            "peak_run_directory_bytes": max(peak_disk, directory_size(batch_dir)),
            "peak_intermediate_bytes": peak_intermediate,
            "shared_cache_bytes_before": cache_start,
            "shared_cache_bytes_after": directory_size(config.paths.cache / "m3"),
            "sampling_seconds": 0.5,
            "disk_sampling_seconds": 2,
            "scope": "engineering trials; CPU percentage uses 100% per logical core",
        }
        write_json(batch_dir / "batch.json", summary)
    return batch_dir, summary


def run_suite(config, config_path, index_path, benchmark=True):
    suite = {
        "status": "running",
        "groups": [],
        "z": 5,
        "parameter_policy": "fixed engineering Z=5; no observation-based selection",
    }
    directory = config.paths.cache.parent / "m3"
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        jobs = [
            ({"year": 2019, "landcover": "copernicus", "scope": "tangnaihai", "z": 5}, "smoke"),
            ({"year": 2019, "landcover": "copernicus", "scope": "full", "z": 5}, "full"),
            ({"year": 2019, "landcover": "fine", "scope": "full", "z": 5}, "full_fine"),
            ({"year": 2019, "landcover": "copernicus", "scope": "full", "z": 5}, "reuse"),
        ]
        for job, label in jobs:
            batch, summary = run_group(config, config_path, index_path, [job], 1, label)
            suite["groups"].append(
                {
                    "label": label,
                    "path": str(batch / "batch.json"),
                    "elapsed_seconds": summary["elapsed_seconds"],
                }
            )
            write_json(directory / "latest_suite.json", suite)
        if benchmark:
            from ecologyhydro.model_inputs import prepare_inputs

            benchmark_jobs = [
                {
                    "year": year,
                    "landcover": "copernicus",
                    "scope": "tangnaihai",
                    "z": 5,
                    "force": True,
                }
                for year in config.study.calibration_years
            ]
            if len(benchmark_jobs) != 4:
                raise ValueError("M3 benchmark requires four calibration years")
            warm_started = time.monotonic()
            for job in benchmark_jobs:
                prepare_inputs(
                    index_path, config.paths.cache, job["year"], job["landcover"], job["scope"]
                )
            suite["benchmark_shared_input_preparation_seconds"] = time.monotonic() - warm_started
            for workers in (1, 2, 4):
                batch, summary = run_group(
                    config, config_path, index_path, benchmark_jobs, workers, f"benchmark_{workers}"
                )
                suite["groups"].append(
                    {
                        "label": f"benchmark_{workers}",
                        "path": str(batch / "batch.json"),
                        "elapsed_seconds": summary["elapsed_seconds"],
                    }
                )
                write_json(directory / "latest_suite.json", suite)
        suite["status"] = "success"
    finally:
        suite["elapsed_seconds"] = time.monotonic() - started
        if suite["status"] != "success":
            suite["status"] = "failed"
        write_json(directory / "latest_suite.json", suite)
        write_json(directory / f"suite_{uuid4().hex}.json", suite)
    return suite
