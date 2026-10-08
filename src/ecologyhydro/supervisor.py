"""Bound long preprocessing jobs and record process-tree resource use."""

import json
import subprocess
import sys
import time
from contextlib import suppress

import psutil


def run_preprocessing(config, config_path, recipe, stages):
    started = time.monotonic()
    command = [
        sys.executable,
        "-m",
        "ecologyhydro.preprocess",
        "--config",
        str(config_path),
        "--recipe",
        str(recipe),
        "--stages",
        *stages,
    ]
    peak = 0
    status = "running"
    child = subprocess.Popen(command)
    process = psutil.Process(child.pid)
    try:
        while child.poll() is None:
            try:
                members = [process, *process.children(recursive=True)]
            except psutil.NoSuchProcess:
                child.wait()
                break
            rss = 0
            for member in members:
                with suppress(psutil.NoSuchProcess):
                    rss += member.memory_info().rss
            peak = max(peak, rss)
            if rss > config.resources.memory_budget_gb * 1024**3:
                status = "memory_limit"
                raise RuntimeError("M2 process tree exceeded configured memory budget")
            if time.monotonic() - started > config.resources.timeout_hours * 3600:
                status = "timeout"
                raise RuntimeError("M2 exceeded configured wall-time budget")
            time.sleep(0.5)
        status = "success" if child.returncode == 0 else "failed"
        return child.returncode
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        if child.poll() is None:
            members = [process]
            with suppress(psutil.NoSuchProcess):
                members.extend(process.children(recursive=True))
            for member in reversed(members):
                with suppress(psutil.NoSuchProcess):
                    member.kill()
            child.wait()
        record = {
            "command": command,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": peak,
            "status": status,
            "scope": "M2 preprocessing only; not M3 AWY simulation performance acceptance",
        }
        path = config.paths.logs / f"m2_resources_{time.time_ns()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
