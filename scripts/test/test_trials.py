"""Resource guards must stop real worker processes and preserve failure evidence."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ecologyhydro import trials


@pytest.mark.parametrize("guard", ["timeout", "memory"])
def test_resource_guard_terminates_worker(tmp_path, monkeypatch, guard):
    original_popen = subprocess.Popen
    children = []

    def sleeping_worker(command, **kwargs):
        child = original_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(trials.subprocess, "Popen", sleeping_worker)
    config = SimpleNamespace(
        paths=SimpleNamespace(cache=tmp_path / "cache"),
        resources=SimpleNamespace(
            timeout_hours=0.000001 if guard == "timeout" else 1,
            memory_budget_gb=0.000001 if guard == "memory" else 24,
        ),
    )
    expected = TimeoutError if guard == "timeout" else MemoryError
    job = {"year": 2019, "landcover": "copernicus", "scope": "full", "z": 5}
    with pytest.raises(expected):
        trials.run_group(config, tmp_path / "config", tmp_path / "index", [job], 1, guard)
    assert children and all(child.poll() is not None for child in children)
    batch = next((tmp_path / "m3").glob("*/batch.json"))
    record = json.loads(batch.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert expected.__name__ in record["failure_reason"]
    termination = next(batch.parent.glob("run_*/termination.json"))
    assert expected.__name__ in json.loads(termination.read_text())["reason"]
