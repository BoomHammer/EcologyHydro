"""Content-addressed, atomic preprocessing artifacts with independent dependencies."""

import hashlib
import json
import logging
import os
import shutil
import time
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

LOGGER = logging.getLogger(__name__)


def fingerprint(path: Path) -> dict:
    """Hash inputs once per run; include sidecars explicitly in the caller."""
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while hashing: {path}")
    return {"path": str(path.resolve()), "sha256": digest, "size": after.st_size}


class ArtifactCache:
    """Publish only complete directories. Never reuse interrupted products."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.fingerprints = {}
        self.events = []
        self.versions = {
            name: version(name) for name in ("GDAL", "numpy", "netCDF4", "pygeoprocessing")
        }

    def build(self, name, inputs, parameters, producer):
        sources = []
        input_stats = {}
        for path in map(Path, inputs):
            stat = path.stat()
            signature = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
            input_stats[path] = (stat.st_size, stat.st_mtime_ns)
            if signature not in self.fingerprints:
                self.fingerprints[signature] = fingerprint(path)
            sources.append(self.fingerprints[signature])
        recipe = {"sources": sources, "parameters": parameters, "versions": self.versions}
        key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:24]
        target = self.root / name / key
        manifest = target / "manifest.json"
        started = time.perf_counter()
        if manifest.exists():
            record = json.loads(manifest.read_text(encoding="utf-8"))
            if all(
                (target / item["name"]).is_file()
                and (target / item["name"]).stat().st_size == item["size"]
                and (target / item["name"]).stat().st_mtime_ns == item["mtime_ns"]
                and fingerprint(target / item["name"])["sha256"] == item["sha256"]
                for item in record["outputs"]
            ):
                self.events.append({"name": name, "key": key, "hit": True})
                LOGGER.info("Cache hit: %s", name)
                return target
            raise RuntimeError(f"Modified cache artifact; move aside before rebuilding: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".{key}.{uuid4().hex}.partial"
        temporary.mkdir()
        LOGGER.info("Building %s", name)
        try:
            details = producer(temporary) or {}
            if any(
                (path.stat().st_size, path.stat().st_mtime_ns) != stat
                for path, stat in input_stats.items()
            ):
                raise RuntimeError(f"Input changed during preprocessing: {name}")
            outputs = [
                {
                    "name": str(path.relative_to(temporary)),
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                    "sha256": fingerprint(path)["sha256"],
                }
                for path in sorted(temporary.rglob("*"))
                if path.is_file()
            ]
            if not outputs:
                raise RuntimeError(f"Producer created no output: {name}")
            elapsed = time.perf_counter() - started
            record = {**recipe, "outputs": outputs, "seconds": elapsed, "details": details}
            (temporary / "manifest.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.rename(temporary, target)
        except BaseException:
            shutil.rmtree(temporary)
            raise
        self.events.append({"name": name, "key": key, "hit": False, "seconds": elapsed})
        LOGGER.info("Completed %s in %.1f s", name, elapsed)
        return target
