"""Select an explicit repaired baseline and reject stale spatial/climate inputs."""

import json
from pathlib import Path

from ecologyhydro.cache import fingerprint
from ecologyhydro.config import load_config, project_root
from ecologyhydro.simulation import valid_completed, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_current_inputs(frozen, current):
    for key in ("grid", "recipe", "aligned", "routing", "biophysical"):
        if key not in frozen or key not in current or frozen[key] != current[key]:
            raise ValueError(f"Baseline is stale relative to current M2: {key}")


def require_run_inputs(prepared, index):
    """Tie the selected batch itself to M2, not just its separately supplied index."""
    expected_mapping = Path(index["routing"]["partitions"]) / "zone_station.csv"
    if Path(prepared["zone_station"]) != expected_mapping or prepared["scope"] != "full":
        raise ValueError("Selected run uses a different catchment or scope")
    year, product = prepared["year"], prepared["landcover"]
    expected = {
        "lulc_path": f"lulc_{product}",
        "precipitation_path": f"{prepared['precipitation']}_{year}",
        "eto_path": f"pet_{year}",
        "pawc_path": "pawc",
        "depth_to_root_rest_layer_path": "root_depth",
    }
    for key, name in expected.items():
        masked = Path(prepared["args"][key])
        manifest = read_json(masked.parent / "manifest.json")
        if Path(manifest["sources"][0]["path"]) != Path(index["aligned"][name]):
            raise ValueError(f"Selected run uses a different M2 input: {name}")


def current_models():
    from ecologyhydro.water_balance import read_csv

    root = project_root()
    selection = root / "config/analysis_baseline.json"
    pointer = read_json(selection)
    index_path = root / pointer["index"]
    index = read_json(index_path)
    require_current_inputs(index, read_json(root / "project/cache/m2/latest.json"))
    summary_path = root / pointer["summary"]
    summary = read_json(summary_path)
    batch_path = Path(summary["batch"])
    batch = read_json(batch_path)
    if batch["status"] != "success":
        raise ValueError("Selected baseline has not completed successfully")
    rows, sources, snapshots, seen = (
        [],
        [selection, index_path, summary_path, batch_path],
        [],
        set(),
    )
    for entry in batch["completed"]:
        directory = Path(entry["directory"])
        state = read_json(directory / "run.json")
        if state["status"] == "reused":
            directory = Path(state["reused_run_dir"])
            state = read_json(directory / "run.json")
        if not valid_completed(state):
            raise ValueError("Baseline result failed output integrity checks")
        snapshot = read_json(directory / "snapshot.json")
        require_run_inputs(snapshot["prepared"], index)
        scenario = snapshot["prepared"].get("science_scenario", {})
        if set(scenario.get("overrides", {})) - {"biophysical_table_path"}:
            raise ValueError("Selected M2 baseline contains additional input overrides")
        # Inputs from the actual run must also remain unmodified.
        for source in snapshot["recipe"]["sources"]:
            # Historical implementation files can evolve; data files cannot.
            if (
                Path(source["path"]).suffix != ".py"
                and fingerprint(Path(source["path"]))["sha256"] != source["sha256"]
            ):
                raise ValueError("Baseline input changed since official execution")
        snapshots.append((directory, snapshot))
        sources += [directory / "stations.csv", directory / "snapshot.json"]
        for row in read_csv(directory / "stations.csv"):
            key = (int(row["year"]), row["landcover"], row["station"])
            if key in seen:
                raise ValueError("Duplicate baseline station/year/product")
            seen.add(key)
            rows.append(
                {
                    **row,
                    "awy_yield_1e8_m3": row["natural_yield_1e8_m3"],
                    "variant": entry["job"]["variant"],
                    "parameter_version": entry["job"]["variant"],
                }
            )
    config = load_config()
    expected = {
        (y, p, s)
        for y in config.study.calibration_years
        for p in ("fine", "copernicus")
        for s in config.study.station_ids
    }
    if seen != expected:
        raise ValueError("Selected baseline must cover exactly the declared development suite")
    return rows, sources, snapshots


def refresh():
    from ecologyhydro.aggregation import write_csv
    from ecologyhydro.et0_trials import yield_floor
    from ecologyhydro.model_inputs import prepare_inputs
    from ecologyhydro.runtime import configure_threads

    config, root = load_config(), project_root()
    configure_threads(config.resources)
    rows, sources, snapshots = current_models()
    index_path = root / read_json(root / "config/analysis_baseline.json")["index"]
    output = root / "project/diagnostics/current_baseline"
    output.mkdir(parents=True, exist_ok=True)
    bounds = []
    for directory, snapshot in snapshots:
        prepared = snapshot["prepared"]
        year, product = prepared["year"], prepared["landcover"]
        headwater = prepare_inputs(index_path, config.paths.cache, year, product, "tangnaihai")
        headwater["args"]["biophysical_table_path"] = prepared["args"]["biophysical_table_path"]
        budget = yield_floor(headwater["args"], headwater["zones"])
        bounds.append(
            {
                **budget,
                "reference_et0_mm": budget["et0_mean_mm"],
                "year": year,
                "landcover": product,
                "station": "唐乃亥",
                "variant": "rotation" if product == "fine" else "central",
                "run_id": directory.name,
            }
        )
    write_csv(output / "stations.csv", rows)
    write_csv(output / "headwater_bounds.csv", bounds)
    write_json(
        output / "manifest.json",
        {
            "sources": [fingerprint(p) for p in sources],
            "implementation": [
                fingerprint(Path(__file__)),
                fingerprint(Path(__file__).with_name("et0_trials.py")),
            ],
            "input_match_current_m2": True,
            "model_rows": len(rows),
            "calibrated": False,
            "validation_used": False,
        },
    )
    return output


if __name__ == "__main__":
    print(refresh())
