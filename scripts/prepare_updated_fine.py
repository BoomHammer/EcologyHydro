"""Rebuild the updated fine map and its sampling strata; reuse verified other forcings."""

import csv
from pathlib import Path

import numpy as np
from calibrate_regional_water import sample_year, sampling_positions
from osgeo import gdal
from run_landcover_comparison import input_paths, worker_setup

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.comparison import resolve
from ecologyhydro.lakes import overlay_lakes
from ecologyhydro.shared_biophysical import load_registry, parameter_table
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import warp


def main():
    config = worker_setup("config/landcover_comparison.yaml")
    output = resolve(config, config["paths"]["prepared"])
    output.mkdir(parents=True, exist_ok=True)
    prior = read_json(config["root"] / "project/comparison_2011_2024/prepared.json")
    grid_path = resolve(config, config["paths"]["grid_index"])
    if fingerprint(grid_path) != prior["grid"]:
        raise ValueError("Grid changed; cannot reuse historical forcing")
    for row in prior["climate"]:
        for key in ("raw", "aligned"):
            if fingerprint(Path(row[key]["path"])) != row[key]:
                raise ValueError("Climate changed; cannot reuse without preparation")
    for product in ("copernicus", "worldcover"):
        if (
            fingerprint(resolve(config, config["products"][product]["aligned"]))
            != prior["landcover"][product]
        ):
            raise ValueError("Other product changed; old sample cache cannot be reused")
    fine = config["products"]["fine"]
    registry_path = resolve(config, config["shared_biophysical_registry"])
    registry = load_registry(registry_path)
    rows = parameter_table(registry, "fine", "parent")
    with resolve(config, fine["classes"]).open(encoding="utf-8-sig", newline="") as stream:
        legend = list(csv.reader(stream))[1:]
    if {int(row[0]) for row in legend} != {row["lucode"] for row in rows}:
        raise ValueError("New legend and parameter map differ")
    built = next(row for row in rows if row["lucode"] == fine["built_code"])
    water = next(row for row in rows if row["lucode"] == fine["water_code"])
    if built["profile"] != "built" or water["profile"] != "water":
        raise ValueError("Configured residential/water codes are not mapped correctly")
    aligned = resolve(config, fine["aligned"])
    aligned.parent.mkdir(parents=True, exist_ok=True)
    native = resolve(config, fine["native_aligned"])
    warp(resolve(config, fine["source"]), native, read_json(grid_path)["grid"], method="near")
    overlay = overlay_lakes(
        native, resolve(config, config["paths"]["lake_mask"]), fine["water_code"], aligned.parent
    )
    if overlay["filled_nodata_pixels"]:
        raise ValueError("Lake overlay must not fill gaps in the new map")
    prepared = {
        **prior,
        "config": fingerprint(config["config_path"]),
        "lake_overlays": {**prior.get("lake_overlays", {}), "fine": overlay},
        "landcover": {
            p: fingerprint(resolve(config, s["aligned"])) for p, s in config["products"].items()
        },
        "updated_fine_source": fingerprint(resolve(config, fine["source"])),
        "updated_fine_legend": fingerprint(resolve(config, fine["classes"])),
    }
    write_json(output / "prepared.json", prepared)
    years = config["forcing_years"]
    paths = {year: input_paths(config, "fine", year) for year in years}
    positions, weights, _ = sampling_positions(
        paths[years[0]], config["evaluation"]["sampling_budget_per_zone"]
    )
    samples = {}
    for year in years:
        samples[year] = sample_year(paths[year], rows, positions, weights)
        print("New fine sampled", year, flush=True)
    data = {key: np.concatenate([samples[y][key] for y in years]) for key in samples[years[0]]}
    data["groups"] = np.concatenate([samples[y]["zone"] + i * 11 for i, y in enumerate(years)])
    sample_dir = output / "samples"
    sample_dir.mkdir(exist_ok=True)
    sample_path = sample_dir / "samples_fine.npz"
    np.savez_compressed(sample_path, **data)
    sources = [
        config["config_path"],
        registry_path,
        resolve(config, fine["source"]),
        resolve(config, fine["classes"]),
        aligned,
        Path(__file__),
        resolve(config, config["paths"]["zones"]),
        resolve(config, config["paths"]["pawc"]),
        resolve(config, config["paths"]["soil"]),
    ]
    sources += [Path(r["aligned"]["path"]) for r in prior["climate"]]
    write_json(
        sample_dir / "manifest.json",
        {
            "sources": [fingerprint(path) for path in sources],
            "sample": fingerprint(sample_path),
            "aligned": fingerprint(aligned),
            "strata": "new map zone/class/soil strata; old fine codes and weights not reused",
            "legend": {int(row[0]): row[-1] for row in legend},
        },
    )
    print("Prepared:", output, flush=True)


if __name__ == "__main__":
    gdal.UseExceptions()
    main()
