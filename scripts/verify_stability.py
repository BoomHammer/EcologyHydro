"""Verify actual stability artifacts, nested fit isolation and frozen inputs."""

import argparse
from pathlib import Path

import numpy as np
from osgeo import gdal

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.closed_boundary import close_boundary
from ecologyhydro.config import project_root
from ecologyhydro.simulation import write_json
from ecologyhydro.stability import stability_metrics, training_subsets
from ecologyhydro.water_balance import read_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/calibration/stability_v1")
    parser.add_argument("--full-pixel", action="store_true")
    args = parser.parse_args()
    root = project_root()
    output = root / args.output
    gdal.UseExceptions()
    scores_folder = output / "full_pixel_cv" if args.full_pixel else output
    protocol = read_json(output / "protocol.json")
    summary = read_json(scores_folder / "summary.json")
    if protocol["years"] != [2019, 2020, 2021, 2022] or protocol["evaluated_2023"]:
        raise ValueError("Stability experiment accessed a non-development year")
    checks = 0
    for source in read_json(output / "manifest.json")["sources"]:
        if fingerprint(Path(source["path"]))["sha256"] != source["sha256"]:
            raise ValueError(f"Experiment input changed: {source['path']}")
        checks += 1
    if args.full_pixel:
        for source in read_json(scores_folder / "protocol.json")["sources"]:
            if fingerprint(Path(source["path"]))["sha256"] != source["sha256"]:
                raise ValueError(f"Full-pixel verification input changed: {source['path']}")
            checks += 1
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    domain = Path(protocol["domain"]).parent
    with (
        gdal.Open(index["routing"]["direction"]) as old,
        gdal.Open(str(domain / "direction.tif")) as new,
        gdal.Open(str(root / "project/diagnostics/gonghe_connectivity_v1/basin_mask.tif")) as mask,
    ):
        expected, counts = close_boundary(
            old.ReadAsArray(), mask.ReadAsArray() == 1, old.GetRasterBand(1).GetNoDataValue()
        )
        np.testing.assert_array_equal(expected, new.ReadAsArray())
    for product in ("fine", "copernicus"):
        parameters = read_json(output / f"parameters_{product}.json")
        cv = read_csv(scores_folder / f"cv_stations_{product}.csv")
        refits = read_csv(output / f"refit_predictions_{product}.csv")
        for variant, members in parameters["members"].items():
            for excluded, names in enumerate(members["folds"]):
                if variant.startswith("subset_mean"):
                    expected_subsets = training_subsets(np.arange(4) != excluded)
                else:
                    expected_subsets = [tuple(i for i in range(4) if i != excluded)]
                actual = [tuple(parameters["fits"][n]["training_indices"]) for n in names]
                if actual != expected_subsets:
                    raise ValueError("Outer-year leakage or missing ensemble member")
            rows = [r for r in cv if r["variant"] == variant]
            if len(rows) != 24 or any(int(r["year"]) not in protocol["years"] for r in rows):
                raise ValueError("Incomplete or invalid held-out predictions")
            pred = np.array([float(r["predicted"]) for r in rows]).reshape(4, 6)
            obs = np.array([float(r["observed"]) for r in rows]).reshape(4, 6)
            repeated = [r for r in refits if r["variant"] == variant]
            estimates = np.array([float(r["predicted"]) for r in repeated]).reshape(4, 4, 6)
            final = np.array([float(r["full_fit_prediction"]) for r in repeated]).reshape(4, 4, 6)
            np.testing.assert_allclose(final, np.broadcast_to(final[0], final.shape))
            actual_metrics = stability_metrics(pred, obs, estimates, final[0])
            for key, value in actual_metrics.items():
                np.testing.assert_allclose(value, summary["products"][product][variant][key])
    previous = read_json(root / "project/calibration/regional_water_v1/manifest.json")
    old_checks = 0

    def check_original(value):
        nonlocal old_checks
        if isinstance(value, dict):
            if "path" in value and "sha256" in value and not value["path"].endswith(".py"):
                if fingerprint(Path(value["path"]))["sha256"] != value["sha256"]:
                    raise ValueError(f"Historical input modified: {value['path']}")
                old_checks += 1
            else:
                for item in value.values():
                    check_original(item)
        elif isinstance(value, list):
            for item in value:
                check_original(item)

    check_original(previous)
    write_json(
        scores_folder / "verification.json",
        {
            "passed": True,
            "experiment_fingerprints_checked": checks,
            "historical_noncode_inputs_unchanged": old_checks,
            "boundary_exit_edges_terminated": sum(counts),
            "nested_training_members_and_reported_metrics_verified": True,
            "no_2023_predictions_in_new_artifacts": True,
        },
    )
    print("Stability artifact verification passed", flush=True)


if __name__ == "__main__":
    main()
