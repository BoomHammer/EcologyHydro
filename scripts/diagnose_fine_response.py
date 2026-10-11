"""Frozen-parameter ablations and reach errors; never select using validation years."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from experiment_shared_parameters import load_samples, sample_directory
from osgeo import gdal
from run_landcover_comparison import worker_setup

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.comparison import accounts, metrics, resolve, training_mask
from ecologyhydro.long_record import REACH_MAP, STATIONS
from ecologyhydro.quality import boundary_mask
from ecologyhydro.regional_transfer import evaluate
from ecologyhydro.shared_biophysical import (
    load_registry,
    parameter_table,
    replace_sample_parameters,
)
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import windows


def case_table(registry, product, case):
    """Change a single mechanism and apply equivalent profiles to every product."""
    mode = "parent" if case.startswith("root_") or case == "parent" else "detail"
    rows = parameter_table(registry, product, mode)
    if case.startswith("root_"):
        detailed = {row["lucode"]: row for row in parameter_table(registry, product, "detail")}
        for row in rows:
            if row["profile"] == case.removeprefix("root_"):
                row["root_depth"] = detailed[row["lucode"]]["root_depth"]
    if case.startswith("bare_"):
        for row in rows:
            if row["profile"] == "bare":
                row["kc"] = float(case.removeprefix("bare_"))
    return rows


def spatial_audit(config, output):
    """Exact areas intersect the supplied basin polygon; no coarse-map truth claim."""
    zone_path = resolve(config, config["paths"]["zones"])
    boundary = config["root"] / "data/Hydrology/黄河流域.shp"
    meta = boundary_mask(boundary, zone_path, output / "basin_mask.tif")
    counts = np.zeros((2, 11, 256), dtype=np.int64)
    with (
        gdal.Open(str(zone_path)) as zones,
        gdal.Open(str(output / "basin_mask.tif")) as mask,
        gdal.Open(str(resolve(config, config["products"]["fine"]["aligned"]))) as fine,
    ):
        gt = zones.GetGeoTransform()
        area = abs(gt[1] * gt[5] - gt[2] * gt[4]) / 1e6
        for window in windows(zones):
            z = zones.ReadAsArray(*window)
            valid = z > 0
            c = fine.ReadAsArray(*window)[valid].astype(int)
            if (c < 0).any() or (c > 255).any():
                raise ValueError("Invalid fine codes in routed domain")
            b = (mask.ReadAsArray(*window)[valid] > 0).astype(int)
            encoded = b * 11 * 256 + (z[valid].astype(int) - 1) * 256 + c
            counts += np.bincount(encoded, minlength=2 * 11 * 256).reshape(2, 11, 256)
    rows = [
        dict(inside_basin=bool(b), zone=z + 1, code=c, area_km2=float(counts[b, z, c] * area))
        for b, z, c in np.argwhere(counts > 0)
    ]
    pd.DataFrame(rows).to_csv(output / "fine_boundary_areas.csv", index=False)
    meta.update(
        routed_inside_km2=float(counts[1].sum() * area),
        routed_outside_km2=float(counts[0].sum() * area),
        boundary_sources=[fingerprint(p) for p in sorted(boundary.parent.glob("黄河流域.*"))],
    )
    write_json(output / "boundary_audit.json", meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    config = worker_setup(run / "execution_protocol.yaml")
    recipe = read_json(run / "experiment.json")
    registry = load_registry(run / "shared_biophysical.yaml")
    # Refuse to diagnose a historical run using a newer sample cache.
    for source in read_json(run / "manifest.json")["sources"]:
        if fingerprint(Path(source["path"])) != source:
            raise ValueError(f"Run input changed: {source['path']}")
    output = run / "mechanism_diagnosis"
    output.mkdir(exist_ok=False)
    spatial_audit(config, output)
    years = np.asarray(config["forcing_years"])
    target, adjustment, eligible, _, _ = accounts(config)
    cases = [
        "parent",
        "detail",
        "root_rice",
        "root_deciduous_tree",
        "root_evergreen_conifer",
        "root_alpine_meadow",
        "root_temperate_steppe",
        "bare_0.30",
        "bare_0.50",
        "no_kc_cap",
    ]
    write_json(
        output / "protocol.json",
        dict(
            cases=cases,
            fitting="none; all fits frozen in source run",
            interpretation="Frozen response sensitivities; not independently validated corrections",
            area_scope="supplied basin polygon intersected with routed zones",
            references=[
                registry["sources"][key] for key in ("INVEST", "FAO_DUAL", "JACKSON", "YANG")
            ],
        ),
    )
    scores, predictions, saturation = [], [], []
    for product in config["products"]:
        raw = load_samples(sample_directory(config, recipe, product), product)
        for case in cases:
            data = replace_sample_parameters(raw, case_table(registry, product, case))
            folds = [("final", None)] + [
                (f"fold{i}", held) for i, held in enumerate(recipe["folds"])
            ]
            for name, held in folds:
                fit = read_json(run / f"fit_{name}.json")
                cap = None if case == "no_kc_cap" else fit.get("kc_cap", 1.3)
                predicted = evaluate(fit["parameters"], data, "reach7", len(years), kc_cap=cap)
                predicted -= adjustment
                masks = (
                    {name: np.isin(years, held)}
                    if held
                    else {
                        "calibration": training_mask(config),
                        "validation": np.isin(years, config["years"]["validation"]),
                        **{str(y): years == y for y in config["years"]["validation"]},
                    }
                )
                for label, keep in masks.items():
                    scores.append(
                        dict(
                            product=product,
                            case=case,
                            period=label,
                            **metrics(predicted, target, eligible & keep[:, None]),
                        )
                    )
                keep = np.isin(years, held) if held else np.ones(len(years), dtype=bool)
                errors = predicted - target
                local = np.diff(errors, axis=1, prepend=0)
                for yi in np.flatnonzero(keep):
                    for si, station in enumerate(STATIONS):
                        predictions.append(
                            dict(
                                product=product,
                                case=case,
                                fit=name,
                                year=int(years[yi]),
                                station=station,
                                predicted=predicted[yi, si],
                                observed=target[yi, si],
                                eligible=bool(eligible[yi, si]),
                                error=errors[yi, si],
                                local_error=local[yi, si],
                            )
                        )
                if case == "detail" and held is None:
                    first = raw["groups"] // 11 == 0
                    for reach in range(7):
                        mask = first & (REACH_MAP[raw["zone"]] == reach) & (data["veg"] == 1)
                        weights = data["weights"]
                        capped = data["kc"] * fit["parameters"][-2] >= cap
                        limited = data["soil"] <= data["root_depth"]
                        saturation.append(
                            dict(
                                product=product,
                                reach=reach,
                                station=STATIONS[reach],
                                vegetation_km2=float(weights[mask].sum() * 1e5),
                                capped_km2=float(weights[mask & capped].sum() * 1e5),
                                soil_limited_km2=float(weights[mask & limited].sum() * 1e5),
                            )
                        )
            print("Diagnosed", product, case, flush=True)
    frame = pd.DataFrame(predictions)
    frame.to_csv(output / "predictions.csv", index=False)
    for (product, case), group in frame[frame.fit != "final"].groupby(["product", "case"]):
        scores.append(
            dict(
                product=product,
                case=case,
                period="cv",
                **metrics(group.predicted, group.observed, group.eligible),
            )
        )
    pd.DataFrame(scores).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(saturation).to_csv(output / "response_saturation.csv", index=False)
    write_json(output / "summary.json", dict(status="complete", source_run=str(run)))
    print(output, flush=True)


if __name__ == "__main__":
    main()
