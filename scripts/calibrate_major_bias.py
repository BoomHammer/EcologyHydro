"""Bounded development-only calibration, shared climate selection, official verification."""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ecologyhydro.aggregation import write_csv
from ecologyhydro.baseline import current_models, read_json
from ecologyhydro.cache import ArtifactCache, fingerprint
from ecologyhydro.config import load_config, project_root
from ecologyhydro.et0_trials import aligned_et
from ecologyhydro.model_inputs import prepare_inputs
from ecologyhydro.reference_checks import REGIONS
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import write_json
from ecologyhydro.trials import run_group
from ecologyhydro.water_balance import read_csv


def predict(parameters, data):
    """Independent Fu calculation on weighted spatial strata, with nested station totals."""
    z, multiplier = parameters
    kc = np.where(data["veg"] == 1, np.minimum(data["kc"] * multiplier, 1.3), data["kc"])
    p, pet = data["p"], data["et"] * kc
    omega = np.minimum(5, 1.25 + z * data["awc"] / p)
    ratio = pet / p
    aet_ratio = (
        1 + ratio - np.exp(np.logaddexp(0, omega * np.log(np.maximum(ratio, 1e-300))) / omega)
    )
    aet = np.where(data["veg"] == 1, p * np.minimum(ratio, aet_ratio), np.minimum(p, pet))
    amounts = (p - aet) * data["weights"]
    annual = np.bincount(data["groups"], weights=amounts, minlength=44).reshape(4, 11)
    return np.cumsum(annual, axis=1)


def fit(data, target, keep=None):
    keep = np.ones((4, 11), dtype=bool) if keep is None else keep

    def residual(parameters):
        return ((predict(parameters, data) - target) / target)[keep]

    fits = [
        least_squares(
            residual,
            start,
            bounds=([1, 0.7], [30, 2]),
            max_nfev=70,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        for start in ([5, 1], [20, 1.3], [29, 1.8])
    ]
    successful = [r for r in fits if r.success and np.isfinite(r.fun).all()]
    if not successful:
        raise ValueError("Calibration failed to converge")
    best = min(successful, key=lambda r: np.mean(r.fun**2))
    return best.x, float(np.sqrt(np.mean(best.fun**2)))


def metrics(prediction, target):
    error = prediction - target
    return {
        "mape_pct": float(np.mean(np.abs(error / target)) * 100),
        "relative_rmse_pct": float(np.sqrt(np.mean((error / target) ** 2)) * 100),
        "rmse_1e8_m3": float(np.sqrt(np.mean(error**2))),
        "bias_pct": float(error.sum() / target.sum() * 100),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    config, root = load_config(), project_root()
    configure_threads(config.resources)
    from osgeo import gdal

    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    baseline, source_paths, snapshots = current_models()
    index_path = root / "project/repairs/soil_routing/m2_index.json"
    index = read_json(index_path)
    output = root / "project/calibration/major_bias"
    output.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        predictions = {p: np.zeros((4, 11)) for p in ("fine", "copernicus")}
        targets = {p: np.zeros((4, 11)) for p in ("fine", "copernicus")}
        for row in read_csv(output / "sampled_comparison.csv"):
            if row["variant"] != "fitted_sample":
                continue
            yi = list(config.study.calibration_years).index(int(row["year"]))
            si = config.study.station_ids.index(row["station"])
            predictions[row["landcover"]][yi, si] = float(row["prediction"])
            targets[row["landcover"]][yi, si] = float(row["observed"])
        jobs = read_json(output / "plan.json")["jobs"]
        verify_fit(
            config,
            root,
            output,
            index_path,
            jobs,
            predictions,
            targets,
            baseline,
            started,
            source_paths,
        )
        return
    if (output / "holdout_2023/frozen_training.json").exists():
        raise ValueError("Training is frozen; do not retune this experiment after holdout")
    years = list(config.study.calibration_years)
    if years != [2019, 2020, 2021, 2022]:
        raise ValueError("This first calibration requires the declared four-year development set")
    write_json(
        output / "protocol.json",
        {
            "years": years,
            "holdout": 2023,
            "holdout_accessed": False,
            "climate_selection": "minimum precipitation relative RMSE over six region proxies",
            "candidates": ["cmfd_prec", "cmfd_bcpr", "terraclimate_pair"],
            "parameters": {"z": [1, 30], "vegetated_kc_multiplier": [0.7, 2], "kc_cap": 1.3},
            "objective": "equal-weight station-year relative squared error, observed regulated Q",
            "interpretation": "effective runoff fit; human regulation not explicitly simulated",
            "spatial_quadrature": "8192/zone, minimum 32/LULC stratum; area weights",
            "same_bounds_and_climate_for_both_landcovers": True,
        },
    )
    with gdal.Open(str(Path(index["routing"]["partitions"]) / "zones.tif")) as ds:
        zones = ds.ReadAsArray().ravel()
        gt = ds.GetGeoTransform()
    pixel_volume = abs(gt[1] * gt[5] - gt[2] * gt[4]) / 1e11
    counts = np.bincount(zones, minlength=12)
    inside = zones > 0
    table_path = root / "data/Hydrology/降水量2018-2023.csv"
    area_path = root / "data/Hydrology/水资源二级区面积.csv"
    references = {r["水资源二级区"]: r for r in read_csv(table_path)}
    areas = {r["水资源二级区"]: float(r["计算面积(万平方千米)"]) * 1e4 for r in read_csv(area_path)}
    provider = root / "project/repairs/climate_reference_full"
    climate_paths, rain_rows = {}, []
    for year in years:
        for name, rain, et in (
            ("cmfd_prec", index["aligned"][f"prec_{year}"], index["aligned"][f"pet_{year}"]),
            ("cmfd_bcpr", index["aligned"][f"bcpr_{year}"], index["aligned"][f"pet_{year}"]),
            (
                "terraclimate_pair",
                provider / f"provider_ppt_{year}_aligned.tif",
                provider / f"provider_pet_{year}_aligned.tif",
            ),
        ):
            climate_paths[name, year] = (str(rain), str(et))
            with gdal.Open(str(rain)) as ds:
                values = ds.ReadAsArray().ravel()[inside].astype(float)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("Incomplete or nonpositive precipitation")
            total = np.bincount(zones[inside], weights=values, minlength=12)
            for region, start, end in REGIONS:
                model_depth = total[start + 1 : end + 1].sum() / counts[start + 1 : end + 1].sum()
                reference_depth = float(references[region][str(year)]) * 1e5 / areas[region]
                rain_rows.append(
                    {
                        "year": year,
                        "climate": name,
                        "region": region,
                        "model_mm": model_depth,
                        "reference_mm": reference_depth,
                        "relative_error": model_depth / reference_depth - 1,
                        "used_for_selection": end != 11,
                        "same_boundary": False,
                    }
                )
            source_paths.extend([Path(rain), Path(et)])
    rain_scores = {
        name: float(
            np.sqrt(
                np.mean(
                    [
                        r["relative_error"] ** 2
                        for r in rain_rows
                        if r["climate"] == name and r["used_for_selection"]
                    ]
                )
            )
        )
        for name in ("cmfd_prec", "cmfd_bcpr", "terraclimate_pair")
    }
    selected = min(rain_scores, key=rain_scores.get)
    write_csv(output / "precipitation_comparison.csv", rain_rows)
    write_json(
        output / "climate_selection.json",
        {
            "scores": rain_scores,
            "selected": selected,
            "selection_uses_runoff": False,
            "regional_boundaries_are_proxies": True,
        },
    )
    print(f"Climate selection: {selected}, precipitation scores: {rain_scores}", flush=True)
    samples, tables, targets, predictions, parameter_records, cv_rows = {}, {}, {}, {}, {}, []
    cache = ArtifactCache(config.paths.cache / "major_bias")
    jobs, sampled_rows = [], []
    for product in ("fine", "copernicus"):
        with gdal.Open(index["aligned"][f"lulc_{product}"]) as ds:
            lulc = ds.ReadAsArray().ravel()
        indices, weights, group = [], [], []
        for zone in range(1, 12):
            members = np.flatnonzero(zones == zone)
            labels = lulc[members]
            for code in np.unique(labels):
                stratum = members[labels == code]
                n = min(len(stratum), max(32, round(8192 * len(stratum) / len(members))))
                chosen = stratum[np.linspace(0, len(stratum) - 1, n).astype(int)]
                indices.extend(chosen)
                weights.extend([len(stratum) / n * pixel_volume] * n)
                group.extend([zone - 1] * n)
        indices = np.asarray(indices, dtype=int)
        codes = lulc[indices].astype(int)
        del lulc

        def sample(path, positions=indices):
            with gdal.Open(str(path)) as dataset:
                values = dataset.ReadAsArray().ravel()[positions].astype(float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid sample input {path}")
            return values

        pawc = sample(index["aligned"]["pawc"])
        depth = sample(index["aligned"]["root_depth"])
        chunks = {k: [] for k in ("p", "et", "kc", "awc", "veg", "weights", "groups")}
        base_chunks = {k: [] for k in chunks}
        target = np.empty((4, 11))
        for yi, year in enumerate(years):
            snapshot = next(
                s
                for _, s in snapshots
                if s["prepared"]["landcover"] == product and s["prepared"]["year"] == year
            )
            table = read_csv(snapshot["model_args"]["biophysical_table_path"])
            tables[product, year] = table
            lookup = {int(row["lucode"]): row for row in table}
            kc = np.array([float(lookup[c]["kc"]) for c in codes])
            veg = np.array([float(lookup[c]["lulc_veg"]) for c in codes])
            roots = np.array([float(lookup[c]["root_depth"]) for c in codes])
            rain, et = climate_paths[selected, year]
            values = {
                "p": sample(rain),
                "et": sample(et),
                "kc": kc,
                "veg": veg,
                "awc": np.minimum(roots, depth) * pawc,
                "weights": np.array(weights),
                "groups": np.array(group, dtype=int) + yi * 11,
            }
            for key in chunks:
                chunks[key].append(values[key])
                base_chunks[key].append(
                    values[key]
                    if key not in ("p", "et")
                    else sample(climate_paths["cmfd_prec", year][0 if key == "p" else 1])
                )
            for si, station in enumerate(config.study.station_ids):
                row = next(
                    r
                    for r in baseline
                    if int(r["year"]) == year
                    and r["landcover"] == product
                    and r["station"] == station
                )
                target[yi, si] = float(row["observed_1e8_m3"])
        data = {k: np.concatenate(v) for k, v in chunks.items()}
        base_data = {k: np.concatenate(v) for k, v in base_chunks.items()}
        if np.any(target <= 0):
            raise ValueError("Targets must be complete and positive")
        parameters, score = fit(data, target)
        samples[product], targets[product] = data, target
        predictions[product] = predict(parameters, data)
        parameter_records[product] = {
            "z": float(parameters[0]),
            "kc_multiplier": float(parameters[1]),
            "sample_relative_rmse": score,
            "at_bound": bool(
                parameters[0] > 29.99
                or parameters[0] < 1.01
                or parameters[1] > 1.99
                or parameters[1] < 0.71
            ),
        }
        for withheld in range(4):
            keep = np.ones((4, 11), dtype=bool)
            keep[withheld] = False
            fitted, _ = fit(data, target, keep)
            predicted = predict(fitted, data)[withheld]
            cv_rows.append(
                {
                    "landcover": product,
                    "withheld_year": years[withheld],
                    **metrics(predicted, target[withheld]),
                    "climate_fixed_from_all_development_years": True,
                }
            )
        for variant, estimate in (
            ("original_sample", predict([5, 1], base_data)),
            ("climate_only", predict([5, 1], data)),
            ("fitted_sample", predictions[product]),
        ):
            for yi, year in enumerate(years):
                for si, station in enumerate(config.study.station_ids):
                    sampled_rows.append(
                        {
                            "landcover": product,
                            "year": year,
                            "station": station,
                            "variant": variant,
                            "prediction": estimate[yi, si],
                            "observed": target[yi, si],
                        }
                    )
        print(f"Fitted {product}: {parameter_records[product]}", flush=True)
        for year in years:
            table = [
                {
                    **r,
                    "kc": min(float(r["kc"]) * parameters[1], 1.3)
                    if int(r["lulc_veg"]) == 1
                    else float(r["kc"]),
                }
                for r in tables[product, year]
            ]
            table_path = output / f"biophysical_{product}_{year}.csv"
            write_csv(table_path, table)
            prepared = prepare_inputs(index_path, config.paths.cache, year, product, "full")
            overrides = {"biophysical_table_path": str(table_path)}
            for field, path in zip(
                ("precipitation_path", "eto_path"), climate_paths[selected, year], strict=True
            ):
                artifact = cache.build(
                    "aligned",
                    [Path(path), Path(prepared["zones"])],
                    {"year": year, "field": field},
                    lambda out, s=path, z=prepared["zones"]: aligned_et(s, z, out),
                )
                overrides[field] = str(artifact / "et0.tif")
            scenario_path = output / f"scenario_{product}_{year}.json"
            write_json(
                scenario_path,
                {
                    "name": "major_bias_fitted",
                    "year": year,
                    "landcover": product,
                    "scope": "full",
                    "overrides": overrides,
                    "et0_label": selected + "_et0",
                    "precipitation_label": selected + "_p",
                },
            )
            jobs.append(
                {
                    "year": year,
                    "landcover": product,
                    "scope": "full",
                    "z": float(parameters[0]),
                    "scenario": str(scenario_path),
                    "variant": "major_bias_fitted",
                }
            )
    write_json(
        output / "parameters.json",
        {
            "climate": selected,
            "parameters": parameter_records,
            "target": "regulated_observed_runoff",
            "holdout_used": False,
        },
    )
    write_csv(output / "sampled_comparison.csv", sampled_rows)
    write_csv(output / "parameter_leave_one_year_out.csv", cv_rows)
    write_json(output / "plan.json", {"jobs": jobs, "holdout_used": False})
    verify_fit(
        config,
        root,
        output,
        index_path,
        jobs,
        predictions,
        targets,
        baseline,
        started,
        source_paths,
    )


def verify_fit(
    config, root, output, index_path, jobs, predictions, targets, baseline, started, source_paths
):
    years = list(config.study.calibration_years)
    directory, batch = run_group(
        config,
        root / "config.yaml",
        index_path,
        jobs,
        2,
        "major_bias_fitted",
        batch_root=output / "runs",
    )
    if batch["status"] != "success":
        raise ValueError("Official verification failed")
    final_rows, results, all_metrics = [], {}, {}
    for entry in batch["completed"]:
        run = Path(entry["directory"])
        state = read_json(run / "run.json")
        if state["status"] == "reused":
            run = Path(state["reused_run_dir"])
        for row in read_csv(run / "stations.csv"):
            product, year, station = row["landcover"], int(row["year"]), row["station"]
            yi, si = years.index(year), config.study.station_ids.index(station)
            prediction = float(row["natural_yield_1e8_m3"])
            results[product, year, station] = prediction
            sampled = predictions[product][yi, si]
            before = next(
                float(r["awy_yield_1e8_m3"])
                for r in baseline
                if r["landcover"] == product and int(r["year"]) == year and r["station"] == station
            )
            final_rows.append(
                {
                    "landcover": product,
                    "year": year,
                    "station": station,
                    "before": before,
                    "after": prediction,
                    "observed": targets[product][yi, si],
                    "sample_relative_error": sampled / prediction - 1,
                }
            )
    sample_error = float(max(abs(r["sample_relative_error"]) for r in final_rows))
    for product in ("fine", "copernicus"):
        rows = [r for r in final_rows if r["landcover"] == product]
        target = np.array([r["observed"] for r in rows])
        all_metrics[product] = {
            stage: metrics(np.array([r[stage] for r in rows]), target)
            for stage in ("before", "after")
        }
    write_csv(output / "official_comparison.csv", final_rows)
    write_json(
        output / "summary.json",
        {
            "batch": str(directory / "batch.json"),
            "metrics": all_metrics,
            "sample_max_relative_error": sample_error,
            "quadrature_accepted": sample_error < 0.01,
            "elapsed_seconds": time.perf_counter() - started,
            "holdout_used": False,
            "status": "development_calibrated_not_holdout_validated",
        },
    )
    source_paths += [
        output / "parameters.json",
        output / "plan.json",
        output / "sampled_comparison.csv",
        root / "data/Hydrology/水资源二级区面积.csv",
        root / "data/Hydrology/降水量2018-2023.csv",
        Path(__file__),
        index_path,
    ]
    write_json(
        output / "manifest.json", {"sources": [fingerprint(p) for p in dict.fromkeys(source_paths)]}
    )
    print(all_metrics, flush=True)
    if sample_error >= 0.01:
        raise ValueError("Spatial quadrature discrepancy exceeds 1%; refine before acceptance")


if __name__ == "__main__":
    main()
