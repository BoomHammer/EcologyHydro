"""User-requested stationwise 2018 rough and 2023 extended-training checks."""

import argparse
import multiprocessing
import time
from contextlib import ExitStack, suppress
from copy import deepcopy
from pathlib import Path

import numpy as np
import psutil
import yaml
from calibrate_regional_water import block_data, open_inputs, sample_year, sampling_positions
from long_inputs import paths_for
from osgeo import gdal, osr
from prepare_long_climate import monthly_data
from scipy.ndimage import map_coordinates

from ecologyhydro.aggregation import write_csv
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import load_config, project_root
from ecologyhydro.long_record import ENDPOINTS, REACH_MAP, STATIONS, YEARS, load_accounts, number
from ecologyhydro.regional_transfer import evaluate
from ecologyhydro.rotation import weighted_kc
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import crs, windows
from ecologyhydro.water_balance import read_csv

ALL_STATIONS = [
    "唐乃亥",
    "贵得",
    "兰州",
    "下河沿",
    "石嘴山",
    "头道拐",
    "龙门",
    "三门峡",
    "花园口",
    "高村",
    "利津",
]


def rough_accounts(root):
    """Historical cumulative account scenarios, never an imputed observation."""
    target, c, s, total, _ = load_accounts(root, years=list(range(2013, 2018)))
    del target
    records = []
    for i, station in enumerate(STATIONS):
        valid = np.isfinite(total[:, i])
        records.append(
            {
                "station": station,
                "mean": float(total[valid, i].mean()),
                "minimum": float(total[valid, i].min()),
                "maximum": float(total[valid, i].max()),
                "consumption_mean": float(np.cumsum(c, axis=1)[valid, i].mean()),
                "reservoir_change_mean": float(np.cumsum(s, axis=1)[valid, i].mean()),
                "source_years": [y for y, ok in zip(range(2013, 2018), valid, strict=True) if ok],
            }
        )
    return records


def inputs(root, index, product, year):
    paths = paths_for(root, index, product, 2018)
    if year == 2023:
        folder = root / "project/repairs/climate_reference_full_holdout"
        paths["p"] = str(folder / "provider_ppt_2023_aligned.tif")
        paths["et"] = str(folder / "provider_pet_2023_aligned.tif")
    elif year != 2018:
        raise ValueError("Only requested validation years allowed")
    return paths


def annual_table(root, index, product, year, positions, weights):
    tables = read_json(root / f"project/calibration/long_record_v1/tables_{product}.json")
    table = deepcopy(tables[str(2018 if year == 2018 else 2019)])
    if year != 2023 or product != "fine":
        return table, None
    with gdal.Open(index["aligned"]["lulc_fine"]) as raster:
        crop = raster.ReadAsArray().ravel()[positions] == 2
        yy, xx = np.divmod(positions[crop], raster.RasterXSize)
        gt = raster.GetGeoTransform()
        transform = osr.CoordinateTransformation(crs(raster.GetProjection()), crs("EPSG:4326"))
        lonlat = np.asarray(
            transform.TransformPoints(
                np.column_stack([gt[0] + (xx + 0.5) * gt[1], gt[3] + (yy + 0.5) * gt[5]])
            )
        )
    path = (
        root
        / "project/repairs/climate_reference_full_holdout/provider/terraclimate_current_pet_2023.nc"
    )
    values, gt, _ = monthly_data(path, "pet", 2023)
    columns, rows = (lonlat[:, 0] - gt[0]) / gt[1] - 0.5, (lonlat[:, 1] - gt[3]) / gt[5] - 0.5
    sums = []
    for month in values:
        sampled = map_coordinates(month, [rows, columns], order=1, mode="constant", cval=np.nan)
        if not np.isfinite(sampled).all() or (sampled < 0).any():
            raise ValueError("Missing crop PET")
        sums.append(float(sampled @ weights[crop]))
    calendar = yaml.safe_load((root / "config/crop_systems.yaml").read_text(encoding="utf-8"))[
        "systems"
    ]["winter_wheat_summer_maize"]["monthly_kc"]
    kc = weighted_kc(sums, calendar)
    for row in table:
        if int(row["lucode"]) == 2:
            row["kc"] = kc
    return table, {
        "year": 2023,
        "kc": kc,
        "source": fingerprint(path),
        "method": "Same-year V1.1 PET, original crop strata and fixed crop calendar",
    }


def all_station_yield(paths, table, fits, volume):
    from natcap.invest.annual_water_yield.annual_water_yield import fractp_op

    totals = {name: np.zeros(11) for name in fits}
    nodata = dict.fromkeys(("out_nodata", "eto", "precip", "depth_root", "pawc"), -9999)
    with ExitStack() as stack:
        datasets = open_inputs(stack, paths)
        for window in windows(datasets["zone"]):
            data, _ = block_data(datasets, window, table)
            regions = REACH_MAP[data["zone"]]
            for name, fit in fits.items():
                x = fit["parameters"]
                kc = np.where(data["veg"] == 1, np.minimum(data["kc"] * x[-2], 1.3), data["kc"])
                depth = np.zeros(len(kc))
                for region in np.unique(regions):
                    mask = regions == region
                    fraction = fractp_op(
                        kc[mask],
                        data["et"][mask],
                        data["p"][mask],
                        data["root_depth"][mask],
                        data["soil"][mask],
                        data["pawc"][mask],
                        data["veg"][mask],
                        nodata,
                        x[region],
                    )
                    depth[mask] = (1 - fraction) * data["p"][mask]
                if not np.isfinite(depth).all() or (depth < -0.0005).any():
                    raise ValueError("Invalid official yield")
                totals[name] += np.bincount(
                    data["zone"], weights=np.maximum(0, depth) * volume, minlength=11
                )
    return {name: values.cumsum() for name, values in totals.items()}


def worker(product, output_string):
    root, output = project_root(), Path(output_string)
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    positions, weights, volume = sampling_positions(inputs(root, index, product, 2018))
    cached = read_json(root / f"project/calibration/long_record_v1/parameters_{product}.json")
    obs = {r["测站"]: r for r in read_csv(root / "data/Hydrology/实测年径流量2013-2023.csv")}
    rough = rough_accounts(root)
    records, fitted, sources, checks = [], {}, [], []
    for year, window in ((2018, "early"), (2023, "all_development")):
        paths = inputs(root, index, product, year)
        table, rotation = annual_table(root, index, product, year, positions, weights)
        write_json(output / f"table_{product}_{year}.json", table)
        if rotation:
            write_json(output / "rotation_2023.json", rotation)
        fits = {a: cached[f"reach7__{a}__{window}"] for a in ("legacy", "sector_2022")}
        for account, fit in fits.items():
            trained = [YEARS[i] for i in fit["training_indices"]]
            if max(trained) >= year or 2018 in trained or 2023 in trained:
                raise ValueError("Training contamination")
            expected = 30 if year == 2018 else 58
            if fit["observations"] != expected:
                raise ValueError("Unexpected training observations")
            fitted[f"{year}__{account}"] = {**fit, "training_years": trained}
        annual = all_station_yield(paths, table, fits, volume)
        sample = sample_year(paths, table, positions, weights)
        for account, fit in fits.items():
            approximate = evaluate(fit["parameters"], sample, "reach7", 1)[0]
            approximate[2:] += fit["parameters"][-1]
            discrepancy = float(np.max(np.abs(approximate / annual[account][ENDPOINTS] - 1)))
            if discrepancy > 0.01:
                raise ValueError("Official and independent sampled yields differ by over 1%")
            checks.append({"year": year, "account": account, "max_yield_fraction": discrepancy})
        sources.extend(fingerprint(Path(p)) for p in dict.fromkeys(paths.values()))
        for account, fit in fits.items():
            _, c, s, adjustment, _ = load_accounts(root, account, years=[year])
            for zi, station in enumerate(ALL_STATIONS):
                observed = number(obs[station].get(str(year)))
                predicted, low, high, known, loss = (np.nan,) * 5
                status = "no_supported_station_account"
                if zi in ENDPOINTS:
                    si = STATIONS.index(station)
                    loss = float(fit["parameters"][-1]) if si >= 2 else 0.0
                    if year == 2018:
                        known = rough[si]["mean"]
                        predicted = annual[account][zi] - loss - known
                        low = annual[account][zi] - loss - rough[si]["maximum"]
                        high = annual[account][zi] - loss - rough[si]["minimum"]
                        status = "rough_past_account_scenario_not_observed_2018_account"
                    else:
                        known = float(adjustment[0, si])
                        predicted = annual[account][zi] - loss - known
                        status = "current_year_conditional_account"
                valid = np.isfinite(observed) and np.isfinite(predicted)
                row = dict(
                    landcover=product,
                    account=account,
                    year=year,
                    station=station,
                    awy_yield=float(annual[account][zi]),
                    observed=observed,
                    predicted=predicted,
                    account_adjustment=known,
                    net_loss=loss,
                    rough_account_low=low,
                    rough_account_high=high,
                    error=predicted - observed if valid else np.nan,
                    relative_error_pct=(predicted / observed - 1) * 100 if valid else np.nan,
                    observation_available=bool(np.isfinite(observed)),
                    score_eligible=bool(valid),
                    account_status=status,
                    official_full_pixel=True,
                    training_years=";".join(
                        map(str, fitted[f"{year}__{account}"]["training_years"])
                    ),
                    training_observations=fit["observations"],
                )
                records.append(
                    {
                        k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                        for k, v in row.items()
                    }
                )
        print(product, year, "all station yields and available accounts complete", flush=True)
    write_csv(output / f"stations_{product}.csv", records)
    write_json(output / f"parameters_{product}.json", fitted)
    write_json(output / f"inputs_{product}.json", {"sources": sources, "kernel_checks": checks})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/calibration/requested_years_v1")
    args = parser.parse_args()
    root = project_root()
    output = (root / args.output).resolve()
    if not output.is_relative_to(root / "project/calibration"):
        raise ValueError("Output must be in calibration")
    output.mkdir(parents=True, exist_ok=False)
    configure_threads(load_config().resources)
    source = root / "project/calibration/long_record_v1"
    for entry in read_json(source / "manifest.json")["sources"]:
        if fingerprint(Path(entry["path"]))["sha256"] != entry["sha256"]:
            raise ValueError("Source experiment fingerprint changed")
    protocol = {
        "user_requested": True,
        "model": "reach7",
        "products": ["fine", "copernicus"],
        "first_stage": {"training": list(range(2013, 2018)), "rough_validation": 2018},
        "second_stage": {"training_window": [2013, 2022], "validation": 2023},
        "training_missing_policy": "No imputation; 2018 excluded; 2014 downstream accounts missing",
        "2018_rough_method": (
            "Past 2013-2017 complete cumulative account mean; historical min/max envelope"
        ),
        "2018_envelope": (
            "Account scenario range only, not statistical uncertainty or confidence interval"
        ),
        "2023_status": "User-requested reused-year evaluation; not a new blind test",
        "parameter_source": str(source),
        "parameter_reuse": "Exact already-completed requested fits",
        "unsupported_stations": [s for s in ALL_STATIONS if s not in STATIONS],
        "management_boundary_limits": "Guide != Longyangxia; Lijin != mouth; station proxies",
        "no_new_processes_or_model_selection": True,
        "references": [
            "https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/annual_water_yield.html"
        ],
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "rough_2018_accounts.json", rough_accounts(root))
    paths = [
        Path(__file__),
        output / "protocol.json",
        root / "scripts/long_inputs.py",
        root / "scripts/calibrate_regional_water.py",
        root / "scripts/prepare_long_climate.py",
        root / "src/ecologyhydro/long_record.py",
        root / "config/crop_systems.yaml",
        source / "manifest.json",
        source / "protocol.json",
    ]
    paths += [
        source / f"{name}_{p}.json"
        for name in ("parameters", "tables")
        for p in ("fine", "copernicus")
    ]
    paths += list((root / "data/Hydrology").glob("*2013-2023.csv"))
    paths += [root / "data/Hydrology/用取水耗水量/2022年黄河流域_地表水耗水量.csv"]
    write_json(output / "manifest.json", {"sources": [fingerprint(p) for p in paths]})
    start, peak = time.perf_counter(), 0
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=worker, args=(p, str(output))) for p in ("fine", "copernicus")
    ]
    for process in processes:
        process.start()
    try:
        while any(p.is_alive() for p in processes):
            memory = psutil.Process().memory_info().rss
            for child in psutil.Process().children(recursive=True):
                with suppress(psutil.NoSuchProcess):
                    memory += child.memory_info().rss
            peak = max(peak, memory)
            if memory > 24 * 1024**3 or time.perf_counter() - start > 12 * 3600:
                raise RuntimeError("Resource limit exceeded")
            if any(p.exitcode not in (None, 0) for p in processes):
                raise RuntimeError("Worker failed")
            time.sleep(1)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
    if any(p.exitcode != 0 for p in processes):
        raise RuntimeError("Worker failed")
    write_json(
        output / "run_summary.json",
        {
            "elapsed_seconds": time.perf_counter() - start,
            "peak_process_tree_mib": peak / 1024**2,
            "production_accepted": False,
        },
    )
    print(output / "run_summary.json", flush=True)


if __name__ == "__main__":
    main()
