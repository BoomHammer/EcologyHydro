"""Freeze inputs and evaluation rules before fitting the monthly pilot."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yaml
from calibrate_regional_water import sample_year, sampling_positions
from diagnose_headwater_balance import spatial_weights
from long_inputs import paths_for
from netCDF4 import Dataset, num2date
from osgeo import gdal
from prepare_long_climate import monthly_data
from validate_requested_years import inputs

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import load_config, project_root
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import write_json


def fetch(spec, output):
    variable, end = spec
    path = output / f"{variable}_2012_{end}.nc"
    if path.exists():
        expected = read_json(path.with_suffix(".json"))["file"]
        if fingerprint(path)["sha256"] != expected["sha256"]:
            raise ValueError("Cached climate changed")
        return path
    query = urlencode(
        dict(
            var=variable,
            north=39,
            south=31.5,
            west=95,
            east=105,
            horizStride=1,
            time_start="2012-01-01T00:00:00Z",
            time_end=f"{end}-12-31T23:59:59Z",
            timeStride=1,
            accept="netcdf",
        )
    )
    url = (
        "https://tds-proxy.nkn.uidaho.edu/thredds/ncss/grid/"
        f"agg_terraclimate_{variable}_1950_CurrentYear_GLOBE.nc?{query}"
    )
    for attempt in range(3):
        try:
            with urlopen(url, timeout=50) as response:
                content = response.read(40 * 1024**2 + 1)
            if len(content) > 40 * 1024**2 or not content.startswith((b"CDF", b"\x89HDF")):
                raise ValueError("Invalid NetCDF download")
            path.write_bytes(content)
            write_json(path.with_suffix(".json"), dict(url=url, file=fingerprint(path)))
            print("Downloaded", path.name, flush=True)
            return path
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    raise RuntimeError("Unreachable")


def temperature(path, variable):
    with Dataset(path) as ds:
        v, t = ds[variable], ds["time"]
        dates = num2date(t[:], t.units, calendar=getattr(t, "calendar", "standard"))
        if [(d.year, d.month) for d in dates] != [
            (y, m) for y in range(2012, 2024) for m in range(1, 13)
        ]:
            raise ValueError("Incomplete temperature")
        if ds.getncattr("version") != "V1.1" or v.units not in ("degC", "C", "degree_Celsius"):
            raise ValueError(f"Unexpected temperature version/units: {v.units}")
        values = np.ma.asarray(v[:], dtype=float).filled(np.nan)
        lat, lon = np.asarray(ds["lat"][:]), np.asarray(ds["lon"][:])
    if lat[0] < lat[-1]:
        lat, values = lat[::-1], values[:, ::-1]
    if lon[0] > lon[-1]:
        lon, values = lon[::-1], values[:, :, ::-1]
    dx, dy = abs(lon[1] - lon[0]), abs(lat[1] - lat[0])
    return values, (lon[0] - dx / 2, dx, 0, lat[0] + dy / 2, 0, -dy)


def main():
    root = project_root()
    output = root / "project/calibration/monthly_pilot_v1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "inputs.json").exists():
        raise ValueError("Completed inputs already exist")
    configure_threads(load_config().resources)
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    write_json(
        output / "protocol.json",
        dict(
            scope="Guide headwater only; unchanged downstream increments; two landcovers",
            models=[
                "existing_reach7",
                "headwater_annual_2_parameters",
                "monthly_bucket_3_parameters",
            ],
            early_training=list(range(2013, 2018)),
            validation=list(range(2019, 2023)),
            rolling="Fit strictly prior years; exclude accountless 2018",
            reused_2023="Report all-development prediction; never use in acceptance or tuning",
            initial="2012 meteorological warmup; initial soil half capacity, snow/routing zero",
            monthly=(
                "Lumped basin climate; landcover x soil-capacity HRUs; fixed snow partition/melt"
            ),
            snow="Linear snowfall -1 to +1 C; melt 3 mm/C/day; assumption, not measured snow",
            acceptance=dict(
                early_and_rolling_guide_rmse_reduction_vs_both_baselines=0.10,
                worst_guide_error_increase_limit_pp=2,
                seven_station_rmse_ratio_limit=1.02,
                initial_state_prediction_sensitivity_limit_pct_observation=2,
                early_training_delete_year_spread_ratio_vs_local_annual_limit=1.10,
            ),
            selection="All gates must pass for both landcovers. No retuning after validation.",
            missing=(
                "2018 climate advances states, no imputed fitting target; 2014 propagate missing"
            ),
            references=[
                "https://storage.googleapis.com/releases.naturalcapitalproject.org/invest/3.17.0/userguide/en/annual_water_yield.html",
                "https://www.climatologylab.org/terraclimate.html",
                "https://hydromad.github.io/articles/tutorial.html",
            ],
            limitations=(
                "Custom minimal water-balance prototype, not IHACRES or official InVEST; "
                "monthly process unvalidated with annual targets; reused development years"
            ),
        ),
    )
    specs = [("tmin", 2023), ("tmax", 2023), ("ppt", 2012), ("pet", 2012)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        downloaded = list(pool.map(lambda s: fetch(s, output), specs))
    zone = root / "project/repairs/closed_routing_v3/zones.tif"
    sources, cache = [zone, Path(__file__), root / "src/ecologyhydro/monthly_pilot.py"], {}

    def reduce(values, gt):
        key = (tuple(gt), values.shape[1:])
        if key not in cache:
            cache[key] = spatial_weights(zone, gt, values.shape[1:])[0]
        w = cache[key]
        selected = w > 0
        data = values.reshape(len(values), -1)[:, selected]
        if not np.isfinite(data).all():
            raise ValueError("Missing contributing forcing")
        return data @ w[selected] / w.sum()

    series = {}
    for variable, path in zip(("tmin", "tmax"), downloaded[:2], strict=True):
        series[variable] = reduce(*temperature(path, variable))
    monthly = pd.read_csv(root / "project/diagnostics/headwater_balance_v1/monthly_balance.csv")
    climate = pd.read_csv(root / "project/calibration/long_record_v1/climate_regions_fine.csv")
    volume = float(climate[climate.region_id == 1].area_km2.iloc[0]) / 1e5
    sources += downloaded + [
        root / "project/diagnostics/headwater_balance_v1/monthly_balance.csv",
        root / "project/calibration/long_record_v1/climate_regions_fine.csv",
    ]
    for variable in ("ppt", "pet"):
        values, gt, _ = monthly_data(output / f"{variable}_2012_2012.nc", variable, 2012)
        first = reduce(values, gt)
        last_path = (
            root
            / "project/repairs/climate_reference_full_holdout/provider"
            / f"terraclimate_current_{variable}_2023.nc"
        )
        values, gt, _ = monthly_data(last_path, variable, 2023)
        last = reduce(values, gt)
        middle = (
            monthly[monthly.region_id == 1].sort_values(["year", "month"])[variable].to_numpy()
            / volume
        )
        series[variable] = np.r_[first, middle, last]
        sources.append(last_path)
    forcing = dict(
        year=np.repeat(np.arange(2012, 2024), 12),
        month=np.tile(np.arange(1, 13), 12),
        p=series["ppt"],
        pet=series["pet"],
        temperature=(series["tmin"] + series["tmax"]) / 2,
    )
    pd.DataFrame(forcing).to_csv(output / "forcing.csv", index=False)
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    crop_path = root / "config/crop_systems.yaml"
    crop = yaml.safe_load(crop_path.read_text(encoding="utf-8"))["systems"][
        "winter_wheat_summer_maize"
    ]["monthly_kc"]
    sources.append(crop_path)
    for product in ("fine", "copernicus"):
        paths = paths_for(root, index, product, 2013)
        positions, weights, _ = sampling_positions(paths)
        tables_path = root / f"project/calibration/long_record_v1/tables_{product}.json"
        tables = read_json(tables_path)
        samples = []
        for year in range(2013, 2024):
            yearly_paths = (
                inputs(root, index, product, year)
                if year == 2023
                else paths_for(root, index, product, year)
            )
            table_path = root / f"project/calibration/requested_years_v1/table_{product}_2023.json"
            table = read_json(table_path) if year == 2023 else tables[str(year)]
            data = sample_year(yearly_paths, table, positions, weights)
            select = data["zone"] <= 1
            samples.append({k: v[select] for k, v in data.items()})
            sources.extend(Path(p) for p in yearly_paths.values())
        annual = {k: np.stack([s[k] for s in samples]) for k in samples[0]}
        np.savez_compressed(output / f"annual_{product}.npz", **annual)
        data = samples[0]
        codes = data["code"].astype(int)
        bins = np.digitize(data["awc"], [25, 75, 150, 300])
        groups = np.unique(np.column_stack([codes, bins]), axis=0)
        unit_rows = []
        for code, band in groups:
            take = (codes == code) & (bins == band)
            w = data["weights"][take]
            unit_rows.append(
                dict(
                    code=int(code),
                    soil_band=int(band),
                    weight=float(w.sum() / volume),
                    awc=float(np.average(data["awc"][take], weights=w)),
                    veg=float(data["veg"][take][0]),
                    kc=float(data["kc"][take][0]),
                )
            )
        frame = pd.DataFrame(unit_rows)
        if not np.isclose(frame.weight.sum(), 1, atol=1e-8):
            raise ValueError("HRU area weights do not sum to one")
        kc = np.tile(frame.kc.to_numpy(), (144, 1))
        if product == "fine":
            kc[:, frame.code == 2] = np.tile(crop, 12)[:, None]
        np.savez_compressed(
            output / f"units_{product}.npz",
            awc=frame.awc.to_numpy(),
            veg=frame.veg.to_numpy(),
            weights=frame.weight.to_numpy(),
            kc=kc,
            volume=volume,
        )
        frame.to_csv(output / f"units_{product}.csv", index=False)
        sources += [tables_path, table_path]
        print("Prepared", product, len(frame), "HRUs", flush=True)
    sources += list((root / "data/Hydrology").glob("*2013-2023.csv"))
    sources += [output / "protocol.json", output / "forcing.csv"]
    sources += list(output.glob("*.npz"))
    write_json(
        output / "inputs.json",
        dict(
            sources=[fingerprint(p) for p in sorted(set(sources))],
            years=list(range(2013, 2024)),
            area_km2=volume * 1e5,
        ),
    )


if __name__ == "__main__":
    main()
