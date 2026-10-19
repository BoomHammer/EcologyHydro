"""Check whether dominant fine classes originate in the source or alignment/overlay."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from osgeo import gdal

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.comparison import load_protocol, resolve
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    output = parser.parse_args().directory.resolve()
    snapshot = output / "execution_protocol.yaml"
    config = load_protocol(snapshot if snapshot.is_file() else "config/landcover_comparison.yaml")
    if not snapshot.is_file():
        recorded = read_json(output / "manifest.json")["sources"]
        if fingerprint(config["config_path"]) not in recorded:
            raise ValueError(
                "Historical run has no matching frozen protocol; do not use current map"
            )
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    source_path = resolve(config, config["products"]["fine"]["source"])
    aligned_path = resolve(config, config["products"]["fine"]["aligned"])
    zone_path = resolve(config, config["paths"]["zones"])
    with gdal.Open(str(zone_path)) as zone, gdal.Open(str(aligned_path)) as aligned:
        gt = zone.GetGeoTransform()
        if gt[2] != 0 or gt[4] != 0:
            raise ValueError("Audit requires an unrotated grid")
        width, height = zone.RasterXSize, zone.RasterYSize
        warped = gdal.Warp(
            "",
            str(source_path),
            format="VRT",
            dstSRS=zone.GetProjection(),
            outputBounds=(gt[0], gt[3] + height * gt[5], gt[0] + width * gt[1], gt[3]),
            width=width,
            height=height,
            resampleAlg="near",
            dstNodata=-9999,
            outputType=gdal.GDT_Int16,
        )
        counts = Counter()
        for window in windows(zone):
            inside = zone.ReadAsArray(*window) > 0
            native_codes = warped.ReadAsArray(*window)[inside].astype(int)
            used_codes = aligned.ReadAsArray(*window)[inside].astype(int)
            if (native_codes < 0).any() or (native_codes > 255).any():
                raise ValueError("Raw fine source has gaps/unknown coding in calculation area")
            values, numbers = np.unique(native_codes * 256 + used_codes, return_counts=True)
            for value, number in zip(values, numbers, strict=True):
                counts[divmod(int(value), 256)] += int(number)
        warped = None
        area = abs(gt[1] * gt[5]) / 1e6
    rows = [
        dict(source_code=a, used_code=b, area_km2=n * area) for (a, b), n in sorted(counts.items())
    ]
    rotation_codes = set(config["products"]["fine"]["rotation_codes"])
    summary = dict(
        source_rotation_area_km2=sum(
            r["area_km2"] for r in rows if r["source_code"] in rotation_codes
        ),
        used_rotation_area_km2=sum(
            r["area_km2"] for r in rows if r["used_code"] in rotation_codes
        ),
        changed_area_km2=sum(r["area_km2"] for r in rows if r["source_code"] != r["used_code"]),
        comparison="Raw source independently nearest-warped to current calculation grid",
        source_fingerprints=[fingerprint(p) for p in (source_path, aligned_path, zone_path)],
        transitions=rows,
    )
    write_json(output / "fine_source_audit.json", summary)
    print({k: v for k, v in summary.items() if k not in {"transitions", "source_fingerprints"}})


if __name__ == "__main__":
    main()
