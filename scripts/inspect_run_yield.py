"""Count invalid/negative official yields without changing model rasters."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from osgeo import gdal

from ecologyhydro.baseline import read_json
from ecologyhydro.spatial import windows


def main():
    gdal.UseExceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    registry = read_json(args.run / "registry.json")
    snapshot = read_json(args.run / "snapshot.json")
    paths = {
        "zones": snapshot["prepared"]["zones"],
        "yield": registry["wyield"],
        "p": registry["precip"],
        "et": registry["eto"],
    }
    counts = {"negative": 0, "missing": 0, "zero_p": 0}
    minimum, negative_sum = 0, 0
    with ExitStack() as stack:
        ds = {k: stack.enter_context(gdal.Open(str(p))) for k, p in paths.items()}
        for block in windows(ds["zones"]):
            inside = ds["zones"].ReadAsArray(*block) > 0
            y = ds["yield"].ReadAsArray(*block)
            mask = ds["yield"].GetRasterBand(1).GetMaskBand().ReadAsArray(*block) > 0
            negative = inside & mask & (y < 0)
            counts["negative"] += int(negative.sum())
            counts["missing"] += int((inside & (~mask | ~np.isfinite(y))).sum())
            counts["zero_p"] += int((inside & (ds["p"].ReadAsArray(*block) == 0)).sum())
            if negative.any():
                minimum = min(minimum, float(y[negative].min()))
                negative_sum += float(y[negative].sum())
    print({**counts, "minimum_mm": minimum, "negative_sum_mm_pixels": negative_sum})


if __name__ == "__main__":
    main()
