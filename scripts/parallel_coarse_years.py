"""Supplement a sealed run with independent year tasks using its official kernel."""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from calibrate_regional_water import block_data, open_inputs
from experiment_separated_selection import check_seal, context
from run_landcover_comparison import input_paths
from verify_class_response import official_depth

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.class_response import collapse_hierarchy, group_for
from ecologyhydro.long_record import ENDPOINTS
from ecologyhydro.shared_biophysical import load_registry, parameter_table
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import windows


def calculate(task):
    folder, year = task
    output, _, config = context(folder)
    models = read_json(output / "models.json")
    registry = load_registry(output / "biophysical.yaml")
    hierarchy = read_json(output / "hierarchy.json")
    labels = {int(key): value for key, value in hierarchy["labels"].items()}
    table = parameter_table(registry, "fine", "detail")
    lookup = np.full(256, -1, int)
    for row in table:
        lookup[row["lucode"]] = group_for(registry, row["profile"])
    totals = {source: np.zeros(11) for source in models}
    started = time.perf_counter()
    with ExitStack() as stack:
        datasets = open_inputs(stack, input_paths(config, "fine", year))
        transform = datasets["zone"].GetGeoTransform()
        volume = abs(transform[1] * transform[5] - transform[2] * transform[4]) / 1e11
        for window in windows(datasets["zone"]):
            data, _ = block_data(datasets, window, table)
            data["response_group"] = lookup[data["code"].astype(int)]
            data["weights"] = np.full(len(data["p"]), volume)
            data = collapse_hierarchy(data, labels, hierarchy["components"])
            for source, model in models.items():
                depth = official_depth(np.array(model["parameters"]), data)
                totals[source] += np.bincount(
                    data["zone"], weights=depth * data["weights"], minlength=11
                )
    result = {source: total.cumsum()[ENDPOINTS].tolist() for source, total in totals.items()}
    saved = output / "parallel_years" / f"natural_fine_level1_{year}.json"
    write_json(saved, result)
    target = output / saved.name
    try:
        # Publish a complete file without replacing any existing primary checkpoint.
        os.link(saved, target)
        published = True
    except FileExistsError:
        published = False
        if year == 2011:
            original = read_json(target)
            for source in models:
                np.testing.assert_array_equal(result[source], original[source])
    seconds = time.perf_counter() - started
    print("COARSE YEAR", year, round(seconds, 2), "published", published, flush=True)
    return dict(year=year, seconds=seconds, published=published, artifact=fingerprint(saved))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    args = parser.parse_args()
    check_seal(args.run)
    directory = args.run / "parallel_years"
    directory.mkdir(exist_ok=False)
    write_json(
        directory / "execution.json",
        {
            "script": fingerprint(Path(__file__)),
            "years": args.years,
            "workers": 2,
            "total_concurrent_workers_with_primary": 4,
            "method": "Identical full_product loop and official_depth; year scheduling only",
        },
    )
    with ProcessPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(calculate, [(str(args.run), year) for year in args.years]))
    write_json(directory / "completed.json", records)
    check_seal(args.run)


if __name__ == "__main__":
    main()
