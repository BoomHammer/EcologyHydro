"""Explicit trial parameter tables, never assign parameters to unknown classes."""

import csv
import math

import yaml

from ecologyhydro.config import UniqueKeyLoader


def export_tables(priors_path, class_tables, output):
    priors = yaml.load(priors_path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    for name, mapping in priors["mapping"].items():
        with class_tables[name].open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            next(reader)
            required = {int(row[0]) for row in reader} - {0, 255}
        codes = {int(code) for code in mapping}
        if codes != required:
            raise ValueError(f"Biophysical mapping mismatch: {name}: {codes ^ required}")
        rows = []
        for code, group in mapping.items():
            values = priors["groups"][group]
            if values["lulc_veg"] not in (0, 1):
                raise ValueError("lulc_veg must be 0 or 1")
            if not all(
                math.isfinite(values[key]) and values[key] > 0 for key in ("root_depth", "kc")
            ):
                raise ValueError("Invalid root depth or Kc")
            rows.append(
                {
                    "lucode": int(code),
                    **values,
                    "group": group,
                    "status": priors["status"],
                    "source": priors["provenance"],
                }
            )
        with (output / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    return {
        "status": priors["status"],
        "provenance": priors["provenance"],
        "formal_experiment_ready": False,
        "mapping_complete": True,
    }
