"""Explicit, single-field prior perturbations shared by sample and full-grid evaluation."""

from copy import deepcopy

import numpy as np


def fit_spec(spec):
    return {k: spec[k] for k in ("model", "kc_cap")}


def candidate_sample(data, recipe, spec, product):
    if "dryland_kc" not in spec:
        return data
    result = data.copy()
    selected = np.isin(data["code"], recipe["dryland_codes"][product])
    result["kc"] = np.where(selected, spec["dryland_kc"], data["kc"])
    return result


def candidate_tables(tables, recipe, spec, product):
    result = deepcopy(tables)
    if "dryland_kc" in spec:
        for rows in result.values():
            for row in rows:
                if int(row["lucode"]) in recipe["dryland_codes"][product]:
                    row["kc"] = spec["dryland_kc"]
                    row["kc_source"] = "predeclared_dryland_sensitivity_not_measured"
    return result
