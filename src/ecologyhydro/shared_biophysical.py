"""Resolve a single evidence registry without product-specific parameter overrides."""

from copy import deepcopy
from math import isfinite, log

import numpy as np
import yaml

from ecologyhydro.config import UniqueKeyLoader

FIELDS = ("lulc_veg", "kc", "root_depth")


def load_registry(path):
    registry = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    if registry["schema_version"] != 1:
        raise ValueError("Unsupported shared parameter schema")
    for name in registry["profiles"]:
        resolve_profile(registry, name)
    for product in registry["mapping"]:
        parameter_table(registry, product)
    return registry


def resolve_profile(registry, name, mode="detail", trail=()):
    if mode not in {"parent", "detail"}:
        raise ValueError("Unknown detail mode")
    if name in trail:
        raise ValueError("Cyclic profile inheritance")
    own = deepcopy(registry["profiles"][name])
    parent = own.pop("parent", None)
    if parent:
        values = resolve_profile(registry, parent, mode, (*trail, name))
        if mode == "parent":
            return values
        if "kc" in own and not own.get("kc_evidence"):
            raise ValueError("Child Kc requires explicit evidence")
        if ("root_depth" in own or "beta" in own) and not own.get("root_evidence"):
            raise ValueError("Child root change requires explicit evidence")
        if "root_depth" in own:
            values.pop("beta", None)
        values.update(own)
    else:
        values = own
    if "beta" in values:
        beta = float(values["beta"])
        if not 0 < beta < 1:
            raise ValueError("Invalid root-distribution beta")
        values["root_depth"] = 10 * log(0.05) / log(beta)
    for field in ("kc", "root_depth"):
        if not isfinite(values[field]) or values[field] <= 0:
            raise ValueError(f"Invalid {field}")
    if values["lulc_veg"] not in (0, 1):
        raise ValueError("Invalid vegetation branch")
    if not values["kc_range"][0] <= values["kc"] <= values["kc_range"][1]:
        raise ValueError("Kc outside declared envelope")
    for field in ("kc_sources", "root_sources"):
        if set(values[field]) - registry["sources"].keys():
            raise ValueError("Unregistered parameter reference")
    values["resolved_profile"] = name
    return values


def parameter_table(registry, product, mode="detail"):
    rows = []
    for code, name in registry["mapping"][product].items():
        values = resolve_profile(registry, name, mode)
        rows.append(
            {
                "lucode": int(code),
                **{field: values[field] for field in FIELDS},
                "profile": name,
                "resolved_profile": values["resolved_profile"],
                "kc_evidence": values["kc_evidence"],
                "root_evidence": values["root_evidence"],
                "kc_sources": "|".join(values["kc_sources"]),
                "root_sources": "|".join(values["root_sources"]),
                "limitation": registry["profiles"][name].get("limitation", ""),
                "status": registry["status"],
            }
        )
    return rows


def replace_sample_parameters(data, rows):
    """Share immutable forcings; allocate only response arrays and recompute AWC."""
    lookup = {r["lucode"]: r for r in rows}
    codes = np.unique(data["code"])
    if missing := set(codes) - lookup.keys():
        raise ValueError(f"Unmapped sampled classes: {missing}")
    output = data.copy()
    for field, dest in (("kc", "kc"), ("root_depth", "root_depth"), ("lulc_veg", "veg")):
        lut = np.zeros(int(codes.max()) + 1)
        for code in codes:
            lut[int(code)] = lookup[int(code)][field]
        output[dest] = lut[data["code"].astype(int)]
    output["awc"] = np.minimum(output["root_depth"], output["soil"]) * output["pawc"]
    return output
