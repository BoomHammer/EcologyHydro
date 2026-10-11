"""Shared class-specific AWY calibration with analytic derivatives and true hierarchy."""

import csv

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from ecologyhydro.long_record import ENDPOINTS, REACH_MAP
from ecologyhydro.regional_transfer import training_data
from ecologyhydro.shared_biophysical import parameter_table, replace_sample_parameters

VEGETATION = ("crop", "tree", "shrub", "grass", "desert_vegetation")
PARAMETERS = [*(f"Z_{i}" for i in range(7)), *VEGETATION, "bare_kc", "loss", "retention"]


def group_for(registry, profile):
    while "parent" in registry["profiles"][profile]:
        profile = registry["profiles"][profile]["parent"]
    return VEGETATION.index(profile) if profile in VEGETATION else (5 if profile == "bare" else -1)


def prepare_response(raw, registry, product):
    rows = parameter_table(registry, product, "detail")
    result = replace_sample_parameters(raw, rows)
    lut = np.full(256, -1, dtype=int)
    for row in rows:
        lut[row["lucode"]] = group_for(registry, row["profile"])
    result["response_group"] = lut[result["code"].astype(int)]
    return result


def thin_samples(raw, budget):
    """Deterministic class-stratified compression, preserving class area exactly."""
    first = np.flatnonzero(raw["groups"] // 11 == 0)
    selected, new_weights = [], []
    for zone in np.unique(raw["zone"][first]):
        members = first[raw["zone"][first] == zone]
        total = raw["weights"][members].sum()
        for code in np.unique(raw["code"][members]):
            ids = members[raw["code"][members] == code]
            area = raw["weights"][ids].sum()
            n = min(len(ids), max(24, round(budget * area / total)))
            selected.extend(ids[np.linspace(0, len(ids) - 1, n).astype(int)])
            new_weights.extend([area / n] * n)
    order = np.argsort(selected)
    selected = np.asarray(selected)[order]
    new_weights = np.asarray(new_weights)[order]
    chunks = []
    years = int(raw["groups"].max() // 11) + 1
    for yi in range(years):
        ids = np.flatnonzero(raw["groups"] // 11 == yi)
        np.testing.assert_array_equal(raw["code"][ids], raw["code"][first])
        chunks.extend(ids[selected])
    result = {key: value[chunks] for key, value in raw.items()}
    result["weights"] = np.tile(new_weights, years)
    return result


def hierarchy_components(raw, registry, legend):
    """Marginalize subclass identity within actual level-one legend categories.

    A broad-class pixel receives the same basin-wide mixture wherever it occurs.
    Fractions use area only, never runoff, climate or individual station location.
    Mixing fluxes keeps urban/desert mixed vegetation branches physically explicit.
    """
    with legend.open(encoding="utf-8-sig", newline="") as stream:
        labels = {int(row["数字"]): row["一级类"] for row in csv.DictReader(stream)}
    table = {row["lucode"]: row for row in parameter_table(registry, "fine", "detail")}
    first = raw["groups"] // 11 == 0
    totals = {}
    for code in np.unique(raw["code"]):
        row = table[int(code)]
        signature = (
            group_for(registry, row["profile"]),
            row["kc"],
            row["root_depth"],
            row["lulc_veg"],
        )
        key = labels[int(code)], signature
        totals[key] = totals.get(key, 0) + raw["weights"][first & (raw["code"] == code)].sum()
    parts = []
    for (parent, signature), area in totals.items():
        group, kc, root, veg = signature
        denominator = sum(v for (p, _), v in totals.items() if p == parent)
        parts.append(
            dict(
                parent=parent,
                group=group,
                kc=kc,
                root=root,
                veg=veg,
                fraction=float(area / denominator),
            )
        )
    return labels, parts


def collapse_hierarchy(raw, labels, components):
    chunks = []
    for component in components:
        codes = [code for code, parent in labels.items() if parent == component["parent"]]
        mask = np.isin(raw["code"], codes)
        part = {key: value[mask] for key, value in raw.items()}
        size = int(mask.sum())
        part["kc"] = np.full(size, component["kc"])
        part["root_depth"] = np.full(size, component["root"])
        part["veg"] = np.full(size, component["veg"])
        part["response_group"] = np.full(size, component["group"], dtype=int)
        part["awc"] = np.minimum(part["root_depth"], part["soil"]) * part["pawc"]
        part["weights"] = part["weights"] * component["fraction"]
        # Erase subclass code: the response depends only on the retained parent label.
        part["code"] = np.full(size, codes[0])
        chunks.append(part)
    return {key: np.concatenate([part[key] for part in chunks]) for key in chunks[0]}


def pixel_response(x, data, jacobian=False):
    x = np.asarray(x, dtype=float)
    p, et, awc = data["p"], data["et"], data["awc"]
    region = REACH_MAP[data["zone"]]
    group = data["response_group"]
    veg = data["veg"] == 1
    scale = x[7 + np.clip(group, 0, 4)]
    effective = data["kc"] * scale
    kc = np.where(veg, np.minimum(effective, 1.3), data["kc"])
    kc = np.where(group == 5, x[12], kc)
    ratio = et * kc / p
    log_ratio = np.log(np.maximum(ratio, 1e-300))
    uncapped = 1.25 + x[region] * awc / p
    omega = np.minimum(5, uncapped)
    log_sum = np.logaddexp(0, omega * log_ratio)
    power = np.exp(log_sum / omega)
    fu = 1 + ratio - power
    fraction = np.where(veg, np.minimum(ratio, fu), np.minimum(1, ratio))
    depth = np.maximum(0, p * (1 - fraction))
    if not jacobian:
        return depth
    active = veg & (fu <= ratio) & (depth > 0)
    probability = expit(omega * log_ratio)
    dz = power * (omega * probability * log_ratio - log_sum) / omega**2 * awc
    dz *= active & (uncapped < 5)
    dratio = p * (
        np.divide(power * probability, ratio, out=np.zeros_like(ratio), where=ratio > 0) - 1
    )
    dratio = np.where(fu <= ratio, dratio, -p)
    dk = dratio * et / p * data["kc"] * veg * (effective < 1.3) * (depth > 0)
    dk = np.where((group == 5) & (ratio < 1), -et, dk)
    return depth, dz, dk


def evaluate(x, data, years, jacobian=False):
    x = np.asarray(x, dtype=float)
    result = pixel_response(x, data, jacobian)

    def aggregate(value):
        return np.bincount(
            data["groups"], weights=value * data["weights"], minlength=years * 11
        ).reshape(years, 11)

    if jacobian:
        depth, dz, dk = result
        derivative = np.zeros((years, 11, 15))
        zder = aggregate(dz)
        for region in range(7):
            derivative[:, :, region] = zder * (region == REACH_MAP)
        ids = data["groups"] * 6 + np.clip(data["response_group"], 0, 5)
        kd = np.bincount(ids, weights=dk * data["weights"], minlength=years * 11 * 6)
        derivative[:, :, 7:13] = kd.reshape(years, 11, 6)
        derivative = derivative.cumsum(axis=1)[:, ENDPOINTS]
    else:
        depth = result
    natural = aggregate(depth).cumsum(axis=1)[:, ENDPOINTS]
    r = x[14]
    state = r / (1 - r) * natural[0]
    predicted = np.empty_like(natural)
    if jacobian:
        state_jac = r / (1 - r) * derivative[0]
        state_jac[:, 14] += natural[0] / (1 - r) ** 2
        output_jac = np.empty_like(derivative)
    for yi, value in enumerate(natural):
        available = state + value
        predicted[yi] = (1 - r) * available
        if jacobian:
            available_jac = state_jac + derivative[yi]
            output_jac[yi] = (1 - r) * available_jac
            output_jac[yi, :, 14] -= available
            state_jac = r * available_jac
            state_jac[:, 14] += available
        state = r * available
    predicted[:, 2:] -= x[13]
    if jacobian:
        output_jac[:, 2:, 13] = -1
        return predicted, output_jac
    return predicted


def fit_shared(datasets, target, adjustment, eligible, keep, mode, seed):
    """Fit runoff residuals only, with equal total weight per map family.

    Hierarchical fine and level-one controls each get half a map-family weight.
    Retention uses chronological forcing, while nonstate fits remove held forcing.
    """
    stateful = mode.startswith("class_storage")
    robust = mode.endswith("_robust")
    horizon = np.flatnonzero(keep)[-1] + 1
    forcing_keep = np.arange(len(keep)) < horizon if stateful else keep
    datasets = {name: training_data(data, forcing_keep) for name, data in datasets.items()}
    target, adjustment, eligible = (a[forcing_keep].copy() for a in (target, adjustment, eligible))
    valid = eligible & keep[forcing_keep, None]
    n_years = len(target)
    free = np.arange(15 if stateful else 14)
    x0 = np.r_[seed[:7], np.full(5, seed[7]), 0.15, seed[8], 0.0]
    lower = np.array([1] * 7 + [0.7] * 5 + [0.05, 0, 0])
    upper = np.array([30] * 7 + [2] * 5 + [0.5, 100, 0.5])
    cache_x, cache_result = None, None

    def compute(values):
        nonlocal cache_x, cache_result
        if cache_x is None or not np.array_equal(values, cache_x):
            x = x0.copy()
            x[free] = values
            residuals, jacobians = [], []
            for name, data in datasets.items():
                pred, jac = evaluate(x, data, n_years, jacobian=True)
                error = pred - adjustment - target
                weight = 0.5 if name.startswith("fine") else 1.0
                scale = target[valid] * np.sqrt(valid.sum() * 3 / weight)
                if robust:
                    # Pseudo-Huber at a fixed 10% relative error: reduce the influence
                    # of exceptional years while preserving equal map-family weights.
                    relative = error[valid] / target[valid]
                    u = np.sqrt(1 + (relative / 0.1) ** 2)
                    factor = np.sqrt(2 / (u + 1))
                    slope = factor * (u + 1) / (2 * u)
                    residuals.append(error[valid] / scale * factor)
                    jacobians.append(jac[valid][:, free] / scale[:, None] * slope[:, None])
                else:
                    residuals.append(error[valid] / scale)
                    jacobians.append(jac[valid][:, free] / scale[:, None])
                if mode == "class_local":
                    local = np.diff(error, axis=1, prepend=0)
                    local_jac = np.diff(jac, axis=1, prepend=0)
                    local_valid = valid & np.column_stack([np.ones(n_years, bool), valid[:, :-1]])
                    local_scale = np.maximum(
                        target, np.column_stack([target[:, 0], target[:, :-1]])
                    )
                    local_scale = local_scale[local_valid] * np.sqrt(local_valid.sum() * 3 / weight)
                    residuals.append(local[local_valid] / local_scale)
                    jacobians.append(local_jac[local_valid][:, free] / local_scale[:, None])
            # Fixed weak shrinkage: different functional types can separate, not run unbounded.
            center = np.eye(5) - np.ones((5, 5)) / 5
            regularizer = np.sqrt(0.005 / 5)
            residuals.append(regularizer * center @ np.log(x[7:12]))
            reg_jac = np.zeros((5, len(free)))
            reg_jac[:, 7:12] = regularizer * center / x[None, 7:12]
            jacobians.append(reg_jac)
            cache_x = values.copy()
            cache_result = np.concatenate(residuals), np.vstack(jacobians)
        return cache_result

    starts = [x0.copy(), np.r_[[15] * 7, [1.2] * 5, 0.3, 20, 0.15]]
    fits = [
        least_squares(
            lambda v: compute(v)[0],
            np.clip(start[free], lower[free], upper[free]),
            jac=lambda v: compute(v)[1],
            bounds=(lower[free], upper[free]),
            max_nfev=100,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        for start in starts
    ]
    converged = [fit for fit in fits if fit.success and np.isfinite(fit.fun).all()]
    if not converged:
        raise RuntimeError("No shared class fit converged")
    best = min(converged, key=lambda fit: fit.cost)
    x = x0.copy()
    x[free] = best.x
    return dict(
        parameters=x.tolist(),
        parameter_names=PARAMETERS,
        mode=mode,
        nfev=int(best.nfev),
        converged_starts=len(converged),
        objective=float(2 * best.cost),
        active_bounds=best.active_mask.tolist(),
        training_indices=np.flatnonzero(keep).tolist(),
        unique_observations=int(valid.sum()),
    )
