"""Station-proxy regional AWY with analytic derivatives and past-only calibration."""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from ecologyhydro.long_record import ENDPOINTS, MACRO_MAP, REACH_MACRO, REACH_MAP


def regional_map(model):
    if model == "macro3":
        return MACRO_MAP
    if model in ("reach7", "pooled7"):
        return REACH_MAP
    raise ValueError("Unknown spatial model")


def evaluate(parameters, data, model, years, jacobian=False):
    """AWY minus the one constant reach loss, with analytic Z/Kc/L derivatives."""
    mapping = regional_map(model)
    n_z = int(mapping.max()) + 1
    x = np.asarray(parameters, dtype=float)
    if x.shape != (n_z + 2,) or not np.isfinite(x).all():
        raise ValueError("Invalid regional parameters")
    region = mapping[data["zone"]]
    p, et, awc = data["p"], data["et"], data["awc"]
    veg = data["veg"] == 1
    effective = data["kc"] * x[-2]
    kc = np.where(veg, np.minimum(effective, 1.3), data["kc"])
    ratio = et * kc / p
    log_ratio = np.log(np.maximum(ratio, 1e-300))
    uncapped = 1.25 + x[region] * awc / p
    omega = np.minimum(5, uncapped)
    log_sum = np.logaddexp(0, omega * log_ratio)
    power = np.exp(log_sum / omega)
    fu_aet = 1 + ratio - power
    fraction = np.where(veg, np.minimum(ratio, fu_aet), np.minimum(1, ratio))
    depth = np.maximum(0, p * (1 - fraction))

    def aggregate(values):
        return np.bincount(
            data["groups"], weights=values * data["weights"], minlength=years * 11
        ).reshape(years, 11)

    cumulative = aggregate(depth).cumsum(axis=1)[:, ENDPOINTS]
    cumulative[:, 2:] -= x[-1]
    if not jacobian:
        return cumulative
    active = veg & (fu_aet <= ratio) & (depth > 0)
    probability = expit(omega * log_ratio)
    d_z = power * (omega * probability * log_ratio - log_sum) / omega**2 * awc
    d_z *= active & (uncapped < 5)
    d_ratio = p * (
        np.divide(power * probability, ratio, out=np.zeros_like(ratio), where=ratio > 0) - 1
    )
    # Rare Fu=min(PET,AET) branch is handled explicitly; nonvegetated Kc is fixed.
    d_ratio = np.where(fu_aet <= ratio, d_ratio, -p)
    d_kc = d_ratio * et / p * data["kc"] * veg * (effective < 1.3) * (depth > 0)
    derivatives = np.zeros((years, 11, n_z + 2))
    inc_z = aggregate(d_z)
    for region_id in range(n_z):
        derivatives[:, :, region_id] = inc_z * (mapping == region_id)[None, :]
    derivatives[:, :, -2] = aggregate(d_kc)
    derivatives = derivatives.cumsum(axis=1)[:, ENDPOINTS]
    derivatives[:, 2:, -1] = -1
    return cumulative, derivatives


def pooling(parameters, model, strength=0.01):
    """Fixed log-Z partial pooling within upper/middle/lower groups, not across them."""
    n = len(parameters)
    if model != "pooled7":
        return np.empty(0), np.empty((0, n))
    centering = np.eye(7)
    for group in np.unique(REACH_MACRO):
        ids = np.flatnonzero(group == REACH_MACRO)
        centering[np.ix_(ids, ids)] -= 1 / len(ids)
    multiplier = np.sqrt(strength / 7)
    residual = multiplier * centering @ np.log(parameters[:7])
    jac = np.zeros((7, n))
    jac[:, :7] = multiplier * centering / parameters[None, :7]
    return residual, jac


def training_data(data, keep):
    """Physically remove excluded years before optimizer callbacks are constructed."""
    selected = np.flatnonzero(keep)
    year_ids = data["groups"] // 11
    mask = np.isin(year_ids, selected)
    subset = {key: value[mask] for key, value in data.items()}
    subset["groups"] = np.searchsorted(selected, year_ids[mask]) * 11 + subset["zone"]
    return subset


def fit_transfer(data, target, adjustment, eligible, keep, model):
    keep = np.asarray(keep, dtype=bool)
    target, adjustment, eligible = (np.asarray(a) for a in (target, adjustment, eligible))
    if target.shape != adjustment.shape or target.shape != eligible.shape or target.shape[1] != 7:
        raise ValueError("Require matching year-by-seven arrays")
    if keep.shape != (len(target),) or not keep.any():
        raise ValueError("Invalid training-year selection")
    n_z = int(regional_map(model).max()) + 1
    subset = training_data(data, keep)
    observed, known, valid = target[keep], adjustment[keep], eligible[keep]
    if (
        valid.sum() < n_z + 2
        or not np.isfinite(observed[valid]).all()
        or not np.isfinite(known[valid]).all()
    ):
        raise ValueError("Insufficient or invalid eligible training observations")
    cache_x, cache_result = None, None

    def compute(x):
        nonlocal cache_x, cache_result
        if cache_x is None or not np.array_equal(cache_x, x):
            pred, jac = evaluate(x, subset, model, int(keep.sum()), jacobian=True)
            scale = observed[valid] * np.sqrt(valid.sum())
            regularization, reg_jac = pooling(x, model)
            residual = np.r_[((pred - known)[valid] - observed[valid]) / scale, regularization]
            derivative = np.vstack([jac[valid] / scale[:, None], reg_jac])
            cache_x, cache_result = x.copy(), (residual, derivative)
        return cache_result

    fits = [
        least_squares(
            lambda x: compute(x)[0],
            [z] * n_z + [kc, loss],
            jac=lambda x: compute(x)[1],
            bounds=([1] * n_z + [0.7, 0], [30] * n_z + [2, 100]),
            max_nfev=150,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        for z, kc, loss in ((5, 1, 10), (20, 1.3, 30), (29, 1.8, 50))
    ]
    converged = [fit for fit in fits if fit.success and np.isfinite(fit.fun).all()]
    if not converged:
        raise ValueError("No converged regional transfer fit")
    best = min(converged, key=lambda fit: fit.cost)
    _, derivative = evaluate(best.x, subset, model, int(keep.sum()), jacobian=True)
    # Data-only Jacobian: regularization must not masquerade as observational identification.
    scaled = derivative[valid] / observed[valid, None] * np.array([29] * n_z + [1.3, 100])
    return {
        "parameters": best.x.tolist(),
        "active_bounds": best.active_mask.tolist(),
        "training_indices": np.flatnonzero(keep).tolist(),
        "observations": int(valid.sum()),
        "converged_starts": len(converged),
        "nfev": int(best.nfev),
        "objective_with_penalty": float(2 * best.cost),
        "data_jacobian_singular_values": np.linalg.svd(scaled, compute_uv=False).tolist(),
        "pooling_strength": 0.01 if model == "pooled7" else 0.0,
    }
