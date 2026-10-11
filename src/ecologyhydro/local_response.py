"""Local accounting calibration and explicitly labelled common-response controls."""

import numpy as np
from scipy.optimize import least_squares

from ecologyhydro.long_record import REACH_MAP
from ecologyhydro.regional_transfer import evaluate, training_data


def local_data(data, reach=2):
    selected = REACH_MAP[data["zone"]] == reach
    return {key: value[selected].copy() for key, value in data.items()}


def common_response(data, codes, proxy):
    """Change response attributes, never spatial weights, climate or class labels.

    Input is already restricted to LTG. This is a process-harmonization control,
    not evidence of vegetation or measured rooting depth at the affected pixels.
    """
    output = data.copy()
    selected = np.isin(data["code"], codes)
    for field, source in (("kc", "kc"), ("veg", "lulc_veg"), ("root_depth", "root_depth")):
        output[field] = np.where(selected, float(proxy[source]), data[field])
    output["awc"] = np.minimum(output["root_depth"], output["soil"]) * output["pawc"]
    return output


def fit_local(data, observed_increment, known_increment, valid, keep, seed, settings):
    """Fit local Z and net loss only, physically excluding held forcing/observations."""
    subset = training_data(data, keep)
    observed, known, eligible = (a[keep] for a in (observed_increment, known_increment, valid))
    if eligible.sum() < 3 or not np.isfinite(observed[eligible] + known[eligible]).all():
        raise ValueError("Insufficient valid local training observations")
    seed = np.asarray(seed, dtype=float)
    cache_x, cache_result = None, None

    def unpack(x):
        result = seed.copy()
        result[[2, 8]] = x
        return result

    def compute(x):
        nonlocal cache_x, cache_result
        if cache_x is None or not np.array_equal(cache_x, x):
            pred, jac = evaluate(unpack(x), subset, "reach7", int(keep.sum()), jacobian=True)
            residual = (pred[:, 2][eligible] - known[eligible] - observed[eligible]) / 100
            derivative = jac[:, 2][:, [2, 8]][eligible] / 100
            cache_x, cache_result = x.copy(), (residual, derivative)
        return cache_result

    fits = [
        least_squares(
            lambda x: compute(x)[0],
            start,
            jac=lambda x: compute(x)[1],
            bounds=(
                [settings["Z_bounds"][0], settings["L_bounds"][0]],
                [settings["Z_bounds"][1], settings["L_bounds"][1]],
            ),
            max_nfev=150,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        for start in settings["starts"]
    ]
    converged = [fit for fit in fits if fit.success and np.isfinite(fit.fun).all()]
    if not converged:
        raise ValueError("No local fit converged")
    best = min(converged, key=lambda fit: fit.cost)
    return dict(
        parameters=unpack(best.x).tolist(),
        active_bounds=best.active_mask.tolist(),
        observations=int(eligible.sum()),
        nfev=best.nfev,
        converged_starts=len(converged),
        local_rmse=float(np.sqrt(np.mean(best.fun**2)) * 100),
    )
