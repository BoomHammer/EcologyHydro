"""Local Z and carryover calibration without forcing equal residence across reaches."""

import numpy as np
from scipy.optimize import least_squares

from ecologyhydro.class_response import evaluate
from ecologyhydro.fine_priority import regional_delivery
from ecologyhydro.independent_local_balance import reach_data
from ecologyhydro.long_record import REACH_MAP
from ecologyhydro.regional_transfer import training_data


def fit_region(data, target, adjustment, eligible, keep, seed, region, recipe):
    if region not in (3, 5, 6):
        raise ValueError("Only the three predefined reaches are supported")
    horizon = np.flatnonzero(keep)[-1] + 1
    forcing = np.arange(len(keep)) < horizon
    local = training_data(reach_data(data, region), forcing)
    valid = (eligible[:, region] & eligible[:, region - 1] & keep)[:horizon]
    observed = np.diff(target[:, region - 1 : region + 1], axis=1).ravel()[:horizon]
    known = np.diff(adjustment[:, region - 1 : region + 1], axis=1).ravel()[:horizon]
    if valid.sum() < 3 or not np.isfinite((observed + known)[valid]).all():
        raise ValueError("Insufficient regional training data")
    base = np.asarray(seed).copy()
    base[13] = 0  # LTG loss belongs only to region 2.
    cached_x, cached = None, None

    def compute(values):
        nonlocal cached_x, cached
        if cached_x is None or not np.array_equal(cached_x, values):
            x = base.copy()
            x[region], x[14] = values
            predicted, jac = evaluate(x, local, horizon, jacobian=True)
            cached_x = values.copy()
            cached = (
                (predicted[:, region] - known - observed)[valid] / 100,
                jac[:, region][:, [region, 14]][valid] / 100,
            )
        return cached

    fits = [
        least_squares(
            lambda v: compute(v)[0],
            start,
            jac=lambda v: compute(v)[1],
            bounds=recipe["bounds"],
            x_scale="jac",
            max_nfev=recipe["max_nfev"],
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        for start in recipe["starts"]
    ]
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.fun).all()]
    if not successful:
        raise RuntimeError("No converged regional storage fit")
    best = min(successful, key=lambda fit: fit.cost)
    return dict(
        parameters=best.x.tolist(),
        region=region,
        success=True,
        training_indices=np.flatnonzero(keep).tolist(),
        forcing_indices=np.flatnonzero(forcing).tolist(),
        observations=int(valid.sum()),
        objective=float(2 * best.cost),
        starts=[dict(success=bool(f.success), cost=float(f.cost), nfev=int(f.nfev)) for f in fits],
    )


def predict(data, seed, head, regional, years):
    x = np.asarray(seed).copy()
    retention = np.full(7, x[14])
    x[0], retention[0] = head[0], head[2]
    for region, parameters in regional.items():
        x[int(region)], retention[int(region)] = parameters
    changed = {key: value.copy() for key, value in data.items()}
    upper = (REACH_MAP[data["zone"]] == 0) & (data["veg"] == 1)
    changed["kc"][upper] *= head[1]
    x[13], x[14] = 0, 0
    natural = evaluate(x, changed, years)
    flow, _ = regional_delivery(natural, retention)
    flow[:, 2:] -= seed[13]
    return flow
