"""Refine a reach balance without using downstream errors to compensate its Z/loss."""

import numpy as np
from scipy.optimize import least_squares

from ecologyhydro.class_response import evaluate
from ecologyhydro.long_record import REACH_MAP
from ecologyhydro.regional_transfer import training_data


def reach_data(data, reach=2):
    selected = REACH_MAP[data["zone"]] == reach
    return {key: value[selected].copy() for key, value in data.items()}


def fit_local_storage(data, target, adjustment, eligible, keep, seed, recipe):
    """Keep chronological forcing, exclude held observations, fit only local Z and L."""
    seed = np.asarray(seed, dtype=float)
    horizon = np.flatnonzero(keep)[-1] + 1
    forcing = np.arange(len(keep)) < horizon
    local = training_data(reach_data(data), forcing)
    observed = np.diff(target[:, 1:3], axis=1).ravel()[forcing]
    known = np.diff(adjustment[:, 1:3], axis=1).ravel()[forcing]
    valid = (eligible[:, 1] & eligible[:, 2] & keep)[forcing]
    if valid.sum() < 3 or not np.isfinite((observed + known)[valid]).all():
        raise ValueError("Insufficient local training balances")
    free = np.array([2, 13])
    cached_x, cached = None, None

    def unpack(values):
        x = seed.copy()
        x[free] = values
        return x

    def compute(values):
        nonlocal cached_x, cached
        if cached_x is None or not np.array_equal(cached_x, values):
            prediction, jac = evaluate(unpack(values), local, horizon, jacobian=True)
            residual = (prediction[:, 2] - known - observed)[valid] / 100
            derivative = jac[:, 2][:, free][valid] / 100
            cached_x, cached = values.copy(), (residual, derivative)
        return cached

    fits = [
        least_squares(
            lambda x: compute(x)[0],
            start,
            jac=lambda x: compute(x)[1],
            bounds=recipe["bounds"],
            x_scale="jac",
            max_nfev=recipe["max_nfev"],
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        for start in recipe["starts"]
    ]
    converged = [fit for fit in fits if fit.success and np.isfinite(fit.fun).all()]
    if not converged:
        raise RuntimeError("Local refinement failed to converge")
    best = min(converged, key=lambda fit: fit.cost)
    return dict(
        parameters=unpack(best.x).tolist(),
        seed_parameters=seed.tolist(),
        training_indices=np.flatnonzero(keep).tolist(),
        forcing_indices=np.flatnonzero(forcing).tolist(),
        observations=int(valid.sum()),
        free_parameter_indices=free.tolist(),
        converged_starts=len(converged),
        success=True,
        local_training_rmse=float(np.sqrt(np.mean(best.fun**2)) * 100),
        starts=[dict(success=bool(f.success), cost=float(f.cost), nfev=int(f.nfev)) for f in fits],
    )


def fit_reach_storage(data, target, adjustment, eligible, keep, local_fit, recipe):
    """Fit each remaining regional Z to its own balance, preserving class scales/state."""
    x = np.asarray(local_fit["parameters"]).copy()
    horizon = np.flatnonzero(keep)[-1] + 1
    forcing = np.arange(len(keep)) < horizon
    observed = np.diff(target, axis=1, prepend=0)
    known = np.diff(adjustment, axis=1, prepend=0)
    paired = eligible & np.column_stack([np.ones(len(keep), bool), eligible[:, :-1]])
    diagnostics = []
    for region in (0, 1, 3, 4, 5, 6):
        subset = training_data(reach_data(data, region), forcing)
        valid = (paired[:, region] & keep)[forcing]
        if valid.sum() < 3:
            raise ValueError("Insufficient reach training balances")
        # The one empirical loss belongs to LTG, not to each extracted reach.
        base = x.copy()
        base[13] = 0
        cached_x, cached = None, None

        def compute(values, *, base=base, region=region, subset=subset, valid=valid):
            nonlocal cached_x, cached
            if cached_x is None or not np.array_equal(cached_x, values):
                trial = base.copy()
                trial[region] = values[0]
                prediction, jac = evaluate(trial, subset, horizon, jacobian=True)
                error = prediction[:, region] - known[forcing, region] - observed[forcing, region]
                cached_x = values.copy()
                cached = error[valid] / 100, jac[:, region, region][valid, None] / 100
            return cached

        fits = [
            least_squares(
                lambda values: compute(values)[0],
                [start],
                jac=lambda values: compute(values)[1],
                bounds=([1], [30]),
                x_scale="jac",
                max_nfev=recipe["max_nfev"],
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            for start in (5, 20, 29)
        ]
        converged = [fit for fit in fits if fit.success and np.isfinite(fit.fun).all()]
        if not converged:
            raise RuntimeError(f"No converged local Z for region {region}")
        best = min(converged, key=lambda fit: fit.cost)
        x[region] = best.x[0]
        diagnostics.append(dict(region=region, z=float(best.x[0]), cost=float(best.cost)))
    return dict(
        parameters=x.tolist(),
        seed_parameters=local_fit["seed_parameters"],
        training_indices=np.flatnonzero(keep).tolist(),
        forcing_indices=np.flatnonzero(forcing).tolist(),
        free_parameter_indices=[*range(7), 13],
        local_refinement=local_fit,
        regions=diagnostics,
        success=True,
    )
