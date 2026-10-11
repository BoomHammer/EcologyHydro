"""Joint station and LTG balance fit with training-only normalization and forcing."""

import numpy as np
from scipy.optimize import least_squares

from ecologyhydro.class_response import PARAMETERS, evaluate
from ecologyhydro.regional_transfer import training_data


def joint_problem(data, target, adjustment, eligible, keep, settings):
    """Return residual/Jacobian closure; never normalize using held observations."""
    horizon = np.flatnonzero(keep)[-1] + 1
    forcing = np.arange(len(keep)) < horizon
    subset = training_data(data, forcing)
    observed, known, available = (a[forcing].copy() for a in (target, adjustment, eligible))
    valid = available & keep[forcing, None]
    local_valid = valid[:, 1] & valid[:, 2]
    if valid.sum() < 15 or local_valid.sum() < 3:
        raise ValueError("Insufficient joint training balances")
    if not np.isfinite(observed[valid]).all() or not np.isfinite(known[valid]).all():
        raise ValueError("Nonfinite training observations/accounts")
    if (observed[valid] <= 0).any():
        raise ValueError("Station relative error requires positive observations")
    local_target = observed[:, 2] - observed[:, 1]
    local_scale = max(1.0, float(np.sqrt(np.mean(local_target[local_valid] ** 2))))
    station_scale = observed[valid] * np.sqrt(valid.sum())
    local_factor = np.sqrt(settings["local_balance_weight"] / local_valid.sum()) / local_scale
    center = np.eye(5) - np.ones((5, 5)) / 5
    regularizer = np.sqrt(settings["class_regularization"] / 5)
    cached_x, cached = None, None

    def compute(x):
        nonlocal cached_x, cached
        if cached_x is None or not np.array_equal(cached_x, x):
            prediction, jac = evaluate(x, subset, horizon, jacobian=True)
            error = prediction - known - observed
            local_error = (error[:, 2] - error[:, 1])[local_valid] * local_factor
            local_jac = (jac[:, 2] - jac[:, 1])[local_valid] * local_factor
            penalty = regularizer * center @ np.log(x[7:12])
            penalty_jac = np.zeros((5, 15))
            penalty_jac[:, 7:12] = regularizer * center / x[None, 7:12]
            cached_x = x.copy()
            cached = (
                np.r_[error[valid] / station_scale, local_error, penalty],
                np.vstack([jac[valid] / station_scale[:, None], local_jac, penalty_jac]),
            )
        return cached

    return compute, dict(
        training_indices=np.flatnonzero(keep).tolist(),
        forcing_indices=np.flatnonzero(forcing).tolist(),
        unique_observations=int(valid.sum()),
        local_balances=int(local_valid.sum()),
        local_scale=local_scale,
        local_balance_weight=settings["local_balance_weight"],
    )


def fit_joint(data, target, adjustment, eligible, keep, starts, settings):
    """Refit all 15 parameters together under the same three-start budget per map."""
    compute, details = joint_problem(data, target, adjustment, eligible, keep, settings)
    lower = np.array([1] * 7 + [0.7] * 5 + [0.05, 0, 0])
    upper = np.array([30] * 7 + [2] * 5 + [0.5, 100, 0.5])
    fits = [
        least_squares(
            lambda x: compute(x)[0],
            np.clip(start, lower, upper),
            jac=lambda x: compute(x)[1],
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=settings["max_nfev"],
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        for start in starts
    ]
    finite = [fit for fit in fits if np.isfinite(fit.fun).all()]
    converged = [fit for fit in finite if fit.success]
    if not finite:
        raise RuntimeError("All joint fits are nonfinite")
    best = min(converged or finite, key=lambda fit: fit.cost)
    return dict(
        parameters=best.x.tolist(),
        parameter_names=PARAMETERS,
        **details,
        success=bool(converged),
        converged_starts=len(converged),
        objective=float(2 * best.cost),
        active_bounds=best.active_mask.tolist(),
        starts=[
            dict(
                initial=np.asarray(start).tolist(),
                success=bool(fit.success),
                cost=float(fit.cost),
                nfev=int(fit.nfev),
                optimality=float(fit.optimality),
            )
            for start, fit in zip(starts, fits, strict=True)
        ],
    )
