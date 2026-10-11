"""Independent response calibration with training-only residuals and equal budgets."""

import numpy as np
from scipy.optimize import least_squares

from ecologyhydro.class_response import PARAMETERS, evaluate
from ecologyhydro.regional_transfer import training_data


def expand_regional(parameters, bare_kc=0.15):
    """Represent the nine-parameter annual model in the class evaluator exactly."""
    x = np.asarray(parameters)
    return np.r_[x[:7], np.full(5, x[7]), bare_kc, x[8], 0.0]


def fit_independent(data, target, adjustment, eligible, keep, mode, seed, settings):
    """Fit one representation; no other map contributes observations or coefficients."""
    if mode not in ("class", "class_storage"):
        raise ValueError("Unsupported independent class model")
    keep = np.asarray(keep, dtype=bool)
    stateful = mode == "class_storage"
    horizon = np.flatnonzero(keep)[-1] + 1
    forcing = np.arange(len(keep)) < horizon if stateful else keep
    subset = training_data(data, forcing)
    observed, known, available = (a[forcing].copy() for a in (target, adjustment, eligible))
    valid = available & keep[forcing, None]
    if valid.sum() < (15 if stateful else 14):
        raise ValueError("Insufficient independent training observations")
    if not np.isfinite(observed[valid]).all() or not np.isfinite(known[valid]).all():
        raise ValueError("Invalid training observations/accounts")
    free = np.arange(15 if stateful else 14)
    lower = np.array([1] * 7 + [0.7] * 5 + [0.05, 0, 0])
    upper = np.array([30] * 7 + [2] * 5 + [0.5, 100, 0.5])
    reference = expand_regional(seed)
    scale = observed[valid] * np.sqrt(valid.sum())
    center = np.eye(5) - np.ones((5, 5)) / 5
    regularizer = np.sqrt(settings["class_regularization"] / 5)
    cache_x, cache_result = None, None

    def compute(values):
        nonlocal cache_x, cache_result
        if cache_x is None or not np.array_equal(values, cache_x):
            x = reference.copy()
            x[free] = values
            prediction, jac = evaluate(x, subset, len(observed), jacobian=True)
            residual = (prediction - known - observed)[valid] / scale
            derivative = jac[valid][:, free] / scale[:, None]
            penalty = regularizer * center @ np.log(x[7:12])
            penalty_jac = np.zeros((5, len(free)))
            penalty_jac[:, 7:12] = regularizer * center / x[None, 7:12]
            cache_x = values.copy()
            cache_result = np.r_[residual, penalty], np.vstack([derivative, penalty_jac])
        return cache_result

    starts = [
        reference,
        np.r_[[15] * 7, [1.2] * 5, 0.3, 20, 0.15],
        np.r_[[29] * 7, [1.8] * 5, 0.1, 50, 0.35],
    ]
    fits = [
        least_squares(
            lambda v: compute(v)[0],
            np.clip(start[free], lower[free], upper[free]),
            jac=lambda v: compute(v)[1],
            bounds=(lower[free], upper[free]),
            x_scale="jac",
            max_nfev=settings["max_nfev"],
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        for start in starts
    ]
    converged = [fit for fit in fits if fit.success and np.isfinite(fit.fun).all()]
    finite = [fit for fit in fits if np.isfinite(fit.fun).all()]
    if not finite:
        raise RuntimeError("All independent class fits are nonfinite")
    best = min(converged or finite, key=lambda fit: fit.cost)
    x = reference.copy()
    x[free] = best.x
    return dict(
        parameters=x.tolist(),
        parameter_names=PARAMETERS,
        mode=mode,
        independent=True,
        nfev=int(best.nfev),
        objective=float(2 * best.cost),
        converged_starts=len(converged),
        success=bool(converged),
        solver_scale="jac",
        solver_tolerance=1e-7,
        active_bounds=best.active_mask.tolist(),
        training_indices=np.flatnonzero(keep).tolist(),
        forcing_indices=np.flatnonzero(forcing).tolist(),
        unique_observations=int(valid.sum()),
        starts=[
            dict(
                success=bool(f.success),
                cost=float(f.cost),
                nfev=int(f.nfev),
                optimality=float(f.optimality),
                message=str(f.message),
            )
            for f in fits
        ],
    )
