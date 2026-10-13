"""From-scratch, past-only calibration without adding fine-class parameter contrasts."""

import numpy as np
from scipy.optimize import least_squares

from ecologyhydro.class_response import evaluate
from ecologyhydro.regional_transfer import training_data


def expansion(mode):
    """Keep the existing profile library; fit one common vegetation multiplier."""
    if mode not in ("global", "regional"):
        raise ValueError("Unknown declared candidate")
    nz = 1 if mode == "global" else 7
    matrix = np.zeros((15, nz + 3))
    matrix[np.arange(7), np.zeros(7, int) if nz == 1 else np.arange(7)] = 1
    matrix[7:12, nz] = 1
    matrix[13, nz + 1] = 1
    matrix[14, nz + 2] = 1
    offset = np.zeros(15)
    offset[12] = 0.15
    return matrix, offset


def fit_past(datasets, target, adjustment, eligible, keep, mode, max_nfev=150):
    """Equal map weights, fixed starts, chronological states, no inherited fitted priors."""
    keep = np.asarray(keep, dtype=bool)
    if not keep.any():
        raise ValueError("No calibration years")
    horizon = int(np.flatnonzero(keep)[-1]) + 1
    forcing = np.arange(len(keep)) < horizon
    subset = {name: training_data(data, forcing) for name, data in datasets.items()}
    target, adjustment, eligible = (np.asarray(a)[:horizon] for a in (target, adjustment, eligible))
    valid = eligible & keep[:horizon, None]
    observed, known = target[valid], adjustment[valid]
    if not len(observed) or (observed <= 0).any() or not np.isfinite(observed + known).all():
        raise ValueError("Invalid training observations")
    matrix, offset = expansion(mode)
    nz = matrix.shape[1] - 3
    scale = observed * np.sqrt(valid.sum() * len(subset))
    lower, upper = np.r_[[1] * nz, 0.7, 0, 0], np.r_[[30] * nz, 2, 100, 0.5]
    cached_values, cached_result = None, None

    def compute(values):
        nonlocal cached_values, cached_result
        if cached_values is None or not np.array_equal(values, cached_values):
            residuals, derivatives = [], []
            for data in subset.values():
                pred, jac = evaluate(matrix @ values + offset, data, horizon, True)
                residuals.append((pred[valid] - known - observed) / scale)
                derivatives.append((jac[valid] @ matrix) / scale[:, None])
            cached_values = values.copy()
            cached_result = np.concatenate(residuals), np.vstack(derivatives)
        return cached_result

    starts = [
        np.r_[[z] * nz, kc, loss, r]
        for z, kc, loss, r in (
            (5, 1, 0, 0.05),
            (15, 1.2, 20, 0.15),
            (29, 1.8, 50, 0.35),
        )
    ]
    fitted = [
        least_squares(
            lambda v: compute(v)[0],
            initial,
            jac=lambda v: compute(v)[1],
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        for initial in starts
    ]
    converged = [f for f in fitted if f.success and np.isfinite(f.fun).all()]
    if not converged:
        raise RuntimeError("No declared start converged")
    best = min(converged, key=lambda f: f.cost)
    return dict(
        parameters=(matrix @ best.x + offset).tolist(),
        mode=mode,
        n_free=len(best.x),
        objective=float(2 * best.cost),
        success=True,
        fit_indices=np.flatnonzero(keep).tolist(),
        forcing_horizon=horizon,
        unique_observations=int(valid.sum()),
        product_count=len(subset),
        starts=[
            dict(success=bool(f.success), nfev=int(f.nfev), objective=float(2 * f.cost))
            for f in fitted
        ],
    )


def choose(rows, modes):
    """Select only from supplied training-fold losses, without target-period results."""
    candidates = []
    for mode in modes:
        selected = [r for r in rows if r["mode"] == mode]
        if not selected or any(not r["success"] for r in selected):
            continue
        error = np.concatenate([r["relative_errors"] for r in selected])
        candidates.append((float(np.mean(error**2)), expansion(mode)[0].shape[1], mode))
    if not candidates:
        raise ValueError("No eligible training-fold candidate")
    error, count, mode = min(candidates)
    return dict(
        mode=mode,
        cv_error_pct=100 * np.sqrt(error),
        n_free=count,
        selection_uses_target_years=False,
    )
