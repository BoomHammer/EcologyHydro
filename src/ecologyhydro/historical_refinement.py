"""Past-only spatial shrinkage and recency weighting, with fixed class contrasts."""

import numpy as np
from scipy.optimize import least_squares

from ecologyhydro.class_response import evaluate
from ecologyhydro.regional_transfer import training_data
from ecologyhydro.separated_selection import expansion, fit_past


def spatial_penalty(values, strength):
    """Shrink log Z toward its spatial mean, without a target-fitted prior."""
    center = np.eye(7) - np.ones((7, 7)) / 7
    factor = np.sqrt(strength / 7)
    residual = factor * (center @ np.log(values[:7]))
    jac = np.zeros((7, 10))
    jac[:, :7] = factor * center / values[None, :7]
    return residual, jac


def fit_history(datasets, target, adjustment, eligible, keep, spec, max_nfev=150):
    """Use the same starts, bounds and observations for every representation."""
    ridge, half_life = spec["ridge"], spec["half_life_years"]
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("Invalid spatial penalty")
    if half_life is not None and (half_life <= 0 or not np.isfinite(half_life)):
        raise ValueError("Invalid recency half-life")
    if ridge == 0 and half_life is None:
        return fit_past(datasets, target, adjustment, eligible, keep, "regional", max_nfev)
    keep = np.asarray(keep, dtype=bool)
    if not keep.any() or not datasets:
        raise ValueError("No historical training data")
    horizon = int(np.flatnonzero(keep)[-1]) + 1
    subset = {
        name: training_data(data, np.arange(len(keep)) < horizon) for name, data in datasets.items()
    }
    target, adjustment, eligible = (np.asarray(a)[:horizon] for a in (target, adjustment, eligible))
    valid = eligible & keep[:horizon, None]
    observed, known = target[valid], adjustment[valid]
    if not len(observed) or (observed <= 0).any() or not np.isfinite(observed + known).all():
        raise ValueError("Invalid historical observations")
    # Forcing is contiguous annual data; weights use elapsed years, including bridge years.
    age = horizon - 1 - np.arange(horizon)
    weights = np.ones(horizon) if half_life is None else np.exp2(-age / half_life)
    cell_weights = np.broadcast_to(weights[:, None], target.shape)[valid]
    scale = np.sqrt(cell_weights / (cell_weights.sum() * len(subset))) / observed
    matrix, offset = expansion("regional")
    lower, upper = np.r_[[1] * 7, 0.7, 0, 0], np.r_[[30] * 7, 2, 100, 0.5]
    cached_values, cached_result = None, None

    def compute(values):
        nonlocal cached_values, cached_result
        if cached_values is None or not np.array_equal(values, cached_values):
            residuals, derivatives = [], []
            for data in subset.values():
                pred, jac = evaluate(matrix @ values + offset, data, horizon, True)
                residuals.append((pred[valid] - known - observed) * scale)
                derivatives.append((jac[valid] @ matrix) * scale[:, None])
            penalty, penalty_jac = spatial_penalty(values, ridge)
            residuals.append(penalty)
            derivatives.append(penalty_jac)
            cached_values = values.copy()
            cached_result = np.concatenate(residuals), np.vstack(derivatives)
        return cached_result

    fitted = [
        least_squares(
            lambda v: compute(v)[0],
            np.r_[[z] * 7, kc, loss, retention],
            jac=lambda v: compute(v)[1],
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        for z, kc, loss, retention in ((5, 1, 0, 0.05), (15, 1.2, 20, 0.15), (29, 1.8, 50, 0.35))
    ]
    converged = [f for f in fitted if f.success and np.isfinite(f.fun).all()]
    if not converged:
        raise RuntimeError("No declared start converged")
    best = min(converged, key=lambda f: f.cost)
    return dict(
        parameters=(matrix @ best.x + offset).tolist(),
        n_free=len(best.x),
        objective=float(2 * best.cost),
        success=True,
        fit_indices=np.flatnonzero(keep).tolist(),
        forcing_horizon=horizon,
        unique_observations=int(valid.sum()),
        product_count=len(subset),
        training_year_weights=weights[keep[:horizon]].tolist(),
        starts=[
            dict(success=bool(f.success), nfev=int(f.nfev), objective=float(2 * f.cost))
            for f in fitted
        ],
    )


def choose_history(records, candidates):
    """Rank all four maps on held historical observations; never rank fine-map wins."""
    scores = []
    for index, mode in enumerate(candidates):
        rows = [r for r in records if r["mode"] == mode]
        if not rows or any(not r["success"] for r in rows):
            continue
        errors = np.concatenate([r["relative_errors"] for r in rows])
        if not len(errors) or not np.isfinite(errors).all():
            raise ValueError("Invalid historical selection errors")
        scores.append((float(np.mean(errors**2)), index, mode))
    if not scores:
        raise ValueError("No converged historical candidate")
    loss, _, mode = min(scores)
    return dict(
        mode=mode,
        cv_error_pct=float(100 * np.sqrt(loss)),
        n_free=10,
        selection_uses_target_years=False,
    )
