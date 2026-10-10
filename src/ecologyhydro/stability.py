"""Fixed-complexity and training-subset averaging experiments for annual runoff."""

from itertools import combinations

import numpy as np
from scipy.optimize import least_squares

from ecologyhydro.reach_loss import apply_reach_loss
from ecologyhydro.regional_calibration import ENDPOINTS, predict


def training_subsets(keep):
    """Delete one TRAINING year; an outer withheld year never enters a member fit."""
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (4,) or keep.sum() < 3:
        raise ValueError("Subset averaging requires three or four training years")
    return [tuple(c) for c in combinations(np.flatnonzero(keep).tolist(), int(keep.sum()) - 1)]


def runoff(fitted, data):
    """Natural yield minus one fitted reach loss; known accounts are applied outside."""
    return apply_reach_loss(
        predict(fitted["parameters"], data)[:, ENDPOINTS], fitted["annual_loss_1e8_m3"]
    )


def fit_stable(data, target, adjustment, keep, regions=4):
    """Fit three or four Zs, shared Kc and L with unchanged bounds and multistarts."""
    target, adjustment = np.asarray(target), np.asarray(adjustment)
    keep = np.asarray(keep, dtype=bool)
    if regions not in (3, 4) or keep.shape != (4,) or keep.sum() < 2:
        raise ValueError("Require three/four regions and at least two training years")
    if target.shape != (4, 6) or adjustment.shape != target.shape:
        raise ValueError("Require four development years by six stations")
    if not np.isfinite(target).all() or (target <= 0).any() or not np.isfinite(adjustment).all():
        raise ValueError("Invalid observations or known water accounts")

    def residual(parameters):
        modeled = apply_reach_loss(predict(parameters[:-1], data)[:, ENDPOINTS], parameters[-1])
        return ((modeled[keep] - adjustment[keep] - target[keep]) / target[keep]).ravel()

    candidates = [
        least_squares(
            residual,
            [z] * regions + [kc, loss],
            bounds=([1] * regions + [0.7, 0], [30] * regions + [2, 100]),
            max_nfev=150,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        for z, kc, loss in ((5, 1, 10), (20, 1.3, 30), (29, 1.8, 50))
    ]
    converged = [f for f in candidates if f.success and np.isfinite(f.fun).all()]
    if not converged:
        raise ValueError("No converged stability calibration")
    best = min(converged, key=lambda f: np.mean(f.fun**2))
    # Normalize parameter units before reporting singular values; diagnostics only.
    scales = np.array([29] * regions + [1.3, 100])
    return {
        "parameters": best.x[:-1].tolist(),
        "annual_loss_1e8_m3": float(best.x[-1]),
        "training_indices": np.flatnonzero(keep).tolist(),
        "active_bounds": best.active_mask.tolist(),
        "converged_starts": len(converged),
        "normalized_jacobian_singular_values": np.linalg.svd(
            best.jac * scales, compute_uv=False
        ).tolist(),
        "multistart_cost_range": [
            float(min(f.cost for f in converged)),
            float(max(f.cost for f in converged)),
        ],
    }


def stability_metrics(held_predictions, target, refits, full_prediction):
    """Separate out-of-year accuracy from delete-year prediction sensitivity.

    refits contains each outer fold evaluated on the same four-year forcing.
    Its dispersion is sensitivity, NOT a confidence/prediction interval.
    """
    held_predictions, target, refits, full_prediction = (
        np.asarray(a, dtype=float) for a in (held_predictions, target, refits, full_prediction)
    )
    if (
        target.shape != (4, 6)
        or held_predictions.shape != target.shape
        or full_prediction.shape != target.shape
        or refits.shape != (4, 4, 6)
        or not all(
            np.isfinite(a).all() for a in (held_predictions, target, refits, full_prediction)
        )
        or (target <= 0).any()
    ):
        raise ValueError("Invalid stability evaluation arrays")
    relative = (held_predictions - target) / target
    yearly_mape = np.abs(relative).mean(axis=1) * 100
    yearly_rmse = np.sqrt(np.mean(relative**2, axis=1)) * 100
    refit_change = (refits - full_prediction) / target
    return {
        "mape_pct": float(np.abs(relative).mean() * 100),
        "relative_rmse_pct": float(np.sqrt(np.mean(relative**2)) * 100),
        "worst_year_mape_pct": float(yearly_mape.max()),
        "worst_year_relative_rmse_pct": float(yearly_rmse.max()),
        "year_mape_std_pct": float(yearly_mape.std()),
        "worst_station_year_absolute_error_pct": float(np.abs(relative).max() * 100),
        "delete_year_prediction_rms_change_pct": float(np.sqrt(np.mean(refit_change**2)) * 100),
        "delete_year_prediction_max_change_pct": float(np.abs(refit_change).max() * 100),
        "negative_heldout_predictions": int((held_predictions < 0).sum()),
        "year_mape_pct": yearly_mape.tolist(),
        "year_relative_rmse_pct": yearly_rmse.tolist(),
    }
