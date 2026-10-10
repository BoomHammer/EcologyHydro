"""Conditional, unidentified net reach loss; never a measured groundwater flux."""

import numpy as np
from scipy.optimize import least_squares

from ecologyhydro.regional_calibration import ENDPOINTS, predict


def apply_reach_loss(flow, loss):
    """Subtract one annual Lanzhou--Toudaoguai loss once at all downstream endpoints."""
    flow = np.asarray(flow, dtype=float)
    if flow.ndim != 2 or flow.shape[1] != 6 or not np.isfinite(flow).all():
        raise ValueError("Expected finite year-by-six-endpoint runoff")
    if not np.isscalar(loss) or not np.isfinite(loss) or loss < 0:
        raise ValueError("Expected a finite nonnegative annual net loss")
    result = flow.copy()
    result[:, 2:] -= loss
    return result


def fit_reach_loss(data, target, adjustment, keep=None):
    """Four Zs, Kc multiplier, and one constant reach loss; development years only."""
    target, adjustment = np.asarray(target), np.asarray(adjustment)
    if target.shape != (4, 6) or adjustment.shape != target.shape:
        raise ValueError("Require four development years and six common endpoints")
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Invalid observed runoff")
    if not np.isfinite(adjustment).all():
        raise ValueError("Invalid water adjustment")
    keep = np.ones(4, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    if keep.shape != (4,) or not keep.any():
        raise ValueError("Invalid development-year mask")

    def residual(parameters):
        base = predict(parameters[:5], data)[:, ENDPOINTS] - adjustment
        modeled = apply_reach_loss(base, parameters[5])
        return ((modeled - target) / target)[keep].ravel()

    candidates = [
        least_squares(
            residual,
            [z] * 4 + [kc, loss],
            bounds=([1] * 4 + [0.7, 0], [30] * 4 + [2, 100]),
            max_nfev=150,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        for z, kc, loss in ((5, 1, 10), (20, 1.3, 30), (29, 1.8, 50))
    ]
    converged = [f for f in candidates if f.success and np.isfinite(f.fun).all()]
    if not converged:
        raise ValueError("No converged reach-loss calibration")
    best = min(converged, key=lambda f: np.mean(f.fun**2))
    return {
        "parameters": best.x[:5].tolist(),
        "annual_loss_1e8_m3": float(best.x[5]),
        "relative_rmse_pct": float(np.sqrt(np.mean(best.fun**2)) * 100),
        "jacobian_singular_values": np.linalg.svd(best.jac, compute_uv=False).tolist(),
        "active_bounds": best.active_mask.tolist(),
        "converged_starts": len(converged),
    }
