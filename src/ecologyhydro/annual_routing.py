"""Exploratory, mass-conserving annual storage and conditional station accounting."""

import numpy as np
from scipy.optimize import least_squares


def route(natural, parameters, initial="equilibrium"):
    """Return delivery, storage change and disconnected water, all as reach volumes.

    Parameters are upper-reach loss, LTG connection fraction, annual retention,
    and lower-reach loss. Constant losses are applied separately in prediction.
    Storage is conceptual catchment carryover, not observed reservoir storage.
    """
    natural = np.asarray(natural, dtype=float)
    _, fraction, retention, _ = parameters
    if natural.ndim != 2 or natural.shape[1] != 7 or not np.isfinite(natural).all():
        raise ValueError("Natural reach volumes must be finite with seven columns")
    if len(natural) == 0 or np.any(natural < -1e-8):
        raise ValueError("Natural reach volumes must be nonnegative and nonempty")
    if not 0 <= fraction <= 1 or not 0 <= retention <= 0.5:
        raise ValueError("Process parameters outside frozen bounds")
    connected = natural.copy()
    connected[:, 2] *= fraction
    disconnected = natural - connected
    if initial == "equilibrium":
        state = connected[0] * retention / (1 - retention)
    elif initial == "zero":
        state = np.zeros(7)
    else:
        raise ValueError("Unknown initial condition")
    delivered, change = np.empty_like(natural), np.empty_like(natural)
    for i, value in enumerate(connected):
        next_state = retention * (state + value)
        delivered[i] = (1 - retention) * (state + value)
        change[i] = next_state - state
        state = next_state
    return delivered, change, disconnected


def predict(natural, adjustment, parameters, initial="equilibrium"):
    delivered, _, _ = route(natural, parameters, initial)
    result = delivered.cumsum(axis=1) - adjustment
    result[:, 2:] -= parameters[0]
    result[:, 6] -= parameters[3]
    return result


def fit_process(natural, target, adjustment, eligible, keep, baseline_loss, free):
    """Fit only explicitly supplied training observations; forcing stays chronological.

    Caller must truncate forcing at the last calibration year. Held-out-year forcing
    is needed for state continuity, but held-out observations never enter residuals.
    """
    mask = np.asarray(eligible) & np.asarray(keep)[:, None]
    observed = np.asarray(target)[mask].copy()
    if not len(observed) or not np.isfinite(observed).all() or np.any(observed <= 0):
        raise ValueError("Finite positive calibration observations required")
    safe_adjustment = np.zeros_like(adjustment)
    safe_adjustment[mask] = np.asarray(adjustment)[mask]
    if not np.isfinite(safe_adjustment).all():
        raise ValueError("Missing calibration accounts")
    reference = np.array([baseline_loss, 1.0, 0.0, 0.0])
    if not free:
        return {"parameters": reference.tolist(), "observations": len(observed)}
    lower, upper = np.array([0, 0, 0, 0]), np.array([100, 1, 0.5, 50])
    indices = np.asarray(free, dtype=int)

    def unpack(values):
        parameters = reference.copy()
        parameters[indices] = values
        return parameters

    def residual(values):
        return (predict(natural, safe_adjustment, unpack(values))[mask] - observed) / observed

    solutions = []
    for fraction, retention, lower_loss in ((1, 0, 0), (0.7, 0.2, 10), (0.3, 0.45, 30)):
        start = np.array([baseline_loss, fraction, retention, lower_loss])[indices]
        solutions.append(
            least_squares(
                residual,
                start,
                bounds=(lower[indices], upper[indices]),
                max_nfev=150,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
        )
    best = min(solutions, key=lambda item: np.sum(item.fun**2))
    if not best.success:
        raise RuntimeError("Selected process fit did not converge")
    return {
        "parameters": unpack(best.x).tolist(),
        "observations": len(observed),
        "active_bounds": best.active_mask.tolist(),
        "nfev": best.nfev,
        "converged_starts": sum(bool(item.success) for item in solutions),
        "relative_rmse": float(np.sqrt(np.mean(best.fun**2))),
    }
