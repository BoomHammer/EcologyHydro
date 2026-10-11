"""Joint AWY and annual carryover calibration, with analytic state derivatives."""

import numpy as np
from scipy.optimize import least_squares

from ecologyhydro.regional_transfer import evaluate


def evaluate_storage(parameters, data, years, jacobian=False):
    """Nine AWY/accounting parameters followed by common retention in [0, .5]."""
    x = np.asarray(parameters, dtype=float)
    awy = x[:9].copy()
    awy[-1] = 0
    if jacobian:
        natural, derivative = evaluate(awy, data, "reach7", years, jacobian=True)
        derivative[:, :, -1] = 0
        derivative = np.concatenate([derivative, np.zeros((years, 7, 1))], axis=2)
    else:
        natural = evaluate(awy, data, "reach7", years)
    retention = x[-1]
    state = retention / (1 - retention) * natural[0]
    predicted = np.empty_like(natural)
    if jacobian:
        state_jac = retention / (1 - retention) * derivative[0]
        state_jac[:, -1] += natural[0] / (1 - retention) ** 2
        output_jac = np.empty_like(derivative)
    for yi, value in enumerate(natural):
        available = state + value
        predicted[yi] = (1 - retention) * available
        if jacobian:
            available_jac = state_jac + derivative[yi]
            output_jac[yi] = (1 - retention) * available_jac
            output_jac[yi, :, -1] -= available
            state_jac = retention * available_jac
            state_jac[:, -1] += available
        state = retention * available
    predicted[:, 2:] -= x[8]
    if jacobian:
        output_jac[:, 2:, 8] = -1
        return predicted, output_jac
    return predicted


def fit_storage(data, target, adjustment, eligible, keep, seed):
    """Chronological forcing must already be truncated at the training horizon."""
    mask = np.asarray(eligible) & np.asarray(keep)[:, None]
    observed, known = target[mask].copy(), adjustment[mask].copy()
    if len(observed) < 10 or not np.isfinite(observed + known).all() or np.any(observed <= 0):
        raise ValueError("Insufficient finite training observations")
    cache_x, cache_result = None, None

    def compute(x):
        nonlocal cache_x, cache_result
        if cache_x is None or not np.array_equal(cache_x, x):
            prediction, jac = evaluate_storage(x, data, len(target), jacobian=True)
            scale = observed * np.sqrt(len(observed))
            cache_result = (
                (prediction[mask] - known - observed) / scale,
                jac[mask] / scale[:, None],
            )
            cache_x = x.copy()
        return cache_result

    starts = [list(seed) + [0.15], [20] * 7 + [1.3, 30, 0.3], [29] * 7 + [1.8, 50, 0.45]]
    fits = [
        least_squares(
            lambda x: compute(x)[0],
            start,
            jac=lambda x: compute(x)[1],
            bounds=([1] * 7 + [0.7, 0, 0], [30] * 7 + [2, 100, 0.5]),
            max_nfev=150,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        for start in starts
    ]
    converged = [fit for fit in fits if fit.success and np.isfinite(fit.fun).all()]
    if not converged:
        raise RuntimeError("No converged joint storage fit")
    best = min(converged, key=lambda fit: fit.cost)
    scaled = compute(best.x)[1] * np.array([29] * 7 + [1.3, 100, 0.5])
    return dict(
        parameters=best.x.tolist(),
        active_bounds=best.active_mask.tolist(),
        observations=len(observed),
        converged_starts=len(converged),
        nfev=best.nfev,
        relative_rmse=float(np.sqrt(2 * best.cost)),
        data_jacobian_singular_values=np.linalg.svd(scaled, compute_uv=False).tolist(),
    )
