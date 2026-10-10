"""Small, conservative monthly headwater experiment; not official InVEST AWY."""

import calendar

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter


def simulate(parameters, forcing, units, initial_fraction=0.5):
    """Snow, finite soil bucket, and linear drainage store, all depths in mm.

    Rain/snow partition is fixed linear between -1 and +1 C. Melt is limited
    by snow availability, using a fixed 3 mm/C/day factor. Soil receives liquid
    input, excess spills, evaporation withdraws available water exponentially.
    Excess enters a linear reservoir. No fitted precipitation correction.
    """
    capacity_scale, et_scale, tau = parameters
    p, pet, temperature = (np.asarray(forcing[k]) for k in ("p", "pet", "temperature"))
    capacity = np.maximum(units["awc"] * capacity_scale, 0.1)
    vegetation = units["veg"] == 1
    soil = np.where(vegetation, capacity * initial_fraction, 0.0)
    snow = 0.0
    records, excesses = [], []
    for t in range(len(p)):
        old_soil, old_snow = soil.copy(), snow
        snowfall = p[t] * np.clip((1 - temperature[t]) / 2, 0, 1)
        snow += snowfall
        days = calendar.monthrange(int(forcing["year"][t]), int(forcing["month"][t]))[1]
        melt = min(snow, max(temperature[t], 0) * 3 * days)
        snow -= melt
        liquid = p[t] - snowfall + melt
        available = soil + liquid
        excess = np.maximum(available - capacity, 0)
        soil = np.minimum(available, capacity)
        kc = np.where(units["veg"] == 1, np.minimum(units["kc"][t] * et_scale, 1.3), units["kc"][t])
        aet = soil * (-np.expm1(-kc * pet[t] / capacity))
        soil -= aet
        # Nonvegetated surfaces retain the AWY supply-limited Kc*PET convention.
        aet = np.where(vegetation, aet, np.minimum(available, kc * pet[t]))
        excess = np.where(vegetation, excess, available - aet)
        soil = np.where(vegetation, soil, 0.0)
        weights = units["weights"]
        excesses.append(float(excess @ weights))
        records.append(
            [
                p[t],
                float(aet @ weights),
                float(soil @ weights),
                snow,
                float((soil - old_soil) @ weights),
                snow - old_snow,
            ]
        )
    fraction = -np.expm1(-1 / tau)
    runoff = lfilter([fraction], [1, -(1 - fraction)], excesses)
    reservoir = runoff * (1 - fraction) / fraction
    delta_reservoir = np.diff(np.r_[0.0, reservoir])
    values = np.asarray(records)
    closure = values[:, 0] - values[:, 1] - runoff - values[:, 4] - values[:, 5] - delta_reservoir
    if np.max(np.abs(closure)) > 1e-8:
        raise ValueError("Monthly water balance does not close")
    return dict(
        p=values[:, 0],
        aet=values[:, 1],
        soil=values[:, 2],
        snow=values[:, 3],
        delta_soil=values[:, 4],
        delta_snow=values[:, 5],
        q=runoff,
        reservoir=reservoir,
        delta_reservoir=delta_reservoir,
        closure=closure,
    )


def annual_bucket(parameters, forcing, units, years, initial_fraction=0.5):
    result = simulate(parameters, forcing, units, initial_fraction)
    return np.array([result["q"][forcing["year"] == y].sum() for y in years]) * units["volume"]


def annual_awy(parameters, data):
    """Same Budyko equations, retaining sampled spatial input heterogeneity."""
    z, scale = parameters
    kc = np.where(data["veg"] == 1, np.minimum(data["kc"] * scale, 1.3), data["kc"])
    phi = kc * data["et"] / data["p"]
    omega = np.minimum(1.25 + z * data["awc"] / data["p"], 5)
    fraction = np.minimum(phi, 1 + phi - (1 + phi**omega) ** (1 / omega))
    fraction = np.where(data["veg"] == 1, fraction, np.minimum(phi, 1))
    return np.sum((1 - fraction) * data["p"] * data["weights"], axis=1)


def fit_model(predict, observed, adjustment, keep, model):
    """Fit only explicit training targets, using deterministic multiple starts."""
    if model == "monthly":
        lower, upper = [0.25, 0.4, 0.1], [4.0, 1.6, 12.0]
        starts = [[1, 1, 1], [0.5, 0.7, 6], [2, 1.3, 3]]
    else:
        lower, upper = [0.1, 0.4], [30, 1.6]
        starts = [[5, 1], [15, 0.7], [1, 1.3]]
    indices = np.flatnonzero(keep)
    if not len(indices) or not np.isfinite(observed[indices] + adjustment[indices]).all():
        raise ValueError("Invalid training targets")

    def residual(x):
        return (predict(x)[indices] - adjustment[indices] - observed[indices]) / observed[indices]

    fits = [
        least_squares(
            residual, x, bounds=(lower, upper), max_nfev=160, ftol=1e-8, xtol=1e-8, gtol=1e-8
        )
        for x in starts
    ]
    successful = [f for f in fits if f.success]
    if not successful:
        raise RuntimeError("No optimizer start converged")
    best = min(successful, key=lambda f: np.sum(f.fun**2))
    scaled = (best.x - lower) / (np.array(upper) - lower)
    return dict(
        parameters=best.x.tolist(),
        training_indices=indices.tolist(),
        train_relative_rmse_pct=float(np.sqrt(np.mean(best.fun**2)) * 100),
        bound_parameters=np.flatnonzero((scaled < 0.01) | (scaled > 0.99)).tolist(),
        starts=[
            dict(success=bool(f.success), cost=float(2 * f.cost), evaluations=int(f.nfev))
            for f in fits
        ],
    )
