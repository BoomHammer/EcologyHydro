"""Audit signed station/reach residuals and fixed-yield storage sensitivity."""

from pathlib import Path

import numpy as np
import pandas as pd
from run_landcover_comparison import worker_setup
from scipy.optimize import least_squares

from ecologyhydro.baseline import read_json
from ecologyhydro.comparison import accounts
from ecologyhydro.fine_focus_goals import assess_focus
from ecologyhydro.fine_priority import regional_delivery
from ecologyhydro.long_record import STATIONS
from ecologyhydro.simulation import write_json


def main():
    reference = Path("experiments/calibration_followup/20261011T154702_556627Z")
    output = Path("experiments/fine_2023_focus/response_diagnosis")
    output.mkdir(exist_ok=True)
    config = worker_setup(reference / "execution_protocol.yaml")
    target, adjustment, eligible, _, _ = accounts(config)
    natural = np.array(
        [read_json(reference / f"natural_fine_{y}.json")["fine"] for y in config["forcing_years"]]
    )
    frozen = read_json(reference / "freeze_fine.json")
    keep = np.isin(config["forcing_years"], [2022, 2023, 2024])

    def prediction(x):
        flow, _ = regional_delivery(natural, x[:7])
        flow[:, 2:] -= x[7]
        return flow - adjustment

    initial = np.r_[frozen["retention"], frozen["seed"][13]]
    base = prediction(initial)
    rows = []
    for yi in np.flatnonzero(keep):
        residual = base[yi] - target[yi]
        local = np.diff(residual, prepend=0)
        for si, station in enumerate(STATIONS):
            rows.append(
                dict(
                    year=config["forcing_years"][yi],
                    station=station,
                    observed=target[yi, si],
                    predicted=base[yi, si],
                    residual=residual[si],
                    local_residual=local[si],
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "residuals.csv", index=False, encoding="utf-8-sig")
    print(frame.to_string(index=False))
    valid = eligible & keep[:, None]
    fit = least_squares(
        lambda x: ((prediction(x) - target) / target)[valid],
        initial,
        bounds=([0] * 8, [0.5] * 7 + [100]),
        max_nfev=500,
    )
    result = dict(
        parameters=fit.x.tolist(),
        success=bool(fit.success),
        annual_errors=(
            100 * np.sqrt(np.mean(((prediction(fit.x) - target) / target)[keep] ** 2, axis=1))
        ).tolist(),
        development_target_fit=True,
    )
    write_json(output / "storage_only.json", result)
    print(result)
    combined = pd.read_csv(reference / "full_predictions.csv")
    for mode, parameters in [("joint", fit.x), ("retention_only", None)]:
        if parameters is None:
            fitted = least_squares(
                lambda x: ((prediction(np.r_[x, initial[7]]) - target) / target)[valid],
                initial[:7],
                bounds=(0, 0.5),
                max_nfev=500,
            )
            parameters = np.r_[fitted.x, initial[7]]
        values = prediction(parameters)
        combined.loc[combined["product"] == "fine", "predicted"] = values[keep].ravel()
        outcome, scores, _ = assess_focus(combined)
        print(mode, outcome)
        print(scores.tail(4).to_string())
        write_json(output / f"{mode}_goals.json", dict(parameters=parameters.tolist(), **outcome))


if __name__ == "__main__":
    main()
