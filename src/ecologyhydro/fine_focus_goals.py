"""User's revised three-table criteria, without changing historical goal reports."""

import numpy as np

from ecologyhydro.accuracy_goals import PRODUCTS, assess
from ecologyhydro.long_record import STATIONS


def assess_focus(frame, years=(2022, 2023, 2024)):
    _, scores, detail = assess(frame, years)
    aggregate_keys = [f"{year}_总误差" for year in years] + ["三年整体"]
    station_keys = [f"{year}_{station}" for year in years for station in STATIONS]
    best = scores[list(PRODUCTS[1:])].gt(scores.fine, axis=0).all(axis=1)
    wins = int(best.loc[station_keys].sum())
    required = int(np.ceil(0.4 * len(station_keys)))
    outcome = dict(
        aggregate_all_best=bool(best.loc[aggregate_keys].all()),
        aggregate_best_count=int(best.loc[aggregate_keys].sum()),
        aggregate_total=len(aggregate_keys),
        station_year_40_percent_best=bool(wins >= required),
        station_year_wins=wins,
        station_year_total=len(station_keys),
        station_year_required=required,
        station_year_win_fraction=wins / len(station_keys),
        transfer_is_secondary=True,
        ties_count_as_win=False,
    )
    return outcome, scores, detail
