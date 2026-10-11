"""Separate strict station/year dominance from pooled relative-RMSE dominance."""

import numpy as np
import pandas as pd

from ecologyhydro.long_record import STATIONS

PRODUCTS = ("fine", "fine_level1", "copernicus", "worldcover")


def assess(frame, years=(2022, 2023, 2024)):
    """Compare all required observations, without rounding, dropping cells or a margin."""
    expected = pd.MultiIndex.from_product([years, STATIONS], names=["year", "station"])
    if set(frame["product"]) != set(PRODUCTS):
        raise ValueError("Exactly four independently calibrated representations required")
    errors, observed = {}, None
    for product in PRODUCTS:
        part = frame[frame["product"] == product].set_index(["year", "station"])
        if part.index.has_duplicates or set(part.index) != set(expected):
            raise ValueError("Missing, duplicate or unexpected station-years")
        part = part.reindex(expected)
        if not part.eligible.eq(True).all():
            raise ValueError("All comparison cells must be eligible")
        if (
            not np.isfinite(part[["predicted", "observed"]]).all().all()
            or (part.observed <= 0).any()
        ):
            raise ValueError("Invalid predictions/observations")
        if observed is not None and not np.array_equal(observed, part.observed.to_numpy()):
            raise ValueError("Different observations across products")
        observed = part.observed.to_numpy()
        errors[product] = 100 * (part.predicted - part.observed).abs() / part.observed
    columns = {}
    for product, error in errors.items():
        columns[product] = {
            f"{year}_{station}": float(error.loc[year, station])
            for year in years
            for station in STATIONS
        }
        columns[product].update(
            {f"{year}_总误差": float(np.sqrt(np.mean(error.loc[year] ** 2))) for year in years}
        )
        columns[product]["三年整体"] = float(np.sqrt(np.mean(error**2)))
    scores = pd.DataFrame(columns)
    details = []
    for key, row in scores.iterrows():
        for other in PRODUCTS[1:]:
            details.append(
                dict(
                    metric=key,
                    comparator=other,
                    fine_error=row.fine,
                    comparator_error=row[other],
                    advantage_percentage_points=row[other] - row.fine,
                    strictly_better=bool(row.fine < row[other]),
                    goal="overall" if key == "三年整体" else "each_item",
                )
            )
    details = pd.DataFrame(details)
    individual = details[details.goal == "each_item"]
    overall = details[details.goal == "overall"]
    all_comparators = individual.groupby("metric").strictly_better.all()
    outcome = dict(
        every_item_better=bool(individual.strictly_better.all()),
        overall_better=bool(overall.strictly_better.all()),
        winning_columns=int(all_comparators.sum()),
        total_columns=len(all_comparators),
        winning_pairwise_items=int(individual.strictly_better.sum()),
        total_pairwise_items=len(individual),
        minimum_margin_required=0.0,
        equality_counts_as_win=False,
        comparison_uses_rounding=False,
    )
    return outcome, scores, details


def assess_transfer(frame, margin=1.0, years=(2022, 2023, 2024)):
    """Check all same-source map comparisons; never average away a failed source."""
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Margin must be finite and nonnegative")
    if set(frame.parameter_source) != set(PRODUCTS):
        raise ValueError("Exactly four parameter sources required")
    matrices, comparisons, reference = {}, [], None
    for source in PRODUCTS:
        part = frame[frame.parameter_source == source]
        _, scores, _ = assess(part, years)
        observations = part.set_index(["product", "year", "station"]).observed.sort_index()
        if reference is not None and not observations.equals(reference):
            raise ValueError("Different observations across parameter sources")
        reference = observations
        pooled = scores.loc["三年整体"]
        matrices[source] = pooled
        for comparator in PRODUCTS[1:]:
            difference = float(pooled.fine - pooled[comparator])
            comparisons.append(
                dict(
                    parameter_source=source,
                    comparator=comparator,
                    fine_error=float(pooled.fine),
                    comparator_error=float(pooled[comparator]),
                    difference_percentage_points=difference,
                    within_margin=bool(difference <= margin),
                )
            )
    details = pd.DataFrame(comparisons)
    outcome = dict(
        transfer_within_margin=bool(details.within_margin.all()),
        passed_comparisons=int(details.within_margin.sum()),
        total_comparisons=len(details),
        margin_percentage_points=float(margin),
        maximum_difference_percentage_points=float(details.difference_percentage_points.max()),
        statistical_noninferiority_test=False,
    )
    return outcome, pd.DataFrame(matrices).T, details
