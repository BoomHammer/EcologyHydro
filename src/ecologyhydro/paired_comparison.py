"""Matched station-year comparisons; ties and simultaneous wins are explicit."""

import numpy as np
import pandas as pd


def paired_scores(frame, reference="fine", rivals=("copernicus", "worldcover", "fine_level1")):
    keys = ["year", "station"]
    parts = {}
    for product in (reference, *rivals):
        part = frame[(frame["product"] == product) & frame.eligible].copy()
        if part.duplicated(keys).any():
            raise ValueError("Duplicated station-year predictions")
        part = part.set_index(keys).sort_index()
        if not np.isfinite(part[["predicted", "observed"]].to_numpy()).all():
            raise ValueError("Nonfinite scored pair")
        parts[product] = part
    base = parts[reference]
    if base.empty:
        raise ValueError("No eligible station-year pairs")
    errors = pd.DataFrame(index=base.index)
    for product, part in parts.items():
        if not part.index.equals(base.index):
            raise ValueError("Pair sets differ; do not discard unmatched observations")
        if not np.array_equal(base.observed, part.observed):
            raise ValueError("Observed runoff differs between compared maps")
        errors[product] = (part.predicted - part.observed).abs()
    records = []
    for rival in rivals:
        delta = errors[rival] - errors[reference]
        records.append(
            dict(
                rival=rival,
                n=len(delta),
                wins=int((delta > 1e-8).sum()),
                ties=int((delta.abs() <= 1e-8).sum()),
                losses=int((delta < -1e-8).sum()),
                win_fraction=float((delta > 1e-8).mean()),
            )
        )
    both = (errors[list(rivals[:2])].sub(errors[reference], axis=0) > 1e-8).all(axis=1)
    records.append(
        dict(
            rival="both_products",
            n=len(both),
            wins=int(both.sum()),
            ties=None,
            losses=None,
            win_fraction=float(both.mean()),
        )
    )
    return records, errors.reset_index()
