"""Verify all three research stages and report failures without changing gates."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.long_record import STATIONS
from ecologyhydro.simulation import write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def metrics(frame):
    error = frame.predicted.to_numpy() - frame.observed.to_numpy()
    relative = error / frame.observed.to_numpy() * 100
    if not np.isfinite(relative).all():
        raise ValueError("Nonfinite scored result")
    return dict(
        n=len(frame),
        relative_rmse_pct=float(np.sqrt(np.mean(relative**2))),
        mape_pct=float(np.abs(relative).mean()),
        worst_pct=float(np.abs(relative).max()),
        mean_error=float(error.mean()),
        mae=float(np.abs(error).mean()),
    )


def main():
    root = project_root()
    output = root / "project/calibration/monthly_pilot_v1"
    run = read_json(output / "run_summary.json")
    for source in run["sources"]:
        if fingerprint(Path(source["path"]))["sha256"] != source["sha256"]:
            raise ValueError(f"Source changed: {source['path']}")
    frame = pd.concat([pd.read_csv(output / f"stations_{p}.csv") for p in ("fine", "copernicus")])
    sensitivity = pd.concat(
        [pd.read_csv(output / f"sensitivity_{p}.csv") for p in ("fine", "copernicus")]
    )
    valid = frame[frame.year.between(2019, 2022) & frame.eligible]
    stats = []
    for keys, group in valid.groupby(["landcover", "phase", "model"]):
        for scope, part in [("all7", group), *[(s, group[group.station == s]) for s in STATIONS]]:
            stats.append(
                dict(
                    zip(("landcover", "phase", "model"), keys, strict=True),
                    scope=scope,
                    **metrics(part),
                )
            )
    stats = pd.DataFrame(stats)
    stats.to_csv(output / "validation_metrics.csv", index=False)
    frame.to_csv(output / "all_station_results.csv", index=False)
    # Recompute errors and assert every prediction was routed by a single Guide change.
    np.testing.assert_allclose(frame.error, frame.predicted - frame.observed, equal_nan=True)
    np.testing.assert_allclose(
        frame.relative_error_pct, 100 * (frame.predicted / frame.observed - 1), equal_nan=True
    )
    for (_product, _phase, year), group in frame.groupby(["landcover", "phase", "year"]):
        original = group[group.model == "existing_reach7"].set_index("station").loc[STATIONS]
        for model in ("annual_local", "monthly"):
            candidate = group[group.model == model].set_index("station").loc[STATIONS]
            finite = np.isfinite(original.predicted)
            np.testing.assert_allclose(
                candidate.predicted[finite] - original.predicted[finite],
                candidate.guide_change[finite],
                atol=1e-9,
            )
            np.testing.assert_allclose(
                np.diff(candidate.predicted), np.diff(original.predicted), atol=1e-9, equal_nan=True
            )
        if year >= 2019:
            for years in group.training_years:
                if max(map(int, years.split(";"))) >= year:
                    raise ValueError("Future year in calibration")
    gates, stability = [], []
    for product in ("fine", "copernicus"):
        lookup = stats[stats.landcover == product].set_index(["phase", "model", "scope"])
        deletion = sensitivity[
            (sensitivity.landcover == product)
            & (sensitivity.type == "delete_training_year")
            & sensitivity.year.between(2019, 2022)
        ]
        spreads = {}
        for model in ("annual_local", "monthly"):
            spread = (
                deletion[deletion.model == model]
                .groupby("year")
                .change_pct_observed.agg(lambda x: x.max() - x.min())
            )
            spreads[model] = float(spread.mean())
            for year, value in spread.items():
                stability.append(
                    dict(
                        landcover=product,
                        model=model,
                        year=int(year),
                        prediction_spread_pct_observed=value,
                    )
                )
        initial = sensitivity[
            (sensitivity.landcover == product)
            & (sensitivity.type == "initial_soil")
            & (sensitivity.fit == "early")
            & sensitivity.year.between(2019, 2022)
        ]
        initial_max = float(initial.change_pct_observed.abs().max())
        tests = {}
        for phase in ("fixed_early", "rolling"):
            monthly = lookup.loc[(phase, "monthly", "贵得")]
            for reference in ("existing_reach7", "annual_local"):
                baseline = lookup.loc[(phase, reference, "贵得")]
                tests[f"{phase}_10pct_improvement_vs_{reference}"] = bool(
                    monthly.relative_rmse_pct <= baseline.relative_rmse_pct * 0.9
                )
                tests[f"{phase}_worst_vs_{reference}"] = bool(
                    monthly.worst_pct <= baseline.worst_pct + 2
                )
                tests[f"{phase}_all7_vs_{reference}"] = bool(
                    lookup.loc[(phase, "monthly", "all7"), "relative_rmse_pct"]
                    <= lookup.loc[(phase, reference, "all7"), "relative_rmse_pct"] * 1.02
                )
        tests["initial_state"] = initial_max <= 2
        tests["delete_year_stability"] = spreads["monthly"] <= 1.1 * spreads["annual_local"]
        gates.append(
            dict(
                landcover=product,
                tests=tests,
                passed=all(tests.values()),
                initial_max_pct=initial_max,
                mean_delete_year_spread_pct=spreads,
            )
        )
    pd.DataFrame(stability).to_csv(output / "delete_year_stability.csv", index=False)
    # Interval errors are in volume, not percentages of potentially negative increments.
    reaches, conditions = [], []
    original = valid[valid.model == "existing_reach7"]
    for (product, phase, year), group in original.groupby(["landcover", "phase", "year"]):
        ordered = group.set_index("station").loc[STATIONS]
        pred, obs = ordered.predicted.to_numpy(), ordered.observed.to_numpy()
        pi, oi = np.diff(np.r_[0.0, pred]), np.diff(np.r_[0.0, obs])
        for i, station in enumerate(STATIONS):
            reaches.append(
                dict(
                    landcover=product,
                    phase=phase,
                    year=int(year),
                    reach=f"{STATIONS[i - 1] if i else '源头'}—{station}",
                    predicted_increment=pi[i],
                    observed_increment=oi[i],
                    incremental_error=pi[i] - oi[i],
                    cumulative_error=pred[i] - obs[i],
                )
            )
        for boundary, i in (("Guide_observed", 0), ("Lanzhou_observed", 1)):
            corrected = pred + obs[i] - pred[i]
            for j in range(2, 7):
                conditions.append(
                    dict(
                        landcover=product,
                        phase=phase,
                        year=int(year),
                        boundary=boundary,
                        station=STATIONS[j],
                        observed=obs[j],
                        predicted=corrected[j],
                    )
                )
    reaches = pd.DataFrame(reaches)
    reaches.to_csv(output / "reach_increment_errors.csv", index=False)
    reach_summary = (
        reaches.groupby(["landcover", "phase", "reach"])
        .incremental_error.agg(
            mean_error="mean",
            mae=lambda x: x.abs().mean(),
            rmse=lambda x: np.sqrt((x * x).mean()),
            positive_years=lambda x: int((x > 0).sum()),
            n="size",
        )
        .reset_index()
    )
    reach_summary.to_csv(output / "reach_summary.csv", index=False)
    conditional = []
    for keys, group in pd.DataFrame(conditions).groupby(["landcover", "phase", "boundary"]):
        conditional.append(
            dict(zip(("landcover", "phase", "boundary"), keys, strict=True), **metrics(group))
        )
    pd.DataFrame(conditional).to_csv(output / "conditional_downstream.csv", index=False)
    closures = []
    for product in ("fine", "copernicus"):
        monthly = pd.read_csv(output / f"monthly_{product}.csv")
        recalculated = (
            monthly.p
            - monthly.aet
            - monthly.q
            - monthly.delta_soil
            - monthly.delta_snow
            - monthly.delta_reservoir
        )
        np.testing.assert_allclose(recalculated, 0, atol=1e-8)
        if (monthly[["soil", "snow", "reservoir", "aet", "q"]].to_numpy() < -1e-9).any():
            raise ValueError("Negative state/flux")
        closures.append(float(recalculated.abs().max()))
        annual = monthly.groupby(["fit", "year"])[
            ["p", "aet", "q", "delta_soil", "delta_snow", "delta_reservoir"]
        ].sum()
        annual.to_csv(output / f"annual_balance_{product}.csv")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    plot_data = valid
    control = "annual_local"
    if (output / "annual_reference_stations.csv").exists():
        corrected = pd.read_csv(output / "annual_reference_stations.csv")
        plot_data = pd.concat(
            [
                valid[valid.model != "annual_local"],
                corrected[corrected.year.between(2019, 2022) & corrected.eligible],
            ]
        )
        control = "annual_original_bounds"
    colors = {"existing_reach7": "#555555", control: "#287bb5", "monthly": "#dd741e"}
    labels = {
        "existing_reach7": "原七区模型",
        control: "上游单独年尺度率定",
        "monthly": "月尺度试点",
    }
    for col, product in enumerate(("fine", "copernicus")):
        subset = plot_data[(plot_data.landcover == product) & (plot_data.phase == "fixed_early")]
        guide = subset[subset.station == "贵得"]
        observed = guide[guide.model == "existing_reach7"].sort_values("year")
        axes[0, col].plot(observed.year, observed.observed, "ko-", label="实测")
        for model, color in colors.items():
            part = guide[guide.model == model].sort_values("year")
            axes[0, col].plot(part.year, part.predicted, "o--", color=color, label=labels[model])
            points = [
                metrics(subset[(subset.model == model) & (subset.station == s)])[
                    "relative_rmse_pct"
                ]
                for s in STATIONS
            ]
            axes[1, col].plot(STATIONS, points, "o-", color=color, label=labels[model])
        axes[0, col].set_title(
            ("精细分类" if product == "fine" else "Copernicus") + "：2013—2017 率定"
        )
        axes[0, col].set_ylabel("贵得年径流量（亿 m³）")
        axes[0, col].set_xticks(range(2019, 2023))
        axes[1, col].set_ylabel("2019—2022 逐站相对 RMSE（%）")
        for ax in axes[:, col]:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "pilot_validation.png", dpi=180)
    fig.savefig(output / "pilot_validation.pdf")
    plt.close(fig)
    summary = dict(
        gates=gates,
        accepted=all(g["passed"] for g in gates),
        max_monthly_closure_mm=max(closures),
        sources_checked=len(run["sources"]),
        downstream_increments_unchanged=True,
        production_changed=False,
        elapsed_seconds=run["elapsed_seconds"],
        peak_process_tree_mib=run["peak_process_tree_mib"],
        postprocessor=fingerprint(Path(__file__)),
    )
    write_json(output / "summary.json", summary)
    print(stats[(stats.scope == "贵得") | (stats.scope == "all7")].to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
