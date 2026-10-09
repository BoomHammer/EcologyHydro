"""Normalize calibration-year sector accounts and reconcile existing totals."""

from pathlib import Path

from ecologyhydro.aggregation import write_csv
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.simulation import write_json
from ecologyhydro.water_balance import ENDPOINTS, REGIONS, read_csv


def main():
    root = project_root()
    directory = root / "data/Hydrology/用取水耗水量"
    output = root / "project/m4/sector_water"
    output.mkdir(parents=True, exist_ok=True)
    records, checks, sources = [], [], [Path(__file__)]
    for year in range(2019, 2023):
        for source in ("地表水", "地下水"):
            for kind in ("用水量", "耗水量"):
                path = directory / f"{year}年黄河流域_{source}{kind}.csv"
                sources.append(path)
                rows = read_csv(path)
                old_path = (
                    root
                    / "data/Hydrology"
                    / f"{source}{'供水' if kind == '用水量' else '耗水'}2018-2023.csv"
                )
                sources.append(old_path)
                old = {r["水资源二级区"]: r for r in read_csv(old_path)}
                if {r["水资源二级区"] for r in rows} != set(REGIONS + ["黄河流域"]):
                    raise ValueError("Unexpected sector account regions")
                for row in rows:
                    region = row["水资源二级区"]
                    values = {
                        k.split("（")[0]: float(v) for k, v in row.items() if k != "水资源二级区"
                    }
                    if any(not 0 <= v < float("inf") for v in values.values()):
                        raise ValueError("Invalid sector water amount")
                    groups = {
                        "agriculture": values.get(
                            "农业", values.get("农田灌溉", 0) + values.get("林牧渔畜", 0)
                        ),
                        "industry": values["工业"],
                        "domestic_public": values.get(
                            "生活", values.get("城镇公共", 0) + values.get("居民生活", 0)
                        ),
                        "ecological": values["生态环境"],
                    }
                    record = {
                        "year": year,
                        "source": source,
                        "kind": kind,
                        "region": region,
                        "total": values["合计"],
                        **groups,
                    }
                    records.append(record)
                    checks.append(
                        {
                            "year": year,
                            "source": source,
                            "kind": kind,
                            "region": region,
                            "sector_sum_minus_total": sum(groups.values()) - values["合计"],
                            "new_minus_old_total": values["合计"] - float(old[region][str(year)])
                            if region in old
                            else None,
                        }
                    )
    lookup = {(r["year"], r["source"], r["kind"], r["region"]): r for r in records}
    basin_checks = []
    for year in range(2019, 2023):
        for source in ("地表水", "地下水"):
            for kind in ("用水量", "耗水量"):
                for sector in ("total", "agriculture", "industry", "domestic_public", "ecological"):
                    residual = (
                        sum(lookup[year, source, kind, r][sector] for r in REGIONS)
                        - lookup[year, source, kind, "黄河流域"][sector]
                    )
                    basin_checks.append(
                        {
                            "year": year,
                            "source": source,
                            "kind": kind,
                            "sector": sector,
                            "regional_sum_minus_basin": residual,
                        }
                    )
    supply_checks = []
    for r in records:
        if r["kind"] == "耗水量":
            supply = lookup[r["year"], r["source"], "用水量", r["region"]]
            for sector in ("total", "agriculture", "industry", "domestic_public", "ecological"):
                if r[sector] > supply[sector] + 0.01:
                    supply_checks.append(
                        {
                            "year": r["year"],
                            "source": r["source"],
                            "region": r["region"],
                            "sector": sector,
                            "consumption_minus_supply": r[sector] - supply[sector],
                        }
                    )
    accounts = {
        (int(r["year"]), r["station"]): r
        for r in read_csv(root / "project/m4/water_balance/station_accounts.csv")
    }
    sources.append(root / "project/m4/water_balance/station_accounts.csv")
    conflict_years = sorted(
        {r["year"] for r in checks if abs(r["new_minus_old_total"] or 0) > 0.011}
    )
    targets = []
    for year in range(2019, 2023):
        for station, n in ENDPOINTS.items():
            account = accounts[year, station]
            q, storage = float(account["observed_1e8_m3"]), float(account["大中型水库年蓄水变化量"])
            totals = {
                f"{src}_{sector}": sum(lookup[year, src, "耗水量", r][sector] for r in REGIONS[:n])
                for src in ("地表水", "地下水")
                for sector in ("agriculture", "industry", "domestic_public", "ecological", "total")
            }
            for agriculture in (0, 1):
                for ecological in (0, 1):
                    for groundwater in (0, 1):

                        def increment(src, values=totals, ag=agriculture, eco=ecological):
                            return (
                                values[f"{src}_industry"]
                                + values[f"{src}_domestic_public"]
                                + ag * values[f"{src}_agriculture"]
                                + eco * values[f"{src}_ecological"]
                            )

                        correction = (
                            increment("地表水") + groundwater * increment("地下水") + storage
                        )
                        targets.append(
                            {
                                "year": year,
                                "station": station,
                                **totals,
                                "agriculture_nonoverlap_assumed": agriculture,
                                "ecological_nonoverlap_assumed": ecological,
                                "groundwater_same_year_effect_assumed": groundwater,
                                "partial_target": q + correction,
                                "observed": q,
                                "reservoir_delta": storage,
                                "industrial_domestic_nonoverlap_assumed": 1,
                                "eligible_for_calibration": False,
                                "source_version_status": "unresolved_original_source_conflict"
                                if year in conflict_years
                                else "totals_reconciled",
                            }
                        )
    write_csv(output / "normalized.csv", records)
    write_csv(output / "row_checks.csv", checks)
    write_csv(output / "basin_checks.csv", basin_checks)
    write_csv(output / "partial_targets.csv", targets)
    write_json(
        output / "manifest.json",
        {
            "sources": [fingerprint(p) for p in sorted(set(sources))],
            "unit": "1e8_m3",
            "validation_year_read": False,
            "2019_harmonization": (
                "agriculture=irrigation+forestry_livestock_fishery; "
                "domestic=public+residential; semantic equivalence provisional"
            ),
            "max_sector_sum_residual": max(abs(r["sector_sum_minus_total"]) for r in checks),
            "max_old_total_difference": max(abs(r["new_minus_old_total"] or 0) for r in checks),
            "max_basin_sum_residual": max(abs(r["regional_sum_minus_basin"]) for r in basin_checks),
            "consumption_exceeds_supply": supply_checks,
            "source_conflict_years": conflict_years,
            "user_confirmation": {
                "date": "2026-10-09",
                "surface_water_2022": "original_sources_conflict_no_authoritative_version",
                "runoff_available": "existing_annual_only",
                "policy": "retain_both_sources_no_correction_no_deterministic_restoration_target",
                "annual_observations_and_climate_retained": True,
            },
            "notes": (
                "Industry/domestic full nonoverlap is also an assumption; "
                "ecological allocations require definition; groundwater storage remains unknown"
            ),
            "counts": {"normalized": len(records), "targets": len(targets)},
        },
    )


if __name__ == "__main__":
    main()
