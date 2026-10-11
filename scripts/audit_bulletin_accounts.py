"""Reconcile visually transcribed bulletin tables with current surface accounts."""

from pathlib import Path

import numpy as np
import pandas as pd

from ecologyhydro.cache import fingerprint
from ecologyhydro.long_record import number, table
from ecologyhydro.simulation import write_json
from ecologyhydro.water_balance import REGIONS

# Local consumption and export, in 1e8 m3. Each pair is transcribed from the
# consumption row, not the supply row. Inner-flow region is included in audit
# totals but excluded from the seven managed river reaches.
TRANSCRIPTION = {
    2022: (
        28,
        20,
        11,
        [1.16, 20.10, 104.91, 7.74, 49.10, 11.54, 20.56, 0.83],
        [0, 0.69, 3.74, 3.71, 0.06, 1.44, 73.19, 0],
    ),
    2023: (
        29,
        21,
        11,
        [1.10, 17.13, 95.09, 8.60, 46.03, 12.83, 15.45, 0.77],
        [0, 0.28, 4.69, 3.51, 0.06, 4.26, 70.75, 0],
    ),
    2024: (
        30,
        24,
        10,
        [1.26, 20.17, 99.23, 11.17, 46.10, 14.77, 17.00, 0.03],
        [0, 0.38, 3.88, 3.17, 0.04, 6.79, 83.10, 0],
    ),
}
BASIN_TOTALS = {
    2022: (215.94, 82.83, 298.77),
    2023: (197, 83.55, 280.55),
    2024: (209.73, 97.36, 307.09),
}


def main():
    output = Path("project/references/water_bulletins")
    output.mkdir(parents=True, exist_ok=True)
    original = Path("data/Hydrology/地表水耗水2011-2024.csv")
    accounts = table(original, "水资源二级区")
    records, sources = [], [original, Path(__file__)]
    for year, (pdf_page, printed_page, table_id, local, exported) in TRANSCRIPTION.items():
        pdf = Path(f"data/Hydrology/{year}年黄河水资源公报.pdf")
        sources.append(pdf)
        for region, inside, outside in zip(REGIONS, local, exported, strict=True):
            total = number(accounts[region][str(year)])
            if abs(total - inside - outside) > 0.011:
                raise ValueError(f"Bulletin/current CSV mismatch: {year} {region}")
            records.append(
                dict(
                    year=year,
                    region=region,
                    local_consumption=inside,
                    export=outside,
                    total_consumption=total,
                    residual=total - inside - outside,
                    pdf=str(pdf),
                    pdf_page=pdf_page,
                    printed_page=printed_page,
                    table=table_id,
                    method="visual_transcription_from_scanned_pdf",
                )
            )
        np.testing.assert_allclose(
            [sum(local), sum(exported), sum(local) + sum(exported)],
            BASIN_TOTALS[year],
            atol=0.011,
            rtol=0,
        )
    pd.DataFrame(records).to_csv(output / "surface_account_split.csv", index=False)
    write_json(
        output / "audit.json",
        dict(
            sources=[fingerprint(p) for p in sources],
            units="1e8 m3",
            rows=len(records),
            max_absolute_residual=max(abs(r["residual"]) for r in records),
            earlier_years="unknown split; do not fill export or agricultural share with zero",
            interpretation="Split preserves total deductions; export cannot irrigate local pixels",
        ),
    )
    print(
        pd.DataFrame(records)
        .groupby("year")[["local_consumption", "export", "total_consumption"]]
        .sum()
        .to_string()
    )


if __name__ == "__main__":
    main()
