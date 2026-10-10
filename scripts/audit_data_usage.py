"""Report existing input usage, PET provenance, and per-station research metrics."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

ROOT = Path(__file__).resolve().parents[1]


def classify(path):
    """Classify reviewed code paths; mere fingerprint inclusion is not model use."""
    name = path.name
    if path.suffix in {".xml", ".sbn", ".sbx", ".pdf"}:
        return (
            "support",
            "元数据、空间索引或参考文档；不作为独立水文输入",
            "docs/M2_PREPROCESSING.md",
        )
    if "TerraClimate_PET" in path.parts:
        return (
            "historical",
            "2019—2023旧PET用于原基线；当前改用提供方V1.1",
            "project/repairs/soil_routing/m2_index.json",
        )
    if "Data_forcing_01mo_010deg" in path.parts:
        if name.startswith("rhum_"):
            return (
                "prepared_only",
                "已裁剪缓存；PM计算用shum，未发现rhum进入水文公式",
                "src/ecologyhydro/climate_review.py",
            )
        return (
            "historical",
            "prec/bcpr用于历史降水试验；其余变量用于PM蒸散对照",
            "project/m4/et0_trials/summary.json",
        )
    if "用取水耗水量" in path.parts:
        if name.startswith("2023"):
            return (
                "no_use_evidence",
                "未发现代码读取或既有数值产物引用",
                "scripts/audit_sector_water.py:2019-2022",
            )
        if name == "2022年黄河流域_地表水耗水量.csv":
            return (
                "current_alternative",
                "当前sector_2022管理账户对照",
                "src/ecologyhydro/long_record.py",
            )
        return (
            "diagnostic",
            "行业合计核查；耗水用于部分还原情景；未独立加入主公式",
            "project/m4/sector_water/normalized.csv",
        )
    if name.endswith("2013-2023.csv"):
        if name.startswith(("实测年径流量", "地表水耗水", "大中型水库")):
            return "current", "现行率定观测或管理扣减", "src/ecologyhydro/long_record.py"
        if name.startswith("降水量"):
            return (
                "diagnostic",
                "降水输入交叉核查及敏感性试验",
                "scripts/check_headwater_precipitation.py",
            )
        return (
            "diagnostic",
            "新旧表一致性与缺失核查；未进入现行径流公式",
            "project/diagnostics/long_record_v2/normalized_fields.csv",
        )
    if name.endswith("2018-2023.csv"):
        if name.startswith(("实测年径流量", "地表水耗水", "大中型水库")):
            return (
                "historical",
                "历史率定/管理扣减；新长表已替代",
                "src/ecologyhydro/water_balance.py",
            )
        return (
            "diagnostic",
            "历史水量账户或降水核查；长表已替代",
            "project/m4/water_balance/manifest.json",
        )
    if name == "水资源二级区面积.csv":
        return "diagnostic", "统计区面积/降水深度及汇水区交叉核查", "scripts/audit_long_record.py"
    if path.suffix in {".shp", ".shx", ".dbf", ".prj", ".cpg"}:
        return (
            "current_preprocessing",
            "矢量数据集组成文件：汇水区、河网或湖泊预处理",
            "project/repairs/soil_routing/m2_index.json",
        )
    if any(folder in path.parts for folder in ("DEM", "Soil", "LCLU")):
        return (
            "current_preprocessing",
            "地形/土壤/地类及属性表，派生输入被当前流程复用",
            "project/repairs/soil_routing/m2_index.json",
        )
    if name == "测站控制面积.csv":
        return (
            "current_preprocessing",
            "站点定位、汇水区及面积核查",
            "project/repairs/soil_routing/m2_index.json",
        )
    return "unresolved", "需要逐项核查", ""


def pet_audit():
    folder = ROOT / "project/repairs/long_climate_v1"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    result = []
    for year in range(2013, 2019):
        path = folder / f"terraclimate_pet_{year}.nc"
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        with Dataset(path) as ds:
            t = ds["time"]
            dates = num2date(t[:], t.units, calendar=getattr(t, "calendar", "standard"))
            pairs = [(d.year, d.month) for d in dates]
            values = np.ma.asarray(ds["pet"][:])
            version, units = ds.getncattr("version"), ds["pet"].units
        record = next(r for r in summary["records"] if r["variable"] == "pet" and r["year"] == year)
        aligned = Path(record["aligned"]["path"])
        raw_ok = hashlib.sha256(path.read_bytes()).hexdigest() == meta["file"]["sha256"]
        with aligned.open("rb") as stream:
            aligned_ok = (
                hashlib.file_digest(stream, "sha256").hexdigest() == record["aligned"]["sha256"]
            )
        if pairs != [(year, m) for m in range(1, 13)] or not raw_ok or not aligned_ok:
            raise ValueError(f"PET provenance failure: {year}")
        result.append(
            dict(
                year=year,
                months=len(pairs),
                version=version,
                units=units,
                raw_sha256_ok=raw_ok,
                aligned_sha256_ok=aligned_ok,
                valid_min=float(values.min()),
                valid_max=float(values.max()),
                raw=str(path.relative_to(ROOT)),
                aligned=str(aligned.relative_to(ROOT)),
                url=meta["url"],
            )
        )
    return pd.DataFrame(result)


def metrics():
    data = pd.read_csv(ROOT / "project/calibration/monthly_pilot_v1/all_station_results.csv")
    data = data[data.year.between(2019, 2022) & data.model.isin(["existing_reach7", "monthly"])]
    rows = []
    for key, group in data.groupby(["landcover", "model", "phase", "station"]):
        obs, pred = group.observed.to_numpy(), group.predicted.to_numpy()
        if len(obs) != 4 or not np.isfinite(obs + pred).all():
            raise ValueError("Expected four complete station-years")
        error = pred - obs
        nse = 1 - np.sum(error**2) / np.sum((obs - obs.mean()) ** 2)
        r2 = np.corrcoef(obs, pred)[0, 1] ** 2
        pbias = 100 * (obs - pred).sum() / obs.sum()
        rows.append(
            dict(zip(["landcover", "model", "phase", "station"], key, strict=True))
            | dict(
                n=4,
                NSE=nse,
                R2=r2,
                PBIAS_observed_minus_predicted_pct=pbias,
                relative_RMSE_pct=100 * np.sqrt(np.mean((error / obs) ** 2)),
                MAPE_pct=100 * np.mean(np.abs(error / obs)),
                meets_2015_reference=bool(nse > 0.5 and r2 > 0.6 and abs(pbias) <= 15),
            )
        )
    return pd.DataFrame(rows)


def main():
    output = ROOT / "project/diagnostics/data_usage_review"
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in sorted((ROOT / "data").rglob("*")):
        if path.is_file():
            status, reason, evidence = classify(path)
            rows.append(
                dict(
                    path=path.relative_to(ROOT).as_posix(),
                    bytes=path.stat().st_size,
                    status=status,
                    reason=reason,
                    evidence=evidence,
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "data_inventory.csv", index=False, encoding="utf-8-sig")
    pet_audit().to_csv(output / "pet_2013_2018.csv", index=False)
    scores = metrics()
    scores.to_csv(output / "station_research_metrics.csv", index=False, encoding="utf-8-sig")
    print(frame.groupby("status").agg(files=("path", "count"), bytes=("bytes", "sum")).to_string())
    print(
        scores.groupby(["landcover", "model", "phase"])
        .agg(
            passed=("meets_2015_reference", "sum"),
            NSE_min=("NSE", "min"),
            NSE_max=("NSE", "max"),
            max_abs_pbias=("PBIAS_observed_minus_predicted_pct", lambda x: x.abs().max()),
        )
        .to_string()
    )
    print("PET: all six years have 12 months and matching raw/aligned fingerprints")


if __name__ == "__main__":
    main()
