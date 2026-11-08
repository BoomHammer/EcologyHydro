"""Run monthly-CMFD annual hillslope estimates without fitting gauged sediment."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from osgeo import gdal

from ecologyhydro.annual_sediment import (
    multiply_vrt,
    nan_warp,
    native_factor,
    raster_grid,
    summarize_year,
)
from ecologyhydro.cache import fingerprint
from ecologyhydro.simulation import write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def report(output, annual, comparison, summary):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    matched = comparison[comparison.hillslope_delivery_1e8_t_yr.notna()]
    stations = matched.station.unique()
    nrows = max(1, (len(stations) + 1) // 2)
    figure, axes = plt.subplots(
        nrows, 2, figsize=(13, 3.25 * nrows), layout="constrained", squeeze=False
    )
    for axis, station in zip(axes.flat, stations, strict=False):
        data = matched[matched.station == station]
        axis.plot(data.year, data.observed_1e8_t_yr, "o-", label="测站实测", color="#1565a2")
        axis.plot(
            data.year,
            data.hillslope_delivery_1e8_t_yr,
            "s--",
            label="CMFD 月尺度近似 R：坡面入河量",
            color="#c16525",
        )
        axis.set(title=station, ylabel="亿吨/年", xticks=[2011, 2015, 2019, 2024], ylim=(0, None))
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(stations) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle(
        "2011—2024 年参考对照\n当前地类固定；未模拟库河冲淤和其他来沙；未率定", fontsize=14
    )
    figure.savefig(output / "annual_reference.png", dpi=160)
    plt.close(figure)
    lines = [
        "# CMFD 月降水驱动的逐年坡面入河泥沙",
        "",
        f"已完成 2011—2024 年 {annual.station.nunique()} 个控制流域的初步估算，"
        f"{len(stations)} 个站与实测匹配。单位：亿吨/年。",
        "**这些是坡面入河量，不是包含水库和河道过程的测站输沙预测。实测只作参考，未率定。**",
        "",
        "| 测站 | 2022 坡面入河 | 2022 实测 | 2023 坡面入河 | 2023 实测 |"
        " 2024 坡面入河 | 2024 实测 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for station in stations:
        data = matched[matched.station == station].set_index("year")
        cells = [station]
        for year in (2022, 2023, 2024):
            cells += [
                f"{data.loc[year, 'hillslope_delivery_1e8_t_yr']:.4f}",
                f"{data.loc[year, 'observed_1e8_t_yr']:.4f}",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "![逐年参考对照](annual_reference.png)",
        "",
        "## 文件",
        "",
        f"- [完整年度结果](station_annual_sediment.csv)：{len(annual)} 行，"
        "入河量、USLE、R 和覆盖率。",
        f"- [实测站年度对照](observations_comparison.csv)：{len(comparison)} 行。",
        "- [站间增量区结果](zone_annual_sediment.csv)：避免将嵌套站累计量重复相加。",
        "- `rasters/R_2011.vrt` 至 `R_2024.vrt`：250 m 近似年度 R；",
        "  `rasters/sed_export_2011.vrt` 至 `sed_export_2024.vrt`：t/ha/年。可用 GDAL/QGIS 打开。",
        "  VRT 按需读取本项目源栅格，不是可单独搬走的 TIFF；需保留基准及其依赖路径。",
        "- [复现与检查记录](summary.json)；[来源指纹](manifest.json)。",
        "",
        "## 方法及限制",
        "",
        "每月降水由 CMFD prec 通量乘当月秒数得到；MFI=sum(P_month²)/P_year。",
        "R_y=R_1991_2020×MFI_y/mean(MFI_1991_2020)。异常在原生网格求取后双线性重采样，",
        "固定共同有效掩膜。该关系是项目近似假设，不能恢复月内暴雨，未验证其逐年 R 精度。",
        "当前地类、K、C/P、地形及 SDR 连通性固定，逐像元 export_y=export_base×异常。",
        "K 未再次乘 0.1317；没有重跑 14 次地形路由。此加速不适用于地类、C 或河网改变。",
        "当前地类不是历史逐年地类；历史治理、库坝调度及河道冲淤未被重建。",
        "CMFD 与当前水量发布版本所用 TerraClimate 不同，本次没有改动水量结果。",
        "",
        "原生气候缺测未落入实际报告域；各年坡面有效像元数与基准一致。",
        f"30 年平均年际因子在目标网格最大偏差：{summary['max_factor_baseline_error']:.3g}；",
        f"30 年平均因子驱动的站点输出相对基准最大偏差："
        f"{summary['max_baseline_relative_error']:.3g}。",
        "平均因子独立重投影并汇总，利用固定 SDR 对 R 的线性关系核查长期基准。",
        "这些是计算一致性检查，不是观测验证。原 SDR 非河道缺失像元仍保留，未填零。",
        "",
        "观测减坡面入河量仅为未解释的综合差额，混合其他来沙、冲淤、引沙和模型误差；",
        "不能当作水库淤积观测或用该差额回填后宣称预测准确。未计算精度排名。",
        "测站与流域按名称完整性检查；不借用邻站或按控制面积比例代算。",
        "",
        "## 现在做不到的部分及替代方案",
        "",
        "现有资料不能唯一约束包含库河过程的年度测站输沙量。水库材料尚未进入项目。",
        "当前替代交付是逐年坡面来沙及参考对照。后续可用已知引沙、累计冲淤按其真实时段",
        "约束质量守恒，周排沙不年化；仍缺失的过程应报告假设情景。若改用经验站点模型，",
        "需要调整实测‘仅参考’的用途，预先固定训练与检验划分；本次没有进行这种拟合。",
        "",
        "## 参考资料",
        "",
        "- [InVEST SDR 公式及过程边界](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/sdr.html)",
        "- [Yin et al. 2015，降水时间分辨率与 R 估算](https://doi.org/10.5194/hess-19-4113-2015)",
        "- [月尺度 MFI 在黄土高原的应用](https://doi.org/10.1155/2013/256352)；不是本项目归一化公式的验证。",
        "- [GDAL VRT](https://gdal.org/en/stable/drivers/raster/vrt.html)：NaN 传播兼容 3.10。",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdr-run", type=Path, required=True)
    parser.add_argument(
        "--proxy",
        type=Path,
        default=Path("project/sediment/monthly_climate_audit/prec/annual_monthly_proxies.npz"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--routing-dir", type=Path)
    parser.add_argument("--baseline-table", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=2)
    args = parser.parse_args()
    if args.routing_dir is None:
        settings = yaml.safe_load(Path("config/sediment.yaml").read_text(encoding="utf-8"))
        args.routing_dir = Path(settings["zone_station"]).parent
    gdal.UseExceptions()
    gdal.SetCacheMax(256 * 1024**2)
    started = time.perf_counter()
    output = args.output or Path("experiments/sediment_annual") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    directory = output / "rasters"
    directory.mkdir()
    write_json(output / "summary.json", {"status": "running"})
    workspace = args.sdr_run / "invest_sdr"
    baseline_path = args.baseline_table or args.sdr_run / "station_sediment.csv"
    baseline = pd.read_csv(baseline_path).set_index("ws_id")
    membership_path = args.routing_dir / "zone_station.csv"
    membership = pd.read_csv(membership_path)
    observation_path = Path("data/Hydrology/输沙量2011-2024.csv")
    observed = pd.read_csv(observation_path, encoding="utf-8-sig").rename(
        columns={"测站": "station"}
    )
    missing = sorted(set(observed.station) - set(membership.station))
    if missing:
        raise ValueError(
            f"Stations with supplied coordinates are absent from routing: {missing}. "
            "Run complete_sediment_stations.py, then pass --routing-dir and --baseline-table."
        )
    names = membership.drop_duplicates("station_id").set_index("station_id").station
    if not names.sort_index().equals(baseline.station.sort_index()):
        raise ValueError("Baseline station IDs/names do not match selected routing")
    zones_path = membership_path.with_name("zones.tif")
    sources = {
        "export": workspace / "sed_export.tif",
        "usle": workspace / "usle.tif",
        "r": workspace / "intermediate_outputs/masked_erosivity.tif",
    }
    with gdal.Open(str(sources["export"])) as template:
        grid = raster_grid(template)
        for name, path in sources.items():
            nan_warp(path, directory / f"base_{name}_nan.vrt", grid)
        with gdal.Warp(
            str(directory / "zones.vrt"),
            str(zones_path),
            format="VRT",
            **grid,
            resampleAlg="near",
            dstNodata=0,
        ):
            pass
        with np.load(args.proxy) as proxy:
            years = proxy["years"].astype(int)
            if not np.array_equal(years, np.arange(1991, 2025)):
                raise ValueError("Expected complete 1991-2024 climate proxy")
            factors = proxy["mfi_anomaly"]
            common = np.isfinite(factors).all(axis=0)
            if np.any(np.isfinite(factors) != common):
                raise ValueError("Climate masks must be fixed across years")
            means = factors[years <= 2020].astype(np.float64).mean(axis=0)
            # Add a mean-factor map to independently check normalization after reprojection.
            for year, factor in [*zip(years, factors, strict=True), ("mean", means)]:
                native = directory / f"native_factor_{year}.tif"
                native_factor(native, proxy["longitude"], proxy["latitude"], factor)
                target = directory / f"factor_{year}.vrt"
                nan_warp(native, target, grid, "bilinear")
                if isinstance(year, str) or year < 2011:
                    continue
                for name, prefix, unit in (
                    ("r", "R", "MJ mm/(ha h year)"),
                    ("export", "sed_export", "t/(ha year)"),
                ):
                    multiply_vrt(
                        [directory / f"base_{name}_nan.vrt", target],
                        directory / f"{prefix}_{year}.vrt",
                        template,
                        unit,
                    )
    print(f"Virtual rasters ready: {output}", flush=True)
    factor_error = 0.0
    with (
        gdal.Open(str(directory / "factor_mean.vrt")) as mean,
        gdal.Open(str(directory / "zones.vrt")) as zones,
    ):
        for y in range(0, mean.RasterYSize, 128):
            window = (0, y, mean.RasterXSize, min(128, mean.RasterYSize - y))
            valid = zones.ReadAsArray(*window) > 0
            values = mean.ReadAsArray(*window)[valid]
            if not np.isfinite(values).all():
                raise ValueError("Incomplete climate coverage in watershed")
            factor_error = max(factor_error, float(np.max(np.abs(values - 1), initial=0)))
    if factor_error > 1e-6:
        raise ValueError("Reprojected baseline factor is not one")
    annual_rows, zone_rows, mean_station = [], [], {}
    calculation_years = ["mean", *years[years >= 2011]]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(summarize_year, calculation_years, [directory] * len(calculation_years))
        for year, totals, counts in results:
            for station_id, members in membership.groupby("station_id", sort=True):
                ids = members.zone_id.to_numpy()
                base = baseline.loc[station_id]
                valid_count = int(counts[0, ids].sum())
                expected = int(round(base.area_km2 * 1e6 / 250**2)) - int(
                    base.missing_output_pixels
                )
                if valid_count != expected:
                    raise ValueError(f"Changed output coverage: {year}, {station_id}")
                total, erosion, r_sum = totals[:, ids].sum(axis=1)
                if year == "mean":
                    mean_station[station_id] = total
                else:
                    annual_rows.append(
                        {
                            "year": year,
                            "station_id": station_id,
                            "station": base.station,
                            "hillslope_delivery_t_yr": total,
                            "hillslope_delivery_1e8_t_yr": total / 1e8,
                            "usle_t_yr": erosion,
                            "mean_r_MJ_mm_ha_h_yr": r_sum / counts[1, ids].sum(),
                            "valid_hillslope_pixels": valid_count,
                            "unresolved_nonstream_pixels": base.unresolved_nonstream_pixels,
                            "hillslope_coverage_fraction": base.hillslope_coverage_fraction,
                            "status": "monthly_proxy_fixed_landcover_hillslope_only",
                        }
                    )
            if year != "mean":
                for zone in sorted(membership.zone_id.unique()):
                    zone_rows.append(
                        {
                            "year": year,
                            "zone_id": zone,
                            "hillslope_delivery_t_yr": totals[0, zone],
                            "usle_t_yr": totals[1, zone],
                        }
                    )
            print(f"Aggregated {year}", flush=True)
    mean_totals = pd.Series(mean_station)
    relative = abs(mean_totals / baseline.independent_raster_sum_t_yr - 1)
    if relative.max() > 1e-6:
        raise ValueError("Thirty-year annual mean does not reproduce baseline")
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(output / "station_annual_sediment.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(zone_rows).to_csv(output / "zone_annual_sediment.csv", index=False)
    comparison = observed.melt(
        id_vars="station",
        value_vars=[str(year) for year in range(2011, 2025)],
        var_name="year",
        value_name="observed_1e8_t_yr",
    )
    comparison.year = pd.to_numeric(comparison.year).astype(int)
    comparison.observed_1e8_t_yr = pd.to_numeric(comparison.observed_1e8_t_yr)
    if (comparison.observed_1e8_t_yr.dropna() < 0).any():
        raise ValueError("Negative observations")
    comparison = comparison.merge(
        annual[["year", "station", "hillslope_delivery_1e8_t_yr"]],
        on=["year", "station"],
        how="left",
        validate="one_to_one",
    ).sort_values(["station", "year"])
    comparison["observed_minus_hillslope_1e8_t_yr"] = (
        comparison.observed_1e8_t_yr - comparison.hillslope_delivery_1e8_t_yr
    )
    comparison["status"] = np.where(
        comparison.hillslope_delivery_1e8_t_yr.notna(),
        "reference_only_unexplained_difference",
        "no_station_catchment",
    )
    comparison.to_csv(output / "observations_comparison.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "formula": "R_y = R_1991_2020 * MFI_y / mean(MFI_1991_2020)",
        "method_status": "project_hypothesis_not_validated_annual_erosivity",
        "observations_use": "reference_only_no_calibration",
        "climate_variable": "CMFD_prec",
        "resampling": "bilinear_common_mask",
        "workers": args.workers,
        "routing_dir": str(args.routing_dir.resolve()),
        "baseline_table": str(baseline_path.resolve()),
        "gdal_version": gdal.VersionInfo(),
        "sources": [
            fingerprint(p)
            for p in [
                args.proxy,
                args.proxy.with_name("audit.json"),
                args.sdr_run / "manifest.json",
                baseline_path,
                *sources.values(),
                zones_path,
                membership_path,
                observation_path,
                Path(__file__),
                Path("src/ecologyhydro/annual_sediment.py"),
            ]
        ],
    }
    write_json(output / "manifest.json", manifest)
    summary = {
        "status": "complete",
        "station_years": len(annual),
        "observation_station_years": len(comparison),
        "matched_stations": int(
            comparison.dropna(subset=["hillslope_delivery_1e8_t_yr"]).station.nunique()
        ),
        "max_factor_baseline_error": factor_error,
        "max_baseline_relative_error": float(relative.max()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report(output, annual, comparison, summary)
    summary["output_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    write_json(output / "summary.json", summary)
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
