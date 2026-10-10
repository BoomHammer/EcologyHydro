"""Verify a completed nine-pair run and export per-pair tables plus a readable report."""

import argparse

import numpy as np

from ecologyhydro.aggregation import write_csv
from ecologyhydro.baseline import read_json
from ecologyhydro.comparison import load_protocol, metrics, resolve, training_mask
from ecologyhydro.water_balance import read_csv


def report(config, run_id):
    root = resolve(config, config["paths"]["output_root"])
    output = (root / run_id).resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError("Select a run directory inside the configured output root")
    summary = read_json(output / "summary.json")
    if summary["status"] != "complete" or summary["pairs"] != 9:
        raise ValueError("Only completed nine-pair runs can be reported")
    frozen = read_json(output / "protocol.json")
    if frozen["years"] != config["years"] or frozen["products"] != config["products"]:
        raise ValueError("Run protocol differs from the selected YAML")
    predictions = read_csv(output / "predictions.csv")
    all_metrics = read_csv(output / "metrics.csv")
    products = list(config["products"])
    expected = {
        (a, b, str(y), s)
        for a in products
        for b in products
        for y in config["forcing_years"]
        for s in config["evaluation"]["managed_stations"]
    }
    actual = {(r["parameter_source"], r["landcover"], r["year"], r["station"]) for r in predictions}
    if actual != expected or len(predictions) != len(expected):
        raise ValueError("Missing or duplicated prediction cells")
    forbidden = config["years"]["warmup"] + config["years"]["bridge"]
    if any(r["scored"] != "False" for r in predictions if int(r["year"]) in forbidden):
        raise ValueError("Warmup/bridge observations entered scoring")
    training_years = np.array(config["forcing_years"])[training_mask(config)].tolist()
    for product in products:
        fit = read_json(output / f"parameters_{product}.json")["final"]
        if fit["training_years"] != training_years:
            raise ValueError("Final fit used the wrong training years")
    pair_dir = output / "pairs"
    pair_dir.mkdir(exist_ok=True)
    validations = []
    for source in products:
        for product in products:
            rows = [
                r
                for r in predictions
                if r["parameter_source"] == source and r["landcover"] == product
            ]
            selected = [r for r in rows if r["phase"] == "validation"]
            observed = np.array(
                [float(r["observed"]) if r["observed"] else np.nan for r in selected]
            )
            predicted = np.array(
                [float(r["predicted"]) if r["predicted"] else np.nan for r in selected]
            )
            result = metrics(predicted, observed, [r["scored"] == "True" for r in selected])
            stored = next(
                r
                for r in all_metrics
                if r["parameter_source"] == source
                and r["landcover"] == product
                and r["scope"] == "period"
                and r["label"] == "validation"
            )
            if int(stored["n"]) != result["n"] or not np.isclose(
                float(stored["rmse"]), result["rmse"]
            ):
                raise ValueError("Reported RMSE differs from independently recomputed predictions")
            validations.append({"parameter_source": source, "landcover": product, **result})
            write_csv(pair_dir / f"{source}_parameters__{product}_landcover.csv", rows)
    write_csv(output / "validation_matrix.csv", validations)
    lines = [
        "# 三产品九组合试跑结果",
        "",
        f"运行：`{output.name}`（UTC 时间戳）。",
        "",
        "两阶段合并率定：2013—2017、2019—2021；正式验证：2022—2024。",
        "2011—2012 预热、2018 桥接均不计分。2022、2023 为历史已见年份。",
        "",
        "## 验证期 RMSE（亿 m³）",
        "",
        "行是参数来源，列是使用的地类；每格使用相同的七站×三年。",
        "",
        "| 参数来源 | fine | copernicus | worldcover |",
        "| --- | ---: | ---: | ---: |",
    ]
    for source in products:
        row = [r for r in validations if r["parameter_source"] == source]
        lines.append("| " + source + " | " + " | ".join(f"{r['rmse']:.3f}" for r in row) + " |")
    lines.extend(
        [
            "",
            "## 验证期相对 RMSE（%）",
            "",
            "| 参数来源 | fine | copernicus | worldcover |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for source in products:
        row = [r for r in validations if r["parameter_source"] == source]
        lines.append(
            "| " + source + " | " + " | ".join(f"{r['relative_rmse_pct']:.3f}" for r in row) + " |"
        )
    lines.extend(
        [
            "",
            "## 输出和限制",
            "",
            "- [全部站年预测](predictions.csv)、[逐年及逐站跨年指标](metrics.csv)。",
            "- [十一站产水](awy_all_stations.csv)不是管理径流；四个未配水账站不作伪造评估。",
            "- `pairs/` 含九份独立组合 CSV，每份 14 年×7 站，缺账户预测留空。",
            "- 同一行横向差异才是在相同率定参数下更换产品；不能只看三个对角单元。",
            "- 三图静态年份／属性细分不同，结论仅适用于该固定映射及管理账户。",
            "- 单站单年 RMSE 为绝对误差；预热、桥接及缺水账单元不计分。",
            "- 三套 Kc 乘数的边界状态见各参数文件，不根据验证期调大上界。",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Verified", len(predictions), "station-year rows and exported", len(validations), "pairs")
    print(output / "report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--run", required=True, help="Completed run directory name, not a data path"
    )
    args = parser.parse_args()
    report(load_protocol(args.config), args.run)


if __name__ == "__main__":
    main()
