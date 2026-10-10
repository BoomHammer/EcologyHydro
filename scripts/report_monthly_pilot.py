"""Publish the three-stage pilot results, including unsuccessful comparisons."""

import numpy as np
import pandas as pd
from summarize_monthly_pilot import metrics

from ecologyhydro.baseline import read_json
from ecologyhydro.config import project_root
from ecologyhydro.long_record import STATIONS
from ecologyhydro.simulation import write_json


def table(headers, rows):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "—" if not np.isfinite(value) else f"{value:.2f}"
        return str(value)

    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(cell, row)) + " |" for row in rows],
        ]
    )


def main():
    root = project_root()
    output = root / "project/calibration/monthly_pilot_v1"
    summary = read_json(output / "summary.json")
    audit = read_json(output / "annual_reference_audit.json")
    frame = pd.read_csv(output / "all_station_results.csv")
    stats = pd.read_csv(output / "validation_metrics.csv")
    lookup = stats.set_index(["landcover", "phase", "model", "scope"])
    corrected = pd.read_csv(output / "annual_reference_stations.csv")
    # Validate the supplemental control's routing independently of its fit script.
    keys = ["landcover", "phase", "year", "station"]
    base = frame[frame.model == "existing_reach7"].set_index(keys)
    control = corrected.set_index(keys)
    original = base.loc[control.index]
    usable = np.isfinite(original.predicted)
    np.testing.assert_allclose(
        (control.predicted - original.predicted)[usable], control.guide_change[usable], atol=1e-8
    )
    matched = pd.read_csv(output / "annual_reference_metrics.csv").set_index(
        ["landcover", "phase", "scope"]
    )
    gate_rows = []
    for gate in summary["gates"]:
        product = gate["landcover"]
        tests = {}
        for phase in ("fixed_early", "rolling"):
            candidate = lookup.loc[(product, phase, "monthly", "贵得")]
            reference = matched.loc[(product, phase, "贵得")]
            tests[f"{phase}_guide_rmse"] = bool(
                candidate.relative_rmse_pct <= 0.9 * reference.relative_rmse_pct
            )
            tests[f"{phase}_guide_worst"] = bool(candidate.worst_pct <= reference.worst_pct + 2)
            tests[f"{phase}_all7"] = bool(
                lookup.loc[(product, phase, "monthly", "all7"), "relative_rmse_pct"]
                <= 1.02 * matched.loc[(product, phase, "all7"), "relative_rmse_pct"]
            )
        tests["deletion"] = (
            gate["mean_delete_year_spread_pct"]["monthly"]
            <= 1.1 * audit["mean_spread_pct"][product]
        )
        tests["original_baseline_and_initial"] = all(
            v
            for k, v in gate["tests"].items()
            if "annual_local" not in k and k != "delete_year_stability"
        )
        gate_rows.append(dict(landcover=product, checks=tests, accepted=all(tests.values())))
    write_json(
        output / "final_assessment.json",
        dict(
            original_preregistered_acceptance=summary["accepted"],
            supplemental_matched_annual_audit=gate_rows,
            accepted=all(g["accepted"] for g in gate_rows),
            production_changed=False,
            stages=dict(
                monthly_pilot="completed",
                temporal_stability="completed_failed_acceptance",
                downstream="completed_diagnosis_no_new_downstream_parameters",
            ),
        ),
    )
    rows = []
    for station in STATIONS:
        rows.append(
            [
                station,
                *[
                    lookup.loc[(p, "fixed_early", m, station), "relative_rmse_pct"]
                    for p in ("fine", "copernicus")
                    for m in ("existing_reach7", "monthly")
                ],
            ]
        )
    station_table = table(
        ["测站", "精细：原模型", "精细：月尺度", "Copernicus：原模型", "Copernicus：月尺度"], rows
    )
    headwater_rows = []
    for product in ("fine", "copernicus"):
        for phase in ("fixed_early", "rolling"):
            headwater_rows.append(
                [
                    product,
                    "固定早期率定" if phase == "fixed_early" else "逐年向前率定",
                    lookup.loc[(product, phase, "existing_reach7", "贵得"), "relative_rmse_pct"],
                    matched.loc[(product, phase, "贵得"), "relative_rmse_pct"],
                    lookup.loc[(product, phase, "monthly", "贵得"), "relative_rmse_pct"],
                ]
            )
    stability_rows = [
        [
            g["landcover"],
            audit["mean_spread_pct"][g["landcover"]],
            g["mean_delete_year_spread_pct"]["monthly"],
            "不通过",
        ]
        for g in summary["gates"]
    ]
    comparison_tables, supplementary = {}, []
    for year, phase in ((2018, "fixed_early"), (2023, "reused_2023")):
        subset = frame[(frame.year == year) & (frame.phase == phase)]
        indexed = subset.set_index(["landcover", "model", "station"])
        results = []
        for station in STATIONS:
            record = dict(
                station=station, observed=indexed.loc[("fine", "monthly", station), "observed"]
            )
            for product in ("fine", "copernicus"):
                for model in ("existing_reach7", "monthly"):
                    row = indexed.loc[(product, model, station)]
                    record[f"{product}_{model}_predicted"] = row.predicted
                    record[f"{product}_{model}_relative_error_pct"] = row.relative_error_pct
            results.append(record)
        pd.DataFrame(results).to_csv(output / f"station_comparison_{year}.csv", index=False)
        comparison_tables[year] = table(
            [
                "测站",
                "实测",
                "精细月尺度模拟",
                "精细偏差 %",
                "Copernicus 月尺度模拟",
                "Copernicus 偏差 %",
            ],
            [
                [
                    r["station"],
                    r["observed"],
                    r["fine_monthly_predicted"],
                    r["fine_monthly_relative_error_pct"],
                    r["copernicus_monthly_predicted"],
                    r["copernicus_monthly_relative_error_pct"],
                ]
                for r in results
            ],
        )
        for (product, model), group in subset.groupby(["landcover", "model"]):
            available = group[np.isfinite(group.observed)]
            supplementary.append(
                dict(year=year, landcover=product, model=model, **metrics(available))
            )
    pd.DataFrame(supplementary).to_csv(output / "supplementary_year_metrics.csv", index=False)
    reaches = pd.read_csv(output / "reach_summary.csv").set_index(["landcover", "phase", "reach"])
    reach_rows = []
    for i in range(1, 7):
        reach = f"{STATIONS[i - 1]}—{STATIONS[i]}"
        fine = reaches.loc[("fine", "fixed_early", reach)]
        cop = reaches.loc[("copernicus", "fixed_early", reach)]
        reach_rows.append([reach, fine.mean_error, fine.mae, cop.mean_error, cop.mae])
    text = f"""# 上游月尺度试点、跨年稳定性与下游区间：三项最终结果

日期：2026-10-10。**三项均已完成；月尺度候选未通过稳定性要求，保留现有七区工作参考，没有更新生产参数。**

## 1. 完成的试点与边界

只替换贵得以上自然产水，其余六区的产水、耗水、水库蓄变和已有净损失参数保持各自原率定结果。
模型连续运行 2012—2023 月气象；2012 只作状态预热，不使用或补造当年水文观测。
2018 连续推进状态但不进入率定，粗略径流仍使用过去完整管理账户的均值。
先以 2013—2017 率定并检验 2019—2022，再以每次预测年前可用资料逐年率定检验。
最后使用 2013—2022 可用资料率定并列出 2023 复用年结果；2023 不参与候选选择。

自编月模型包括固定雨雪分配／融雪、有限土壤水桶、受水分限制蒸散和线性径流储水。
自由参数共三个：土壤容量倍率 0.25—4、植被 Kc 倍率 0.4—1.6、径流滞蓄时间 0.1—12 月。
非植被保持供水限制的 Kc×PET 蒸散；植被有效 Kc 上限仍为 1.3。
两套地类使用同样方程、范围和优化预算；精细 19、Copernicus 26 个地类×土壤容量单元。
气候为贵得以上面积加权月平均，土壤参数为原确定性样点按面积加权分组；不是全像元月水文模型。
温度在 -1 至 +1 ℃线性分雨雪，正温融雪系数固定 3 mm/℃/日，是未率定的试点假设。
2012 年初土壤半满、雪和径流库为零；检验土壤初值 0／满容量的影响。

这不是官方 InVEST AWY 或 IHACRES 的实现。月模型仍只有年径流率定约束；
没有证明其月蒸散、融雪或地下水过程真实。
年尺度对照保留空间分布，而月试点使用流域平均气候，因此差异也包含空间简化，不能全归因于时间步长。

## 2. 2019—2022 逐站结果

下表为 2013—2017 固定率定后的**逐站相对 RMSE（%）**，每站四年，越小越好。

{station_table}

精细七站总体相对 RMSE 从 13.70% 降至 9.12%；Copernicus 从 14.65% 降至 13.29%。
但 Copernicus 贵得由 11.38% 变为 15.54%，头道拐也恶化。总分下降不能证明上游过程修正成功。
月试点只是改变所有下游站共同接收的贵得来水；本轮的下游局地误差没有被修正。

## 3. 年尺度对照与跨年稳定性

为排除单独率定上游就能改善的解释，另做只用贵得资料的两个参数年尺度对照。
初始对照采用与月模型相同的 Kc 范围 0.4—1.6，Z 范围 0.1—30；两个参数均卡上限，删年波动接近零。
**这种零波动是边界锁死，不代表参数识别良好。**
因而补充恢复原模型 Z=1—30、Kc 倍率=0.7—2 的年尺度对照。
此补充发生在初始评分后，单独保留审计协议；月模型、验收门槛和主检验结果均未改变，不把补充对照称为预先注册验证。

下表仅针对贵得，指标为相对 RMSE（%）：

{table(["地类", "检验方式", "原七区", "单独年尺度（原范围）", "月尺度"], headwater_rows)}

删去 2013—2017 中任意一年径流目标，重新率定，比较 2019—2022 预测范围。
指标为各检验年 `(五次预测最大值−最小值)/实测` 的四年平均（%），不是置信区间或误差指标。

{table(["地类", "年尺度对照波动", "月尺度波动", "判定"], stability_rows)}

精细月模型虽降低部分平均误差，但对率定年份更敏感；Copernicus 同时出现上游误差变大。
两套月模型的容量倍率和 Kc 倍率在所有主要率定窗口均达到上限，参数补偿与结构不足的可能性仍大。
初始土壤水状态影响通过检查，但不等于融雪假设、气候输入或初始地下储量已经验证。
事先门槛要求两套地类在固定／逐年率定下贵得 RMSE 同时改善至少 10%，
最差误差不明显变差，七站总体不恶化，且删年波动受控。
原协议及补充年尺度对照审计均不支持采用，不能放宽门槛来宣布成功。

## 4. 2018 粗略检查与 2023 复用年结果

水量单位亿 m³，偏差为 `(模拟−实测)/实测`；正为高估。2018 的管理水量是历史账户情景，不是观测。

{comparison_tables[2018]}

2018 四站粗略结果明显改善，但没有纳入正式稳定性评分，贵得本身没有当年实测。

{comparison_tables[2023]}

2023 七站仍全部偏高，且两套地类逐站都比原模型更高；
精细贵得由 +16.84% 变为 +18.63%，Copernicus 由 +15.92% 变为 +26.66%。
这进一步说明不能只展示改善的年份。2023 已多次查看，仍不是新的盲测。

## 5. 下游区间检查结果

由于月候选未通过，本项完成区间诊断，不将其作为已验收上游模型再率定下游。
下表是原七区模型 2019—2022 固定早期率定的区间误差，单位亿 m³；
采用相邻测站径流差，允许区间净增量为负。
平均有符号误差反映偏高／偏低，平均绝对误差（MAE）反映大小，不计算易失真的区间百分比。

{table(["区间", "精细平均误差", "精细 MAE", "Copernicus 平均误差", "Copernicus MAE"], reach_rows)}

贵得—兰州在固定和逐年率定下，两套地类的四个检验年均偏低；这是持续存在的局地问题。
兰州—头道拐固定率定多偏高，部分抵消上游低估；现有证据不支持把它当作主要共同误差来源。
花园口—利津误差也较大，但利津与入海口管理账户边界不完全一致，不能直接据此加损失参数。

另外复核“兰州来水已知”的条件诊断：固定早期率定的下游五站相对 RMSE 为精细 5.82%、Copernicus 7.15%。
它只定位剩余局地误差，不属于独立预测精度。实际月试点没有达到这个条件上限。

## 6. 结论与后续优先级

1. 月尺度过程有改善精细分类部分年份的信号，但当前最小试点没有稳定通过验证；
   保留为失败／局部有效实验，不进入生产。
2. 两套地类对同一过程结构的响应不同，不能据本次结果宣称精细分类已获得可靠的跨年优势。
3. 上游重点应同时覆盖贵得以上和贵得—兰州：前者仍有跨年过程问题，后者有持续局地低估。
4. 下一轮应优先增加上游降水、蒸散及雨雪季节分配的独立约束；
   在这些约束不足时，继续增加自由参数容易补偿错误输入。

这里没有证明真实误差唯一来自某个过程。当前月模型的雨雪与空间简化也可能导致失配，不能据失败结果否定所有月模型。

## 7. 复现、核验及资料来源

结果目录：`project/calibration/monthly_pilot_v1/`。
包含 `protocol.json`、气候下载 URL/指纹、60 项来源校验、所有率定参数、多起点优化状态、
月收支、逐站预测、删年结果和区间误差。
主试点率定与模拟约 {summary["elapsed_seconds"]:.1f} 秒，
峰值进程树约 {summary["peak_process_tree_mib"]:.0f} MiB，两个单线程进程。
该时间不含前期下载、输入整理和补充年尺度对照。
月水量闭合最大残差 {summary["max_monthly_closure_mm"]:.2g} mm，为数值舍入量级。
核验包含缺失账户传播、未来目标排除、预测因果性、非负储量／通量、
官方 AWY 方程对照及下游区间增量不变。14 项相关测试及原长序列 672 行核验通过。

![三项结果图](../project/calibration/monthly_pilot_v1/pilot_validation.png)

- [全部逐站预测](../project/calibration/monthly_pilot_v1/all_station_results.csv)
- [2018 逐站对照](../project/calibration/monthly_pilot_v1/station_comparison_2018.csv)
- [2023 逐站对照](../project/calibration/monthly_pilot_v1/station_comparison_2023.csv)
- [跨年逐站指标](../project/calibration/monthly_pilot_v1/validation_metrics.csv)
- [区间误差](../project/calibration/monthly_pilot_v1/reach_increment_errors.csv)
- [最终判定](../project/calibration/monthly_pilot_v1/final_assessment.json)

现有完成产物可重建汇总：

```powershell
./scripts/run.ps1 python scripts/summarize_monthly_pilot.py
./scripts/run.ps1 python scripts/report_monthly_pilot.py
```

首次计算顺序为 `prepare_monthly_pilot.py` → `run_monthly_pilot.py` →
`summarize_monthly_pilot.py` → `check_monthly_reference.py` → 再汇总和报告。
输入准备与主率定拒绝覆盖已完成产物；补充对照 `--rebuild` 只重建自身审计。

方法依据与局限：

- [InVEST AWY 官方方法与年尺度局限](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest/3.17.0/userguide/en/annual_water_yield.html)。
- [TerraClimate 提供方：月气候、模型水量与局限](https://www.climatologylab.org/terraclimate.html)。
  本试点取 V1.1 P/PET/Tmin/Tmax，不以外部模拟 AET/Q 为实测率定目标。
- [hydromad 概念水文框架](https://hydromad.github.io/articles/tutorial.html)：土壤水量核算与径流滞蓄分开表达；本项目未复现其具体模型。
"""
    (root / "docs/MONTHLY_PILOT_RESULTS.md").write_text(text, encoding="utf-8")
    print(pd.DataFrame(supplementary).to_string(index=False))
    print(gate_rows)


if __name__ == "__main__":
    main()
