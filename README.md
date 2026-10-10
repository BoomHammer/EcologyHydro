# 黄河流域年度径流模拟

基于 InVEST Annual Water Yield（AWY），比较精细分类、Copernicus、ESA WorldCover v200，并研究气候和地类变化对年水量的影响。
工程可输出 11 站产水；目前管理水量可对应的径流比较为七站，利津单列边界近似。

## 当前进展

**2026-10-11 当前协议**：2011—2012 预热，2013—2017 与 2019—2021 合并率定，2018 桥接不评估，
2022—2024 正式验证。新水文表已接入，2011—2024 成对 P/PET 已补齐；三个地类产品各率定一套参数，
再固定参数执行 **3×3 九组合**。详情及最新试跑结果见 [三产品交叉实验](docs/LANDCOVER_CROSS_COMPARISON.md)。
所有当前输入路径、时期、下载来源及资源限制由 [YAML 配方](config/landcover_comparison.yaml) 管理。
本轮已完成并通过全像元精度检查，输出 882 行七站条件径流、1386 行十一站产水及九份独立组合 CSV。
三年验证的产品排序依赖参数来源，不能据此宣布精细图普遍更优。

数据预处理、汇水区修复、年模型率定、时间顺序验证和上游月尺度试点均已完成。
**已有结果可支持继续科研分析；CMIP6 和地类变化情景尚未开展。**

下表是**历史双产品试验**：2013—2017 固定率定、2019—2022 七站检验，不是新协议结果；相对 RMSE 越小越好。

| 方案 | 精细分类相对 RMSE | Copernicus 相对 RMSE | 达文献参考标准的站数（精细／Copernicus） |
| --- | ---: | ---: | ---: |
| 七区年模型 | 13.70% | 14.65% | 4/7、3/7 |
| 贵得以上月模型＋原下游区间 | 9.12% | 13.29% | 6/7、6/7 |

“文献参考标准”是逐站 NSE＞0.50、R²＞0.60、绝对 PBIAS≤15%，不等于相对 RMSE＜15%。
每站仅四个检验年，且这些年份已参与开发；数值来源、文献和限制见 [科研适用性核查](docs/DATA_USAGE_AND_RESEARCH_ASSESSMENT.md)。
精细月模型利津 NSE 为 0.453，仍是薄弱站。两套静态地类图年份不同，优势的稳健程度需要配对比较。

七区年模型保留为 AWY 工作参考，月模型作为扩展对照。
月试点未通过原定模型替换规则，与“研究不可继续”不同。
后续先整理地类比较，再用冻结参数开展地类变化及 CMIP6 情景，报告敏感性。
独立 ET、积雪资料可用于专项过程问题，不作为所有后续工作的前置条件。

## 数据与入口

- 原始数据在 `data/`；基础缓存、补下载资料和实验结果在 `project/`。
- `data/Climate/TerraClimate_PET/` 仅有 2019—2023 年；**2013—2018 P/PET 已补齐**，位于 `project/repairs/long_climate_v1/`，每年 12 个月及指纹已核验。
- 新协议统一年度 P/PET 在 `project/comparison_2011_2024/climate/`，覆盖 2011—2024；已有原始文件按 YAML 复用，另补 2011、2012、2024。
- [runoff_experiment.json](config/runoff_experiment.json) 定位当前状态，各实验 `protocol.json` 定义年份与规则。
- `config.yaml` 同步当前率定／验证年份；实际交叉实验须使用 `config/landcover_comparison.yaml`，旧专题脚本不是当前入口。
- 2018 缺管理账户，当前协议正常计算产水但不评估；2022、2023 已被历史开发查看，2024 新增。

## 阅读顺序

| 文档 | 用途 |
| --- | --- |
| [开发状态与计划](DEVELOPMENT.md) | 当前任务、下一步顺序、交付条件 |
| [三产品交叉实验](docs/LANDCOVER_CROSS_COMPARISON.md) | 当前时期、WorldCover、九组合、运行命令及结果 |
| [数据与科研适用性核查](docs/DATA_USAGE_AND_RESEARCH_ASSESSMENT.md) | PET 来源、逐文件使用清单、NSE/PBIAS、方法更正 |
| [月尺度试点](docs/MONTHLY_PILOT_RESULTS.md) | 月模型、原删年检验、下游区间诊断 |
| [长序列验证](docs/LONG_RECORD_VALIDATION.md) | 七区年模型、三区对照及时间验证 |
| [指定年份结果](docs/REQUESTED_YEAR_VALIDATION.md) | 2018 粗略检查和 2023 复用年结果 |
| [数据说明](docs/DATA_DESCRIPTION.md) | 数据、单位、年份及管理账户口径 |
| [预处理](docs/M2_PREPROCESSING.md)、[参数来源](docs/M4_PARAMETER_SOURCES.md)、[输出字段](docs/OUTPUT_FIELDS.md) | 方法、参考文献、字段定义 |

其余专题报告保留历史依据，阶段性的“下一步”以 DEVELOPMENT 为准。
历史实验代码仍承担复现和核验依赖。已撤下的过程代理约束实验见核查报告第 4 节；
删除及恢复说明见 [清理记录](project/archives/cleanup_monthly_constraints_20261010.md)。

## 运行与验证

独立 Windows 环境：Python 3.12、InVEST 3.20.2、GDAL 3.10 系列，构建锁见 `locks/conda-win-64.lock`。
首次安装执行 `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap.ps1`。
已有环境在根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/prepare_comparison.py --config config/landcover_comparison.yaml
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/run_landcover_comparison.py --config config/landcover_comparison.yaml
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python -m ecologyhydro doctor
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python -m pytest
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python -m ruff check src scripts
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python -m ruff format --check src scripts
```

| 工作 | scripts 下的入口 |
| --- | --- |
| 当前三产品准备、九组合运行 | `prepare_comparison.py`、`run_landcover_comparison.py`（均显式传 `--config`） |
| 当前九组合核验与独立结果导出 | `report_landcover_comparison.py`（传 `--config` 和 `--run`） |
| 长序列气候准备、率定 | `prepare_long_climate.py`、`calibrate_long_record.py` |
| 长序列核验、汇总 | `verify_long_record.py`、`summarize_long_record.py` |
| 月试点准备、率定 | `prepare_monthly_pilot.py`、`run_monthly_pilot.py` |
| 月试点汇总、报告 | `summarize_monthly_pilot.py`、`report_monthly_pilot.py` |
| 指定年份核算 | `validate_requested_years.py`、`summarize_requested_years.py` |
| 数据使用、PET 和科研指标审计 | `audit_data_usage.py` |

准备和率定入口通常拒绝覆盖完成目录；阅读脚本参数后再复现，已有结果无需重复率定。
旧审计输出在 `project/diagnostics/data_usage_review/`，对应更新前快照，不代表用户替换表格后的当前库存。

大栅格分块、缓存复用，默认双进程、每任务单线程；内存预算 24 GiB，单次上限 12 小时。
原始数据和历史结果保留；`AGENTS.md` 仅由人工修改。
