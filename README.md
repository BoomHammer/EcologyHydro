# 黄河流域年度径流模拟

基于 InVEST Annual Water Yield（AWY），使用精细分类和 Copernicus 两套土地覆盖，模拟黄河干流 11 个测站的年水量，为后续 CMIP6 和土地覆盖变化实验提供基础。

**已完成共和内流盆地、跨年蓄变和区域降水重建的进一步检验，精度尚未验收。** 当前实验由 [runoff_experiment.json](config/runoff_experiment.json) 定位。最新见 [连通性与跨年蓄变诊断](docs/CONNECTED_STORAGE_DIAGNOSIS.md)，前轮 [内流区与荒漠诊断](docs/DRYLAND_CONNECTIVITY_DIAGNOSIS.md) 及 [分区与水量诊断](docs/REGIONAL_WATER_DIAGNOSIS.md) 保留。

本轮发现共和内流盆地被通用填洼误连到黄河的证据。排除候选范围后，精细六站留一年／2023 MAPE 为 10.15%／22.33%；新增外部土壤／雪水蓄变后为 13.32%／6.65%。**复用年大幅改善没有通过跨年稳定性检查，不能将 6.65% 宣称为正式精度。** 开发年排序仍支持上一轮候选，未替换生产参数。

精细地类在相同六站上，统一 Z 无管理的训练／2023 MAPE 为 15.85%／43.45%；三分区加管理降至 8.63%／24.16%。四分区进一步改善开发年留一年误差至 10.23%，但 2023 为 24.38%，未继续改善。新 2023 结果均为复用测试年；管理账户仍是条件情景，尚未覆盖全部 11 站。

新加一个未识别净损失参数后，精细六站训练／留一年／2023 MAPE 为 6.86%／9.95%／23.10%，改善有限。空间核查不支持整片内流区计入兰州—头道拐是主要原因；却发现裸沙 Kc=0.15 的假设不受 Z 调节，以及贵得上游较大的参考内流面重叠。未按 2023 选择较高裸地 Kc 或裁剪生产边界。

以下是保留的旧 **11 站** 冻结实验，不能与新六站平均值直接混比：

| 土地覆盖 | 原工程基线训练 MAPE | 率定后训练 MAPE | 2023 测试 MAPE |
| --- | ---: | ---: | ---: |
| 精细分类 | 316.46% | 14.46% | 41.10% |
| Copernicus | 342.33% | 14.00% | 42.15% |

训练为 2019—2022 年。旧实验直接拟合受调控实测径流；新诊断显式加入已知管理水量并重新率定，但地下水、引调水、蒸散重叠和统计边界仍未闭合，不能据此认定精细分类稳定更优。

## 阅读入口

- [开发状态与下一步](DEVELOPMENT.md)：进度、优先事项及实验约束。
- [数据说明](docs/DATA_DESCRIPTION.md)：数据、单位、确认口径和已知冲突。
- [预处理](docs/M2_PREPROCESSING.md)：网格、土壤、湖泊、内流河与缓存。
- [参数来源](docs/M4_PARAMETER_SOURCES.md)：文献、轮作近似和率定参数。
- [本轮结果](docs/MAJOR_BIAS_CALIBRATION.md)：方法、冻结测试及性能。
- [分区与水量诊断](docs/REGIONAL_WATER_DIAGNOSIS.md)：最新对照、四分区补充实验及未闭合水量。
- [内流区与荒漠诊断](docs/DRYLAND_CONNECTIVITY_DIAGNOSIS.md)：空间图、裸地参数敏感性、净损失对照及剩余问题。
- [连通性与跨年蓄变](docs/CONNECTED_STORAGE_DIAGNOSIS.md)：共和闭流域路径、土壤／雪水账户、降水重建及失败对照。
- [输出字段](docs/OUTPUT_FIELDS.md)：产水、实测径流及指标含义。

## 环境与检查

在项目根目录用 PowerShell 执行。项目使用独立 Windows x64 环境：Python 3.12、InVEST 3.20.2、GDAL 3.10 系列；具体构建见 `locks/conda-win-64.lock`。已有环境直接使用 `scripts/run.ps1`，新环境才执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
```

常用检查：

```powershell
./scripts/run.ps1 python -m ecologyhydro doctor
./scripts/run.ps1 python -m pytest
./scripts/run.ps1 python -m ruff check src scripts
./scripts/run.ps1 python -m ruff format --check src scripts
```

若执行策略阻止脚本，使用 `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 ...`。

## 当前结果复核

```powershell
./scripts/run.ps1 python scripts/calibrate_major_bias.py --verify-only
./scripts/run.ps1 python scripts/validate_major_bias.py
```

第一条复算或复用已选训练参数，第二条校验冻结输入并复算或复用 2023 测试。原实验已冻结，不允许原地重新率定。历史工程基线由 `config/analysis_baseline.json` 定位。

`config.yaml` 管理年份、资源和路径；通用输入及 Z 的空值是模板占位，实际任务从 M2 索引、情景和冻结参数装配，无需逐项人工补填。预处理配方见 `config/m2.yaml`。

## 目录与脚本

| 路径 | 用途 |
| --- | --- |
| `data/` | 原始数据 |
| `config/` | 配方、先验与实验索引 |
| `src/ecologyhydro/` | 预处理、模型执行、汇总、试验和冻结校验 |
| `scripts/test/` | 自动化测试 |
| `project/cache/` | 基础数据缓存及指纹 |
| `project/calibration/major_bias/` | 当前训练、参数与冻结测试结果 |
| `project/repairs/`、`project/m4/` | 修复基线及历史诊断产物 |
| `experiments/` | 通用模拟入口的运行输出 |
| `docs/` | 五份专题文档 |

除环境脚本外，保留以下入口：

- `fetch_climate_reference.py`、`check_provider_climate.py`：提供方气候下载及转换。
- `calibrate_major_bias.py`、`validate_major_bias.py`：训练与冻结测试。
- `calibrate_regional_water.py`、`refine_regional_water.py`：新三／四分区诊断；拒绝覆盖已有实验。
- `summarize_regional_water.py`：对照指标、结构误差下限与 PNG/PDF 图；加 `--refinement project/calibration/regional_water_v2` 汇总四分区。
- `audit_dryland_connectivity.py`、`audit_bare_evaporation.py`：内流面与荒漠产水核查、裸地蒸发敏感性。
- `calibrate_reach_loss.py`、`summarize_dryland_reach.py`：一个区间净损失参数的开发年检验、空间与六站对照图。
- `audit_guide_connectivity.py`：青海湖／共和盆地身份与 D8 路径核查，独立候选范围。
- `prepare_storage_diagnostic.py`、`prepare_reference_precipitation.py`：月末土壤／雪水差值及条件降水重建。
- `calibrate_connected_water.py`、`summarize_connected_water.py`、`verify_connected_outputs.py`：新对照、开发年排序及实际产物检查。
- `audit_routing_topology.py`、`check_m2_repairs.py`：河网和修复检查。
- `audit_sector_water.py`：行业供耗水核对。
- `audit_m4.py`、`audit_model_chain.py`：水量账户和独立方程审计；其函数也用于回归测试，完整入口针对历史基线。
- `inspect_run_yield.py`：运行栅格数值诊断。
- `run_repaired_baseline.py`：历史固定参数基线重建，会写入基线索引；当前冻结实验无需执行。
- `export_mod16a2gf_gee.js`：可选蒸散导出，目前暂停使用。

分块读取、依赖缓存和双进程并发控制资源；每模型内部单线程，进程树预算 24 GiB、单次上限 12 小时。`AGENTS.md` 仅由人工修改。
