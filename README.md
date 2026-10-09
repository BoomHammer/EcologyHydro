# 黄河流域年度径流模拟

基于 InVEST Annual Water Yield（AWY），使用精细分类和 Copernicus 两套土地覆盖，模拟黄河干流 11 个测站的年水量，为后续 CMIP6 和土地覆盖变化实验提供基础。

**研究推进暂告一段落。第一轮率定及 2023 测试已完成，精度尚未验收。** 当前实验由 [runoff_experiment.json](config/runoff_experiment.json) 定位。

| 土地覆盖 | 原工程基线训练 MAPE | 率定后训练 MAPE | 2023 测试 MAPE |
| --- | ---: | ---: | ---: |
| 精细分类 | 316.46% | 14.46% | 41.10% |
| Copernicus | 342.33% | 14.00% | 42.15% |

训练为 2019—2022 年。当前直接拟合受调控实测径流，尚未显式处理沿程耗水和水库蓄变，不能据此认定精细分类稳定更优。2023 已用于测试，后续参考其误差改模型时须标为复用测试年。

## 阅读入口

- [开发状态与下一步](DEVELOPMENT.md)：进度、优先事项及实验约束。
- [数据说明](docs/DATA_DESCRIPTION.md)：数据、单位、确认口径和已知冲突。
- [预处理](docs/M2_PREPROCESSING.md)：网格、土壤、湖泊、内流河与缓存。
- [参数来源](docs/M4_PARAMETER_SOURCES.md)：文献、轮作近似和率定参数。
- [本轮结果](docs/MAJOR_BIAS_CALIBRATION.md)：方法、冻结测试及性能。
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
- `audit_routing_topology.py`、`check_m2_repairs.py`：河网和修复检查。
- `audit_sector_water.py`：行业供耗水核对。
- `audit_m4.py`、`audit_model_chain.py`：水量账户和独立方程审计；其函数也用于回归测试，完整入口针对历史基线。
- `inspect_run_yield.py`：运行栅格数值诊断。
- `run_repaired_baseline.py`：历史固定参数基线重建，会写入基线索引；当前冻结实验无需执行。
- `export_mod16a2gf_gee.js`：可选蒸散导出，目前暂停使用。

分块读取、依赖缓存和双进程并发控制资源；每模型内部单线程，进程树预算 24 GiB、单次上限 12 小时。`AGENTS.md` 仅由人工修改。
