# 黄河流域年度径流模拟

本项目基于 InVEST Annual Water Yield，比较精细分类、精细分类的大类、Copernicus 和 ESA WorldCover 对黄河流域测站年径流的模拟，并为后续 CMIP6 和地类变化实验提供基础模型。

数据、方法及软件的参考文献、DOI 和使用状态见 [参考文献汇总](docs/REFERENCES.md)。

从 2026-10-19 起，新的水文实验必须同时计算水和沙。新增 InVEST SDR 入口 `scripts/run_sediment.py`，配置 `config/sediment.yaml`；分类体系 C/P 转换、K 不重复乘 0.1317、年均 R 的适用范围及交付要求见 [水沙联合实验约定](docs/SEDIMENT_SDR.md)。SDR 坡面入河泥沙与测站输沙量分别报告。

## 当前任务

已按最新指示实施“先分离选型并封存、再做两组公平比较”，细类参数库保持不变，执行记录见 [RIGOROUS_COMPARISON_PLAN.md](docs/RIGOROUS_COMPARISON_PLAN.md)。原三参数开发期达标版本保留，其三个历史对照保持冻结；新公平实验单独验收。当前状态与执行边界见 [DEVELOPMENT.md](DEVELOPMENT.md)。

本轮[公平比较结果](experiments/separated_selection/20261011T175947_776825Z/three_tables.html)：独立率定 **0/4、7/21**，共用参数 **1/4、7/21**，均未达标。旧基线的 4/4、14/21 不代表新实验获胜。

以后固定输出三张表：

1. 2022、2023、2024 年总误差及三年整体误差：精细分类必须 4/4 严格最优。
2. 2022—2024 的 21 个站年误差：精细分类必须至少 9/21 严格最优。
3. 4×4 参数互换矩阵：用于辅助确认精细分类互换后没有明显偏弱。

保留的开发期基线为 [三张结果表](experiments/scientific_robustness_final/20261011T172623_968568Z/report/three_tables.html)：表一 4/4，表二 14/21；2023 年误差 5.791023%，三年整体 5.696205%。详细口径见 [验收目标](docs/TWO_ACCURACY_GOALS.md)，该版本的对照、敏感性和资料缺口见 [基线实验记录](docs/SCIENTIFIC_ROBUSTNESS.md)，首版实验及清理记录见 [FINE_2023_FOCUS.md](docs/FINE_2023_FOCUS.md)。

2022—2024 直接参与拟合和选型；历史 CV 从上一版 18.536% 改善至 17.874%，仍差于原历史基线的 14.091%。对等追加搜索中精细分类仅为 1/4、5/21，未达同样目标。结果不能作为独立验证、地类优越性或未来情景可靠性的证明。新独立验证协议已封存，但仍待完整外部资料。

## 水文口径

- 旱地是雨养农田，不灌溉；水浇地才允许灌溉补水。
- InVEST 年产水量与测站实测径流并非同一概念。河道传输、库坝调蓄、耗水、外调水和储量变化需要单独核对。
- `data/Hydrology/` 中新增的三份水资源公报用于核对概念、单位和水量账户，不能将外调水直接当成本地灌溉，也不能重复扣除已经计入 ET 的耗水。

## 项目结构

```text
data/          原始气象、DEM、地表覆盖、土壤和水文数据
project/       可复用的预处理数据与缓存
experiments/   每次实验的协议、日志、候选与结果
config/        实验配置
scripts/       程序入口与报告脚本
src/           Python 源码
docs/          当前实验记录、方法、数据与字段说明
```

## 当前入口

- 分离选型与公平比较：`scripts/experiment_separated_selection.py`，配方 `config/separated_selection.yaml`
- 筛选配置：`config/scientific_robustness.yaml`、`config/scientific_robustness_refined.yaml`
- 筛选：`scripts/experiment_scientific_robustness.py`
- 当前发布：`scripts/publish_scientific_robustness.py`，配置 `config/scientific_robustness_publish.yaml`
- 报告：`scripts/report_fine_focus.py`
- 目标判定：`src/ecologyhydro/fine_focus_goals.py`
- 冻结对照：`experiments/calibration_followup/20261011T154702_556627Z`

旧实验入口、未使用配方和重复文档已清理。当前计算和测试仍依赖的公共函数保留，部分文件沿用旧名称。历史源码、配置和文档可从[清理前快照](experiments/maintenance/20261012_experiment_cleanup/source_before_cleanup.zip)恢复；全部历史实验结果继续保留。

## 环境与检查

Windows 环境使用 Python 3.12、InVEST 3.20.2 和 GDAL 3.10 系列，依赖锁见 `locks/conda-win-64.lock`。首次安装：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
```

常用检查：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python -m ecologyhydro doctor
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python -m pytest
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python -m ruff check src scripts
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python -m ruff format --check src scripts
```

大栅格计算使用分块和缓存，默认采用受控并行并限制内存占用，以适配 32 GiB 内存。外部数据与方法必须在相应专题文档中记录来源和参考文献。`AGENTS.md` 只能由人工修改。
