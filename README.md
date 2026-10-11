# 黄河流域年度径流模拟

本项目基于 InVEST Annual Water Yield，比较精细分类、精细分类的大类、Copernicus 和 ESA WorldCover 对黄河流域测站年径流的模拟，并为后续 CMIP6 和地类变化实验提供基础模型。

## 当前任务

精细分类的两个开发期数值目标已完成，其余三种方案保持冻结。当前状态与执行边界见 [DEVELOPMENT.md](DEVELOPMENT.md)。

以后固定输出三张表：

1. 2022、2023、2024 年总误差及三年整体误差：精细分类必须 4/4 严格最优。
2. 2022—2024 的 21 个站年误差：精细分类必须至少 9/21 严格最优。
3. 4×4 参数互换矩阵：用于辅助确认精细分类互换后没有明显偏弱。

最新固定结果为 [三张结果表](experiments/fine_2023_focus/20261011T164652_986860Z/report/three_tables.html)：表一 4/4，表二 14/21；2023 年误差 5.958465%，三年整体 5.567379%。详细口径见 [验收目标](docs/TWO_ACCURACY_GOALS.md)，完整过程、参数、科学限制及清理记录见 [本次实验记录](docs/FINE_2023_FOCUS.md)。

2022—2024 直接参与本轮拟合和选型；历史折重拟合 CV 从 14.091% 升到 18.536%。结果不能作为独立验证、地类优越性或未来情景预测可靠性的证明。完整站年、固定模型、水量账户和不利结果均保留。

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

- 配置：`config/fine_2023_focus.yaml`
- 实验：`scripts/experiment_fine_2023_focus.py`
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
