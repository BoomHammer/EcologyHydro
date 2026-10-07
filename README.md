# EcologyHydro

基于 InVEST Annual Water Yield 的黄河流域年径流模拟项目。开发顺序和三个实验的设计见 [DEVELOPMENT.md](DEVELOPMENT.md)。目前已完成 M0 环境与工程初始化，尚未接入真实研究数据或运行流域模拟。

## 环境

已验证 Windows x86-64、Python 3.12.15、InVEST 3.20.2、GDAL 3.10.3。全部原生依赖来自 conda-forge，具体版本、构建和下载校验值保存在 [环境锁文件](locks/conda-win-64.lock)。

采用项目内 micromamba 环境 `.venv/`，`uv` 仅用于无依赖的本项目可编辑安装及依赖检查。GDAL 与 InVEST 的原生库由同一包管理器提供，这是 [InVEST 官方推荐的安装方式](https://invest.readthedocs.io/en/stable/installing.html)。环境选择及验收记录见 [M0 环境说明](docs/M0_ENVIRONMENT.md)。

在项目根目录的 PowerShell 执行以下命令。`ExecutionPolicy Bypass` 只作用于该次子进程，不修改系统执行策略：

```powershell
# 新机器上从锁文件创建环境；已有匹配环境时只核验并重装本项目代码。
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap.ps1

# 验证配置结构，输出解析后的绝对路径。
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python -m ecologyhydro validate-config

# 检查原生模块、栅格/矢量读写及 TaskGraph 双进程计算。
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python -m ecologyhydro doctor

# 运行测试与代码规范检查。
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python -m pytest
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 ruff check .
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 ruff format --check .
```

`scripts/run.ps1` 会定位项目根目录并在独立环境中执行命令；不会依赖系统默认 Python 或已激活的 Anaconda 环境。IDE 可选择 `.venv/python.exe`，原生库运行仍推荐通过脚本或正确激活该 conda 环境。

不要在本环境中直接执行 `uv sync` 或用 pip 升级 GDAL；本项目的完整锁是 conda 显式锁，不是 `uv.lock`。`environment.yml` 用于有意更新依赖时重新求解，不能替代精确锁文件。

## 配置与目录

编辑 [config.yaml](config.yaml)，所有相对路径均相对于项目根目录，即使配置文件位于 `config/` 也不改变此规则。未知字段、重复 YAML 键、重复站点、校准/验证年份重叠、无效资源预算都会报错。

| 配置 | 用途 |
| --- | --- |
| `paths` | 气象、DEM、土地覆盖、土壤、`data/Hydrology/` 观测数据及缓存、日志、实验目录 |
| `study` | 站点、校准/验证年份、投影和分辨率；M1/M2 补齐 |
| `inputs` | AWY 所需文件路径及数据版本；目前为未配置状态 |
| `model.z_parameter` | 率定参数，未指定默认科学值 |
| `resources` | 初始并发上限 2、每进程数值库线程 1、内存预算 24 GB、超时上限 12 小时 |
| `random_seed`、`log_level` | 随机种子及日志级别 |

`validate-config` 接受 M0 的空研究参数；加入 `--require-inputs` 后会检查所需文件、站点、网格参数与 Z 是否已填写。当前模板运行该模式应以退出码 2 拒绝启动。空间一致性与正式模型校验在 M2/M3 实现。

`doctor` 仅使用临时合成数据，完成后清理临时文件；UTF-8 日志和 JSON 报告写入 `project/logs/`。`project/cache/` 保存共享缓存，正式实验结果写入 `experiments/`；这些产物和原始数据均被 Git 忽略。

线程限制已在导入数值库前设置，双进程已通过自检。内存预算与 12 小时上限目前是经过验证的配置约束，实际进程树监控、超时终止、批量调度和全流域性能验收属于 M3，尚未实现。

## 开发规则

Python 源码放在 `src/ecologyhydro/`，测试放在 `scripts/test/`。提交前运行测试、`ruff check .` 和 `ruff format --check .`。新增模块通过可编辑安装立即生效；改变依赖后需重新求解、锁定并完成环境自检。

下一步是 M1：盘点实际数据、确定测站及观测口径、划分校准与验证期。`AGENTS.md` 仅由人工修改。
