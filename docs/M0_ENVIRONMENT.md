# M0 环境与工程初始化记录

验收日期：2026-10-08。

## 环境选择

初始系统 Python 为 3.14.7，系统 `uv` 的 WinGet 链接无法正常启动。项目选择 Python 3.12，并以 conda-forge 原生包建立独立环境，避免 Windows 上单独编译 GDAL 及混用二进制依赖。未修改系统 Python、已有 Anaconda 环境或 `AGENTS.md`。

InVEST 3.20.2 的 PyPI 发布元数据要求 Python >=3.10、GDAL 3.10.*；项目将 GDAL 限制在该系列，并通过实际导入和读写测试验收。资料来源：[发布元数据](https://pypi.org/pypi/natcap.invest/3.20.2/json)、[官方安装说明](https://invest.readthedocs.io/en/stable/installing.html)。

| 组件 | 已验证版本 |
| --- | --- |
| Python | 3.12.15 |
| InVEST | 3.20.2 |
| GDAL Python 绑定 / 运行库 | 3.10.3 / 3.10.3 |
| NumPy / pandas / SciPy | 2.5.3 / 3.0.6 / 1.18.1 |
| PyGeoprocessing / TaskGraph | 2.4.11 / 0.11.2 |
| Pydantic / PyYAML / psutil | 2.13.5 / 6.0.3 / 7.2.2 |
| pytest / Ruff / uv | 9.1.1 / 0.16.10 / 0.12.21 |
| micromamba | 2.9.0 |

`locks/conda-win-64.lock` 锁定全部 conda 包的下载地址、构建和 MD5；`scripts/bootstrap.ps1` 另外使用 SHA256 核验固定版本的 micromamba 下载包。环境仅面向 Windows x86-64，其他平台需要单独求解与验证。

`environment.yml` 描述环境意图，`pyproject.toml` 描述本项目的 Python 依赖、包结构和 Ruff/pytest 配置。复现时先使用 conda 显式锁创建环境，再通过环境内的 uv 离线、无依赖、无构建隔离地安装本地包，防止 uv 替换原生依赖。不生成会误导为可单独安装完整原生环境的 `uv.lock`。

## 工程产物与验证范围

- `src/ecologyhydro/config.py`：YAML 安全读取、重复键拒绝、类型和范围校验、项目根目录路径解析、模型输入文件存在性检查。
- `src/ecologyhydro/logging_utils.py`：控制台和 UTF-8 文件日志，重复初始化不会累加处理器，也不改动根日志配置。
- `src/ecologyhydro/runtime.py`：在数值库导入前限制 OpenMP、BLAS、MKL、NumExpr 和 GDAL 线程。
- `src/ecologyhydro/diagnostics.py`：真实磁盘 GeoTIFF/GeoPackage 读写、坐标系/属性/NoData 核验、PyGeoprocessing 计算和 TaskGraph 双进程执行。
- `scripts/run.ps1` 与 `scripts/bootstrap.ps1`：项目环境运行入口、锁定环境安装和依赖检查。
- `scripts/test/`：配置错误、相对路径、训练/验证隔离、输入存在性和中文日志测试。

M0 首次成功的 `doctor` 检查约耗时 8 秒，使用两个 TaskGraph 工作进程、每进程一个数值库线程。测试使用 3×2 合成栅格及一个矢量点，仅验证软件与原生库链路，不能据此推断全流域性能或模型精度。成功报告保存于 `project/logs/environment_*.json`，完整检查命令见 README。

配置/日志测试共 14 项通过；Ruff 检查及格式检查通过。`uv pip check` 用于确认 Python 包依赖关系，原生库实际兼容性另由 `doctor` 验证。

## 留给后续阶段的工作

- M1：真实数据盘点、测站清单、年份划分和数据版本；当前 `null` 与空列表表示尚未确定。
- M2：研究投影、公共网格、完整空间质量检查和测站汇水区；自检 EPSG 仅为合成夹具坐标系。
- M3：官方 AWY 完整调用、进程树资源监控、12 小时超时控制和全域运行验收。目前未运行真实水文模拟。
- M1 已因本地 CMFD 文件采用 NetCDF4 格式，补充 `netCDF4 1.7.4` 并更新环境锁；原有 Python/InVEST/GDAL/NumPy 版本保持不变。xarray 等其他依赖待实际需要时再引入。
