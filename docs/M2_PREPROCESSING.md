# 预处理与模型约定

更新：2026-10-10。M2 工程预处理及修复完成。官方 AWY、率定和测试结果见 [本轮结果](MAJOR_BIAS_CALIBRATION.md)。

## 配方和入口

```powershell
./scripts/run.ps1 python -m ecologyhydro preprocess
# 仅需部分阶段时：
./scripts/run.ps1 python -m ecologyhydro preprocess --stages routing quality
```

配置为 `config.yaml`，配方为 `config/m2.yaml`。预处理发布 `project/cache/m2/latest.json` 并保留运行索引。当前实验使用固定的 `project/repairs/soil_routing/m2_index.json`；不将新预处理结果覆盖进冻结实验。

基础配方保留 CMFD prec 工程基线；当前 TerraClimate V1.1 成对气候由实验情景覆盖，未改写基线。

## 网格及输入

- WGS84 Albers 等积投影，中央经线 105°、标准纬线 25°/47°，250 m，当前 8901 × 5012 像元，每像元 0.0625 km²。
- Copernicus 外包范围决定网格，原点向外对齐；边界 shp 不硬裁完整上游。面积用实际等积区域，不以表格替代。
- 类别最近邻、连续属性双线性重采样。未定义类别触发质量检查，不静默删去其面积。
- 气候先在原生网格年度聚合，再重投影；缺月、重复月、异常单位及无效覆盖报错。CMFD 通量乘月秒数，旧 GEE PET 只乘一次 0.1，提供方 NetCDF 按属性解码。
- 全球土壤只裁研究区窗口，其他大栅格也分块读取。

## 湖泊、河网与测站

1. 现有湖泊 shp 生成共用掩膜，分别覆盖 fine=25、Copernicus=80，保留岛屿、原图和改变面积记录。覆盖地类与排除内流拓扑是两个步骤。
2. 按 HydroRIVERS 下游连接和上游面积识别外流干流；测站吸附干流线段最近点，另存原点、匹配点及位移，不改吸支流。
3. 外流支流采用 `ENDORHEIC=0` 且上游面积至少 1000 km² 的河段，按 `NEXT_DOWN` 构建无环连接，降低河道 5 m 并约束 D8；这是工程近似，不是实测河床。
4. 内流河道在填洼前设为外流域之外的排水边界。湖泊 shp 未含青海湖，另以闭流证据、种子及连通 Copernicus 水面排除，不用精细图错误湖泊类别圈定。
5. 出口在 2 km 搜索上限内选已约束干流最近像元，避免按最大累积量跨过支流汇口。检查完整上游嵌套和已标注内流河/湖排除。
6. 生成非重叠 `zones.tif`、`zones.gpkg` 和 `zone_station.csv`，每个分区在单站累计中只计一次。

青海湖种子和搜索框在配方中，闭流依据见 [FAO](https://www.fao.org/4/ag186e/AG186E03.htm)，河网字段见 [HydroRIVERS 技术文档](https://data.hydrosheds.org/file/technical-documentation/HydroRIVERS_TechDoc_v10.pdf)。

仍有 286 个河网拓扑采样点分歧。新增 HydroBASINS 面状核查发现，贵得上游增量区 2 与参考内流面重叠约 10790.94 km²；兰州—头道拐重叠约 196.56 km²。点状内流河排除不等于整个内流汇水面已排除，详见 [空间核查及水量敏感性](DRYLAND_CONNECTIVITY_DIAGNOSIS.md)。参考图与 HydroRIVERS 同源且可能遗漏小盆地，尚未直接裁剪生产边界。面积和降水表用于复核，不强行匹配。唐乃亥计算面积约 122,925.875 km²、贵得约 145,376.875 km²，须区分各自参考口径。

## 土壤与参数

2026-10-10 后续定位：贵得增量区的大重叠主要属于共和盆地参考内流面 `4060051460`，不是青海湖本身。D8 抽查显示通用填洼可将内部洼地抬高约 112 m 并形成外排。`project/diagnostics/gonghe_connectivity_v1/zones.tif` 仅为排除此固定参考面的候选产水域；生产 D8、旧索引和历史边界没有替换。详见 [路径证据与精度检验](CONNECTED_STORAGE_DIAGNOSIS.md)。

HWSD2 按单元 ID 关联 `HWSD2_SMU/HWSD2_LAYERS`，取 `SEQUENCE=1` 优势组分。当前按分层质地、粗颗粒、矿物及盐分等计算并积分容量，已替代旧单元 AWC 简单除以 1000 的处理。

土壤限制层深度仍按等级赋 1500/750/300/50 mm，属于代理。植被根深与限制层分别保存，AWY 使用较小者。不同表同名 AWC 不能混用。

功能组先验、候选参数和轮作近似分别见 `biophysical_priors.yaml`、`biophysical_candidates.yaml`、`crop_systems.yaml`。当前最终运行表位于率定目录，来源及局限见 [参数文档](M4_PARAMETER_SOURCES.md)。

## 水量含义与数值检查

AWY 的 `Y=P−AET` 已扣模型蒸散。入渗是内部转移，不是永久消失，不能任意再扣。年度 AWY 不显式模拟完整地下水、跨年蓄变、灌溉回归水及水库调度；当前累计产水不是严格天然径流。

Float32 零边界会产生微小负产水。汇总采用固定 0.0005 mm 容差，仅小负值按零汇总，并记录数量、最小原值及改变量，原栅格不修改。2023 两像元改变量共约 0.0136 m³；更大负值、缺失或面积丢失仍报错。

## 缓存和复核

缓存分网格、土壤、气候裁剪/年和、对齐、地类参数/湖泊覆盖、河网及质量检查。manifest 记录输入和实现指纹、参数、版本、输出校验和及耗时；成功才发布。相关输入或实现变化使缓存失效，外部修改校验失败时拒绝静默复用。

保留历史缓存及冻结索引。独立官方任务双进程，每任务单线程；预处理避免大量并发争用磁盘，限值见根配置。

专项检查为 `scripts/check_m2_repairs.py`、`scripts/audit_routing_topology.py`。旧阶段验收及重复诊断报告已合并，详细历史产物仍在 `project/cache/`、`project/repairs/`、`project/m4/`。
