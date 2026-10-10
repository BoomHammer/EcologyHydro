# 共和内流盆地、跨年蓄变与降水输入的进一步检验

日期：2026-10-10。延续 [内流区与荒漠诊断](DRYLAND_CONNECTIVITY_DIAGNOSIS.md)。新产物为 `connected_water_v1/` 和 `reference_rainfall_v1/`，旧参数、原始数据及历史结果保留。

后续已转向 [稳定性诊断](STABILITY_DIAGNOSIS.md)，完成实际 D8 闭流约束和训练子集组合检验。本篇保留前轮结果；其中旧边界的误差排序不作为恢复错误连通范围的理由。

## 本轮结论

找到了较明确的汇水范围问题：共和盆地沙珠玉内流水系的大部分被旧 D8 填洼连到黄河。仅排除已标内流河的线状像元不足以保护完整闭流盆地。

排除这一候选范围并重新率定后，精细六站 2023 MAPE 从 23.10% 小幅降至 22.33%，但开发年留一年 MAPE 从 9.95% 小幅升至 10.15%。进一步加入外部土壤／积雪蓄变后，2023 降至 **6.65%**，留一年却升至 **13.32%**。因此本轮**没有取得跨年稳定优于上一轮的方案**；6.65% 不能作为正式精度宣称。

只按开发年留一年相对 RMSE 在当前候选中排序，两套地类、两版账户仍均选出上一轮 `reach_loss_v1`。这只是误差排序，不表示其旧边界正确或已经可作为生产模型。尚无生产参数获准替换，也没有土地覆盖优劣结论。

## 1. 重叠区域的身份与错误路径

此前贵得上游增量区 2 与全部参考内流面重叠约 10790.94 km²。本轮逐盆地定位发现，主要重叠属于 HydroBASINS `HYBAS_ID=NEXT_SINK=4060051460`，位于青海南山南侧的共和盆地茶卡—沙珠玉水系范围；不是青海湖盆地本身。

青海湖对应 `4060050460`，与贵得增量区仅重叠 26.06 km²；其剔除候选仅用于排除最初的身份猜测，没有进入后续率定。两者必须区分。

空间参考采用 HydroBASINS Asia level 06 v1c 的固定多边形和 NEXT_SINK，未用测站径流或参考面积拟合边界。独立依据包括：

- 共和县政府年鉴明确区分青海湖、沙珠玉内陆水系与黄河外流水系，并区分茶卡—沙珠玉盆地与恰卜恰等河谷。[共和县地理环境（2024年）](https://www.gonghe.gov.cn/lnb/zjgh1/ghnj1__zjgh/ghnj/content_1013653549)
- 关于共和盆地水系演化的研究描述沙珠玉水系与黄河之间的分水岭及其闭流性质。[Drainage system evolution in Gonghe Basin, Northwest Tibetan Plateau: A story of tectonic geomorphology](https://www.sciencedirect.com/science/article/abs/pii/S0169555X26001534)
- HydroBASINS 中 ENDO=2 的 NEXT_DOWN 可表示虚拟连接，不能直接理解为地表外流连接。[HydroBASINS 技术说明](https://data.hydrosheds.org/file/technical-documentation/HydroBASINS_TechDoc_v1c.pdf)

上述依据支持“这一内流水系不应直接作为黄河地表贡献区”，不等于整个地理意义的共和盆地全部闭流，也不排除地下水交换。多边形分水岭仍有分辨率不确定性。

### 路径证据

从该参考面内、原贵得增量区中确定性抽取 12 个像元，沿原 D8 追踪：11 条路径从同一出口附近离开盆地，约为 **100.44224°E、36.23088°N**，路径上的最大填洼抬升为 **112.246 m**；另一个边缘样本仅行进 4 个像元便出界。前 11 条路径在出界前没有经过强制河网连接像元。

因此证据更指向通用填洼把真实内流洼地处理成外排通道，而非强制干流河网直接穿过盆地。现有 HydroRIVERS 在该面内仅有 4 条中心点落入的内流河段；它们被排除不意味着周边完整汇水范围已排除。12 条路径是机制证据，不是逐像元真值验证。

![原模型闭流盆地内的 D8 路径](../project/diagnostics/gonghe_connectivity_v1/routes.png)

新候选只把这一固定参考面内的分区像元置为域外，保留原始 `zones.tif` 和 D8。未重跑并宣称已修复整套 DEM 路由；它是明确、有依据的贡献范围候选。

| 项目 | 原面积 km² | 排除后 km² | 测站参考 km² |
| --- | ---: | ---: | ---: |
| 唐乃亥累计 | 122925.88 | 122919.88 | 121972 |
| 贵得累计 | 145376.88 | 134620.19 | 133650 |
| 兰州累计 | 234613.75 | 223857.06 | 222551 |

移除贵得增量区 10750.69 km²，唐乃亥以上另移除边缘 6.00 km²。贵得面积偏差由约 8.77% 到约 0.73%，作为事后交叉检查，不是边界选择目标。

## 2. 土壤与积雪蓄变：检验年际记忆

新增 TerraClimate **V1.1** 的 `soil` 和 `swe`，通过提供方 NCSS 获取研究区 2018—2023 年每年 12 月状态。原始 NetCDF 变量描述明确为月末土壤水和月末雪水当量；NetCDF4 自动解码比例因子，未重复乘 0.1。使用 `S(本年12月底)−S(上年12月底)`，不把月末储量求和当成年通量。原始请求 URL、变量属性、检索时间和 SHA256 均保存。

TerraClimate 的水量状态来自简化水量平衡模型，使用静态参考地表覆盖、与当前降水／PET 同源的气候输入；不是独立实测，也不完整表达地下水。[提供方方法与局限](https://www.climatologylab.org/terraclimate.html)

条件算子为：

`Q = ΣY_AWY − ΣC_surface − ΣΔS_reservoir − ΣΔS_soil,snow − L_downstream`

土壤／积雪系数固定为 1，不额外率定。正蓄变扣水、负蓄变加水；储量只按空间累积，不跨年再次累加，不把它重复计入水库蓄变。保留负模拟流量作为失败诊断。该算子并未证明外部储量与 AWY 蒸散响应完全一致。

修正候选范围内的累计土壤＋积雪蓄变（亿 m³）：

| 年份 | 贵得 | 兰州 | 头道拐 | 龙门 | 三门峡 | 花园口 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2019 | −17.21 | −27.92 | −27.01 | −25.25 | −16.12 | −16.22 |
| 2020 | −17.40 | −16.91 | −12.02 | −7.74 | 20.90 | 21.15 |
| 2021 | 1.48 | −7.00 | −9.95 | −3.02 | 13.90 | 23.42 |
| 2022 | −13.03 | −20.90 | −26.23 | −32.21 | −85.78 | −92.34 |
| 2023 | 14.75 | 20.80 | 25.18 | 25.92 | 28.45 | 30.24 |

2023 的正蓄变与“同年产水高估”的方向一致；但 2022 下游较大的模型储量释放带来另一组困难。精细六站留出 2022 时，原净损失模型 MAPE 为 11.27%，新增蓄变模型为 **25.98%**；留出 2021 则从 8.48% 改善为 5.94%。这说明新增项并未稳定转移，不能只凭 2023 变好认定跨年过程已识别。

## 3. 独立于径流拟合的降水重建对照

使用用户已有的二级区降水体积和面积，先换算区域平均降水深度：`毫米=降水量(亿m³)×10/面积(万km²)`。对前六个站间代理区逐年缩放 TerraClimate P，保持区内空间比例，使区域均值与表格深度一致；没有用径流确定缩放系数，没有把统计面积代替像元面积，没有调整 PET。花园口以下及计算域外保持原降水。

这是带额外降水资料的回顾性重建，2023 使用当年降水作为气象输入，未使用当年径流拟合。由于二级区与测站范围仍非完全一致，不将其称为经过验证的逐像元偏差订正，也不能直接转用未来 CMIP6。

例如修正范围后，上游 2023 TerraClimate 均值 628.10 mm、表格代理值 586.40 mm，系数 0.9336；兰州—头道拐分别为 170.87 和 214.97 mm，系数 1.2581。各区、各年偏差不同，简单全流域乘一个系数没有依据。

这项对照在开发年表现明显变差，说明把不完全同边界的降水表强行映射到当前空间模型，不能自动解决问题。既可能涉及降水空间分配和统计范围，也可能暴露裸地蒸发、参数范围或模型结构问题；不能据此断定观测降水表本身错误。

## 4. 全部结果与选择规则

训练仍为 2019—2022，六个共同测站；两套地类、两版账户均完整计算。每次留一年重新拟合所有自由参数，三初始点、相对平方误差目标及参数边界保持不变。仅范围修正为 5 参数，含净损失的其余方案均为 6 参数。新储量和降水项没有按径流增加可调系数。

下表为旧年度账户版本，均为 MAPE（%）。训练及 2023 使用完整有效像元官方内核复算；留一年使用经核验的分层抽样。**2023 已反复用于诊断，所有结果均为复用年。**

| 地类 | 方案 | 训练 | 留一年 | 2023 |
| --- | --- | ---: | ---: | ---: |
| 精细 | 上轮净损失模型，原范围 | 6.86 | **9.95** | 23.10 |
| 精细 | 新范围，仅四 Z＋Kc | 7.92 | 10.39 | 23.62 |
| 精细 | 新范围＋净损失 | 6.96 | 10.15 | 22.33 |
| 精细 | 新范围＋净损失＋土壤／积雪蓄变 | 8.34 | 13.32 | **6.65** |
| 精细 | 新范围＋净损失＋区域降水重建 | 11.22 | 19.02 | 21.67 |
| Copernicus | 上轮净损失模型，原范围 | 8.09 | **11.76** | 23.69 |
| Copernicus | 新范围，仅四 Z＋Kc | 8.64 | 12.26 | 23.73 |
| Copernicus | 新范围＋净损失 | 8.16 | 12.03 | 22.80 |
| Copernicus | 新范围＋净损失＋土壤／积雪蓄变 | 9.38 | 14.59 | **7.41** |
| Copernicus | 新范围＋净损失＋区域降水重建 | 9.35 | 15.56 | 22.64 |

另一账户版本、相对 RMSE、每个留年及站点结果完整保存，不按误差选账户。预设选择指标是开发年留一年相对 RMSE：精细上轮为 12.98%，范围修正＋净损失为 13.23%，新增蓄变为 17.69%，降水重建为 23.64%。因此没有把 6.65% 的复用年结果提升为正式方案。

![六站复用年对照](../project/calibration/connected_water_v1/reused_2023_comparison.png)

新增蓄变方案的精细 2023 六站模拟值依次为 211.61、324.74、193.98、184.59、295.10、308.17 亿 m³；对应贵得、兰州、头道拐、龙门、三门峡、花园口。它说明储量假设能改变年际偏差，但不是地下水／冰雪／土壤贡献已被唯一分离的证据。

## 5. 复现、性能与验证

主要文件：

- `project/diagnostics/gonghe_connectivity_v1/`：固定内流面、候选分区、面积变化、D8 样本路径和图；青海湖身份排查保存在 `guide_connectivity_v1/`，不进入率定。
- `project/diagnostics/storage_change_v1/`：原始 NetCDF、URL、变量属性、年蓄变栅格和两种范围的空间账户。
- `project/diagnostics/reference_precipitation_v1/`：区域降水系数、重建栅格、来源及适用限制。
- `project/calibration/connected_water_v1/`：全部范围／蓄变对照、逐折参数、合并指标、开发年选择、实际产物验证、PNG/PDF。初次运行的驱动代码快照与其 manifest 哈希一致，后续驱动扩展了降水选项。
- `project/calibration/reference_rainfall_v1/`：降水重建对照的完整训练和复用年结果。

范围／蓄变组两进程耗时约 **405 秒**、峰值 **1275 MiB**；降水组约 **198 秒**、峰值 **1198 MiB**。两组计算约 10 分钟，不含参考资料下载、栅格准备和本轮研究时间。均逐块读取、每工作进程单线程，监控 24 GiB／12 小时限制。官方 `fractp_op` 全像元复算不等于完整 `execute` 工作空间。

20 项相关测试通过，覆盖原有官方公式、净损失、留年隔离，加上储量空间累计／逐年隔离、蓄变符号、缺失与重复账户拒绝、D8 出界／终止／环路。实际产物另核查域外像元不变、区域降水均值、测站水量恒等式；移除像元共 172107 个，重建降水区域均值最大误差约 4.64×10⁻⁸ mm。两组抽样与全像元结果最大差异占实测 0.484%，低于 1% 阈值。51 个原诊断非代码输入指纹一致，旧冻结记录核验通过；Ruff 检查通过。

```powershell
# 已有诊断目录拒绝覆盖；重做时对支持的入口换新 --output。
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/audit_guide_connectivity.py --basin gonghe --output project/diagnostics/gonghe_repeat
# 下列准备入口仅在各自目标目录不存在时运行。
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/prepare_storage_diagnostic.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/prepare_reference_precipitation.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/calibrate_connected_water.py --output project/calibration/connected_repeat
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/calibrate_connected_water.py --reference-rainfall --output project/calibration/rainfall_repeat
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/summarize_connected_water.py --rainfall
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/verify_connected_outputs.py
```

下一步重点应是 2022—2023 储量变化与实际蒸散的一致性，以及闭流域终止约束在完整 DEM 路由中的实现。直接外加其他模型的储量、按某一年选择参数，或仅增加更多 Z，都不足以建立可靠的跨年模拟。
