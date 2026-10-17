# 数据与方法参考文献汇总

整理完成日期：2026-10-18；在线核对日期：2026-10-17—18（Asia/Hong_Kong）。

本文汇总当前工作区、实验配置、源码及清理前归档中可追溯的数据和方法来源，供论文写作及复现使用。每条给出用途与 DOI；没有查到可靠 DOI 的资料保留原始出处并明确标记，不用其他论文的 DOI 代替数据本身的 DOI。

**使用状态**：主链＝现有年尺度计算或其预处理；历史＝历史实验实际使用；诊断＝核查或方法讨论；计划＝尚未发现进入已完成计算的证据。文献支持方法思想，不等于文献验证了本项目参数或研究结论。

核查依据：[数据说明](DATA_DESCRIPTION.md)、[参数来源](M4_PARAMETER_SOURCES.md)、[共享参数库](../config/shared_biophysical.yaml)、[实验输入配置](../config/landcover_comparison.yaml)、[科学稳健性记录](SCIENTIFIC_ROBUSTNESS.md)、[严格比较方案](RIGOROUS_COMPARISON_PLAN.md)，以及[清理前源码与文档快照](../experiments/maintenance/20261012_experiment_cleanup/source_before_cleanup.zip)中的 `docs/`。已按出版方、数据提供方及 Crossref/DataCite 元数据核对可识别条目；部分网站不能直接解析全文，已在相关条目说明核对边界。

## 1. 数据、文件与引用对照

| 数据或资料 | 本地文件／位置 | 使用状态与用途 | 引用 |
| --- | --- | --- | --- |
| TerraClimate V1.1 月降水、参考蒸散 | `project/comparison_2011_2024/climate/`；复用 `project/repairs/long_climate_v1/`、`climate_reference_full/`、`climate_reference_full_holdout/` | 主链：2011—2024 成对 P/PET | D01 |
| TerraClimate 温度、AET、Q、soil、SWE | `project/diagnostics/headwater_balance_v1/`、`storage_change_v1/` 等 | 历史月试点及过程诊断；AET/Q/soil/SWE 为模型产物，不是独立实测 | D01 |
| 旧 GEE TerraClimate PET 月 TIFF | `data/Climate/TerraClimate_PET/` | 历史：2019—2023；小版本未确定，不能与 V1.1 无说明混用 | D01、S05 |
| CMFD V0200 | `data/Climate/Data_forcing_01mo_010deg/` | 历史 P、偏差校正 P、月均 PM 蒸散近似；原始文件覆盖 1951—2024 | D02、D03、M05 |
| HWSD 2.0 | `data/Soil/hwsd.bil`、`HWSD2.mdb` | 主链：分层土壤容量及限制层深度代理 | D04 |
| Copernicus 2019 土地覆盖 | `data/LCLU/Copernicus_Landcover_100m.tif` 及类别表 | 主链：粗产品对照；本地为重投影文件 | D05、D06 |
| ESA WorldCover 2021 v200 | `data/LCLU/ESA_WorldCover_v200.tif` 及类别表 | 主链：另一粗产品对照；本地栅格不是原生 10 m | D07 |
| ALOS DSM | `data/DEM/ALOS_DSM_100m.tif` | 主链：由 30 m 重采样的高程数据；原产品具体版本待补 | D08 |
| HydroRIVERS v1.0 Asia 黄河裁剪 | `data/Hydrology/HydroRIVERS_v10_as_shp/` | 主链：干支流拓扑、测站匹配及水文约束 | D09 |
| HydroBASINS Asia level 06 v1c 标准版 | `project/references/hydrobasins_as06/` | 诊断及主链修复：内流面识别、闭流路由修正 | D09 |
| 自制精细分类及大类合并图 | `data/LCLU/vegetation_model_20261010_100411ImperviousIce.tif` 及同源 CSV | 主链：用户确认代表 2023 年；大类图是同源派生结果 | U01 |
| 黄河流域范围、湖泊矢量 | `data/Hydrology/黄河流域.shp`、`黄河流域湖泊.shp` | 主链：研究范围参考、共用水体掩膜；原始发布来源未确定 | U02 |
| 实测径流、耗水、库容变化、统计降水、面积 | `data/Hydrology/*2011-2024.csv`、`测站控制面积.csv`、`水资源二级区面积.csv` | 主链及核查，逐项用途见 U03；不把所有水量相加或重复扣减 | U03 |
| 2022、2023、2024 年黄河水资源公报 | `data/Hydrology/2022年黄河水资源公报.pdf` 等 | 已用于水量账户、单位与本地耗水／外调水拆分核查 | U04 |
| MOD16A2GF Collection 6.1 | `scripts/export_mod16a2gf_gee.js`；`data/Climate/MOD16A2GF_061/` | 计划：已有导出脚本，不能写成已取得并参与率定的数据 | P01、P02 |
| CMIP6／ScenarioMIP | 项目研究目标；尚无可追溯的具体模式数据清单 | 计划：仅列框架文献，不能替代未来具体 GCM 数据引用 | P03、P04 |
| 土壤保持措施 P 与植被覆盖管理 C | `data/LCLU/P_300.tif`、`nature_table_2_cropland_c_factor.csv`、`nature_table_3_non_cropland_c_factor.csv` | 新增、尚未实际使用；拟用于土壤侵蚀／保持分析 | U05 |
| 降雨侵蚀力 R | `data/Soil/R-factor1991-2020.tif` | 新增、尚未实际使用；1991—2020 气候态多年平均 R 因子 | U06 |
| 土壤可侵蚀性 K | `data/Soil/epic-k-factor.tif` | 新增、尚未实际使用；全球 EPIC K 因子栅格 | U07 |
| 输沙量 | `data/Hydrology/输沙量2011-2024.csv` | 现有文件；尚未发现进入年产水计算的代码引用，来源待补 | U08 |

目前的精细、Copernicus、WorldCover 分别代表不同年份，固定用于多年模拟；文献引用不能把这些静态图变成逐年历史土地覆盖。

## 2. 数据集及产品文献

### D01 — TerraClimate

Abatzoglou, J. T., Dobrowski, S. Z., Parks, S. A., & Hegewisch, K. C. (2018). **TerraClimate, a high-resolution global dataset of monthly climate and climatic water balance from 1958–2015**. *Scientific Data*, 5, 170191.

- DOI：[10.1038/sdata.2017.191](https://doi.org/10.1038/sdata.2017.191)。类型：数据论文。
- 项目用途：历史与当前气候强迫、过程诊断的共同来源。
- **版本补充**：现有主链使用 V1.1，须同时引用[提供方 V1.1 说明](https://www.climatologylab.org/terraclimate.html)。V1.1 的时变输入改用 ERA5；2018 年论文不能独自说明新版全部改动。未查到可单独确认的 V1.1 数据 DOI。
- 原始下载 URL、变量属性、时间范围与 SHA256 见各下载目录 JSON；论文的 1958—2015 是原论文数据范围，不是项目实际使用时期。未直接下载的 ERA5、WorldClim 不列为项目独立输入。

### D02 — CMFD 2.0 数据集

何杰，阳坤，李新，唐文君，邵长坤，姜尧志，丁宝弘（2024）．**中国区域地面气象要素驱动数据集 v2.0（1951—2020）**．国家青藏高原数据中心。

- DOI：[10.11888/Atmos.tpdc.302088](https://doi.org/10.11888/Atmos.tpdc.302088)。类型：数据集。
- 核对来源：[联合研制方清华大学发布说明](https://www.dess.tsinghua.edu.cn/info/1108/6505.htm)。这里保留发布说明中的题名与年份；本地 V0200 文件延伸至 2024 年，不能把初始发布的结束年份冒充本地覆盖范围。
- 使用状态：历史／候选气候及蒸散试验，不是当前 TerraClimate 成对强迫。具体更新批次仍应以本地文件元数据及下载记录为准。

### D03 — CMFD 基础方法论文

He, J., Yang, K., Tang, W., Lu, H., Qin, J., Chen, Y., & Li, X. (2020). **The first high-resolution meteorological forcing dataset for land process studies over China**. *Scientific Data*, 7, 25.

- DOI：[10.1038/s41597-020-0369-y](https://doi.org/10.1038/s41597-020-0369-y)。
- 用途：CMFD 多源气象融合方法背景；与 D02 配套引用，不把这篇早期论文写成 V2.0 全部更新的专门说明。

### D04 — HWSD 2.0

FAO & IIASA (2023). **Harmonized World Soil Database version 2.0**. Rome and Laxenburg.

- DOI：[10.4060/cc3823en](https://doi.org/10.4060/cc3823en)。类型：数据库技术报告；[IIASA 原文](https://pure.iiasa.ac.at/id/eprint/18595/1/cc3823en.pdf)。
- 用途：`src/ecologyhydro/soil.py` 的分层质地、粗颗粒、矿物与盐分修正和容量积分；源码明确引用报告表 2.2／2.3。
- 限制：`1500/750/300/50 mm` 限制层等级代理是项目转换方案，不是报告提供的逐像元实测根深；HWSD 也不是植被根深资料。

### D05 — Copernicus 2019 土地覆盖数据

Buchhorn, M., Smets, B., Bertels, L., De Roo, B., Lesiv, M., Tsendbazar, N.-E., Herold, M., & Fritz, S. (2020). **Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019: Globe** (V3.0.1). Zenodo.

- DOI：[10.5281/zenodo.3939050](https://doi.org/10.5281/zenodo.3939050)。类型：2019 年产品数据集；[记录页](https://zenodo.org/records/3939050)。
- 用途：Copernicus 粗产品对照的原产品引用。项目已确认 2019 年，原始下载记录未完整保留；V3.0.1 是该引用记录的版本，不能仅凭本地重命名 TIFF 宣称已逐字节确认其下载批次。

### D06 — Copernicus 类别定义与产品手册

Buchhorn, M., Smets, B., Bertels, L., De Roo, B., Lesiv, M., Tsendbazar, N.-E., Li, L., & Tarko, A. (2021). **Copernicus Global Land Service: Land Cover 100m: version 3 Globe 2015–2019: Product User Manual** (dataset v3.0, document issue 3.4). Zenodo.

- DOI：[10.5281/zenodo.4723921](https://doi.org/10.5281/zenodo.4723921)。类型：产品技术文档。
- 用途：`shared_biophysical.yaml` 的粗产品分类语义、裸地与稀疏植被、森林子类解释。论文 *Copernicus Global Land Cover Layers—Collection 2* 讲的是 Collection 2，不拿其 DOI 替代 Collection 3 数据版本引用。

### D07 — ESA WorldCover 2021 v200

Zanaga, D., Van De Kerchove, R., Daems, D., et al. (2022). **ESA WorldCover 10 m 2021 v200**. Zenodo.

- DOI：[10.5281/zenodo.7254221](https://doi.org/10.5281/zenodo.7254221)。类型：数据集。
- 配套：[Product User Manual V2.0](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf)；手册未查到独立 DOI。
- 用途：WorldCover 产品对照、类别到共享生物物理参数的映射。2021 是制图年，2022 是数据发布年；原生 10 m 不等于本地处理后分辨率。

### D08 — ALOS／AW3D30 地表高程

Tadono, T., Nagai, H., Ishida, H., Oda, F., Naito, S., Minakawa, K., & Iwamoto, H. (2016). **Generation of the 30 m-mesh global digital surface model by ALOS PRISM**. *The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences*, XLI-B4, 157–162.

- DOI：[10.5194/isprs-archives-XLI-B4-157-2016](https://doi.org/10.5194/isprs-archives-XLI-B4-157-2016)。类型：产品方法论文；题名按[出版方记录](https://isprs-archives.copernicus.org/articles/XLI-B4/157/2016/)核对。
- 用途：本地 ALOS 30 m 来源、100 m 派生 DSM 的产品家族参考；[JAXA 产品页](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm)。原始版本和瓦片清单尚不完整，不能据此断言本地数据就是某个 AW3D30 小版本。DSM 不等同于裸地 DTM。

### D09 — HydroRIVERS 与 HydroBASINS

Lehner, B., & Grill, G. (2013). **Global river hydrography and network routing: baseline data and new approaches to study the world's large river systems**. *Hydrological Processes*, 27(15), 2171–2186.

- DOI：[10.1002/hyp.9740](https://doi.org/10.1002/hyp.9740)。类型：两个产品官方共同推荐的方法论文；不是本地裁剪文件的独立数据 DOI。
- 提供方：[HydroRIVERS](https://www.hydrosheds.org/products/hydrorivers)、[HydroBASINS](https://www.hydrosheds.org/products/hydrobasins)。
- 字段与版本说明：[HydroRIVERS v1.0 技术文档](https://data.hydrosheds.org/file/technical-documentation/HydroRIVERS_TechDoc_v10.pdf)、[HydroBASINS v1c 技术文档](https://data.hydrosheds.org/file/technical-documentation/HydroBASINS_TechDoc_v1c.pdf)。
- 用途：`NEXT_DOWN`／`ENDORHEIC` 河网拓扑和内流盆地边界。二者共享 HydroSHEDS 来源，不能当作互相独立的实测验证；本地湖泊 shp 没有证据属于 HydroLAKES，不为其套用 HydroLAKES 引用。

## 3. 水文模型、参数与空间处理方法

### M01 — InVEST Annual Water Yield

Natural Capital Alliance (2026). **InVEST 3.20.2**.

- DOI：[10.60793/natcap-invest-3.20.2](https://doi.org/10.60793/natcap-invest-3.20.2)。类型：官方版本引用。
- 核对依据：[官方用户指南首页推荐引用](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/index.html)；[Annual Water Yield 方法说明](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/annual_water_yield.html)。本次 DOI 解析接口未成功返回记录，但官方首页明确给出这一版本与 DOI，故按官方推荐保留。
- 本地 `pyproject.toml` 锁定 `natcap.invest==3.20.2`。旧实验文档还引用过 3.17.0／3.19.0 指南，不意味着实际软件版本随链接改变。
- 用途：年度 `Y=P−AET`、植被 Fu/Budyko 分支、非植被蒸散上限、土壤有效储水和 Z 参数。测站径流还受耗水、蓄变、输移及交换影响；项目附加路由和净损失不属于官方 AWY 原生过程。

### M02 — Fu 公式原始出处

傅抱璞（1981）．**论陆面蒸发的计算**．《大气科学》，5(1)。英文常引为：Fu, B. P. *On the calculation of the evaporation from land surface*.

- DOI：**尚未完成原刊核验**。检索发现候选号 `10.3878/j.issn.1006-9895.1981.01.03`，但原刊及解析接口本次未取得可核对记录；不计入本文已核实 DOI。
- 用途：AWY 指南明确追溯的公式来源。页码在二次引文中有差异，本表暂不填；可同时引用已核实的 M03 和 M01 说明实际计算式。

### M03 — Fu/Budyko 年蒸散函数

Zhang, L., Hickel, K., Dawes, W. R., Chiew, F. H. S., Western, A. W., & Briggs, P. R. (2004). **A rational function approach for estimating mean annual evapotranspiration**. *Water Resources Research*, 40, W02502.

- DOI：[10.1029/2003WR002710](https://doi.org/10.1029/2003WR002710)。
- 用途：年度蒸散与降水、潜在蒸散及流域特征之间的函数关系；主链和独立公式复核的理论来源。

### M04 — 根深、土壤储水与 Budyko 参数化

Donohue, R. J., Roderick, M. L., & McVicar, T. R. (2012). **Roots, storms and soil pores: Incorporating key ecohydrological processes into Budyko's hydrological model**. *Journal of Hydrology*, 436–437, 35–50.

- DOI：[10.1016/j.jhydrol.2012.02.033](https://doi.org/10.1016/j.jhydrol.2012.02.033)。
- 用途：AWY 土壤有效储水、有效根深和季节性参数化依据；本项目实际 `omega` 表达式及截断规则以 M01 和对应软件版本为准。

### M05 — FAO-56 蒸散、作物系数与有效根深

Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). **Crop evapotranspiration: Guidelines for computing crop water requirements**. FAO Irrigation and Drainage Paper 56. Rome: FAO. ISBN 92-5-104219-5.

- DOI：**未在本次核对的 FAO 原版中找到可靠 DOI**；[FAO 原文](https://www.fao.org/4/X0490E/X0490E00.htm)。
- 具体使用：[第 6 章表 12](https://www.fao.org/4/X0490E/x0490e0b.htm)的阶段 Kc／水面系数；[第 7 章](https://www.fao.org/4/X0490E/x0490e0c.htm)的双作物系数与土壤蒸发；[第 8 章表 22](https://www.fao.org/4/x0490e/x0490e0e.htm)的有效根深。
- 项目中的年度 Kc、月 Kc 插值、统一轮作历和 1400 mm 谷物根深均含迁移假设；不是 FAO 给出的黄河流域实测年度参数。历史月均气象 PM 近似也不能直接称为完整标准日尺度 FAO-56 计算。

### M06 — 全球根系垂直分布

Jackson, R. B., Canadell, J., Ehleringer, J. R., Mooney, H. A., Sala, O. E., & Schulze, E. D. (1996). **A global analysis of root distributions for terrestrial biomes**. *Oecologia*, 108, 389–411.

- DOI：[10.1007/BF00333714](https://doi.org/10.1007/BF00333714)；[作者原文](https://jacksonlab.stanford.edu/sites/g/files/sbiybj20871/files/media/file/oecol96c.pdf)。
- 用途：`shared_biophysical.yaml` 中 beta 分布到根生物量 d95 的换算：`d95_mm=10×ln(0.05)/ln(beta)`。d95、FAO 有效根深和土壤限制层深度是不同概念。

### M07 — 青藏高原草地根系分配

Yang, Y., Fang, J., Ji, C., & Han, W. (2009). **Above- and belowground biomass allocation in Tibetan grasslands**. *Journal of Vegetation Science*, 20, 177–184.

- DOI：[10.1111/j.1654-1103.2009.05566.x](https://doi.org/10.1111/j.1654-1103.2009.05566.x)。
- 用途：高寒草甸浅层根生物量分配依据。当前共享库未沿用早期用草原单个累计比例外推 d95 的做法；不可把 30 cm 内根生物量占比写成最大根深。

### M08 — 冬小麦—夏玉米轮作日历

Sun, H., Zhang, X., Chen, S., Pei, D., & Liu, C. (2007). **Effects of harvest and sowing time on the performance of the rotation of winter wheat–summer maize in the North China Plain**. *Industrial Crops and Products*, 25, 239–247.

- DOI：[10.1016/j.indcrop.2006.12.003](https://doi.org/10.1016/j.indcrop.2006.12.003)。已通过出版方提交的 Crossref 元数据核对原 PII `S0926669006001580`。
- 用途：`config/crop_systems.yaml` 的历史轮作日历参考。只支持区域制度背景，不证明所有水浇地像元具有相同播种日期或月 Kc。

### M09 — 华北平原长期轮作田间研究

Wang, N., Ai, Z., Zhang, Q., et al. (2025). **The variations of wheat–maize production, soil organic carbon, and carbon footprints: insights from a 20-year on-farm observational experiment in the North China Plain**. *Frontiers in Plant Science*, 16, 1547431.

- DOI：[10.3389/fpls.2025.1547431](https://doi.org/10.3389/fpls.2025.1547431)；[原文记录](https://pmc.ncbi.nlm.nih.gov/articles/PMC12066458/)。
- 用途：已登记的轮作背景文献；不是项目全部耕地的本地调查，也不是月 Kc 参数表的直接来源。

### M10 — D8 流向与河网提取

O'Callaghan, J. F., & Mark, D. M. (1984). **The extraction of drainage networks from digital elevation data**. *Computer Vision, Graphics, and Image Processing*, 28, 323–344.

- DOI：[10.1016/S0734-189X(84)80011-0](https://doi.org/10.1016/S0734-189X%2884%2980011-0)。元数据已由 Crossref 核对。
- 用途：实际使用的 D8 单流向方法基础，执行层见 `watersheds.py`、`conditioning.py` 与 PyGeoprocessing。
- 项目降低河道 5 m、出口搜索半径、内流边终止及嵌套区域差分属于工程实现和边界约定，不是这篇论文给出的黄河专用参数。

### M11 — InVEST 年水量平衡不确定性及勘误

Hamel, P., & Guswa, A. J. (2015). **Uncertainty analysis of a spatially explicit annual water-balance model: case study of the Cape Fear basin, North Carolina**. *Hydrology and Earth System Sciences*, 19, 839–853.

- 原文 DOI：[10.5194/hess-19-839-2015](https://doi.org/10.5194/hess-19-839-2015)。
- 同作者同年勘误，*HESS*, 19, 855：DOI [10.5194/hess-19-855-2015](https://doi.org/10.5194/hess-19-855-2015)。引用相关公式和参数解释时须同时检查勘误。
- 用途：`RIGOROUS_COMPARISON_PLAN.md` 中气候、Z 参数与模型结构不确定性的依据。

### M12 — 年际储水与径流记忆

Berghuijs, W. R., Woods, R. A., Anderson, B. J., Hemshorn de Sánchez, A. L., & Hrachowitz, M. (2025). **Annual memory in the terrestrial water cycle**. *Hydrology and Earth System Sciences*, 29, 1319–1333.

- DOI：[10.5194/hess-29-1319-2025](https://doi.org/10.5194/hess-29-1319-2025)。
- 用途：年际储量与记忆诊断的研究依据。该文指出简单线性排水不能重现其分析的全部记忆信号，不能据此宣称本项目线性储留系数已经得到物理验证。

### M13 — 线性水库方法参考

U.S. Army Corps of Engineers, Hydrologic Engineering Center. **HEC-HMS Technical Reference Manual: Linear Reservoir Model**.

- DOI：**网页未提供独立 DOI**；[官方技术说明](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/baseflow/linear-reservoir-model)。
- 用途：历史年度滞蓄试验的线性水库概念参考。项目递推式、年时间步、初始状态和参数边界为自身设计；没有运行 HEC-HMS，也没有移植其完整基流模型。

### M14 — 月尺度概念模型框架

Andrews, F. T., Croke, B. F. W., & Jakeman, A. J. (2011). **An open software environment for hydrological model assessment and development**. *Environmental Modelling & Software*, 26, 1171–1185.

- DOI：[10.1016/j.envsoft.2011.04.006](https://doi.org/10.1016/j.envsoft.2011.04.006)；[hydromad 教程](https://hydromad.github.io/articles/tutorial.html)。
- 用途：归档 `MONTHLY_PILOT_RESULTS.md` 引用的土壤水量核算与径流路由分离思想；项目没有调用 hydromad，也没有复现其某个完整模型。
- `monthly_pilot.py` 的雨雪阈值 −1～1 ℃、融雪系数 3 mm/(℃·d)、土壤桶初值及蒸散结构是项目假设，未发现逐项外部参数标定来源。

## 4. 率定、评价、稳定性及流域解释

### M15 — 分布式率定与正则化

Jay-Allemand, M., Javelle, P., Gejadze, I., Arnaud, P., Malaterre, P.-O., Fine, J.-A., & Organde, D. (2020). **On the potential of variational calibration for a fully distributed hydrological model: application on a Mediterranean catchment**. *Hydrology and Earth System Sciences*, 24, 5519–5538.

- DOI：[10.5194/hess-24-5519-2020](https://doi.org/10.5194/hess-24-5519-2020)。
- 用途：空间参数自由度、等效性和正则化讨论。项目 ridge 公式、尺度、候选强度是项目设计；最终选中零惩罚的版本不能写成正则化改善了精度。

### M16 — 子时期率定与时间一致性

Gharari, S., Hrachowitz, M., Fenicia, F., & Savenije, H. H. G. (2013). **An approach to identify time consistent model parameters: sub-period calibration**. *Hydrology and Earth System Sciences*, 17, 149–161.

- DOI：[10.5194/hess-17-149-2013](https://doi.org/10.5194/hess-17-149-2013)。
- 用途：历史子时期检验、参数时间稳定性；本项目的分期、折划分和选优规则另见配置，不声称原样复现论文算法。

### M17 — 空间共同率定

Bárdossy, A., Huang, Y., & Wagener, T. (2016). **Simultaneous calibration of hydrological models in geographical space**. *Hydrology and Earth System Sciences*, 20, 2913–2928.

- DOI：[10.5194/hess-20-2913-2016](https://doi.org/10.5194/hess-20-2913-2016)。
- 用途：`config/scientific_robustness.yaml` 已登记的空间参数共享、迁移方法背景。

### M18 — 空间参数化与多测站率定

Zheng, X., Liu, D., Wang, H., et al. (2026). **What can hydrological modelling gain from spatially explicit parameterization and multi-gauge calibration?** *Hydrology and Earth System Sciences*, 30, 2493–2521.

- DOI：[10.5194/hess-30-2493-2026](https://doi.org/10.5194/hess-30-2493-2026)。
- 用途：归档 `INDEPENDENT_LOCAL_BALANCE.md` 的方法参考；原研究采用 VIC，不能把其算法或性能直接归给本项目 AWY 扩展。

### M19 — 模型选择偏差

Cawley, G. C., & Talbot, N. L. C. (2010). **On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation**. *Journal of Machine Learning Research*, 11(70), 2079–2107.

- DOI：**期刊原页未提供，未查到可靠注册 DOI**；[JMLR 原文](https://jmlr.org/papers/v11/cawley10a.html)。
- 用途：选型与最终评价分离、避免验证年泄漏。不能把文献网页标识或数据库 accession number 写成 DOI。

### M20 — 扰动训练集与组合预测

Breiman, L. (1996). **Bagging predictors**. *Machine Learning*, 24, 123–140.

- DOI：[10.1007/BF00058655](https://doi.org/10.1007/BF00058655)。
- 用途：归档稳定性实验的组合预测思想。项目采用确定性删去一个训练年，不是原论文的有放回 bootstrap bagging；不能直接声称生成了 bootstrap 置信区间。

### M21 — Nash–Sutcliffe 效率

Nash, J. E., & Sutcliffe, J. V. (1970). **River flow forecasting through conceptual models part I — A discussion of principles**. *Journal of Hydrology*, 10, 282–290.

- DOI：[10.1016/0022-1694(70)90255-6](https://doi.org/10.1016/0022-1694%2870%2990255-6)。
- 用途：已经使用的 NSE 指标的原始方法出处。逐站时间序列 NSE 与跨站混合 NSE 不可混称。

### M22 — 水文模型评价准则（2007）

Moriasi, D. N., Arnold, J. G., Van Liew, M. W., Bingner, R. L., Harmel, R. D., & Veith, T. L. (2007). **Model Evaluation Guidelines for Systematic Quantification of Accuracy in Watershed Simulations**. *Transactions of the ASABE*, 50(3), 885–900.

- DOI：[10.13031/2013.23153](https://doi.org/10.13031/2013.23153)；[USDA 原文](https://www.ars.usda.gov/ARSUserFiles/30980000/graphics/MoriasiModelEval.pdf)。
- 用途：NSE、RSR、PBIAS 与常引用的月尺度评价准则背景，不是本项目或期刊统一验收线。

### M23 — 水文模型评价准则（2015）

Moriasi, D. N., Gitau, M. W., Pai, N., & Daggupati, P. (2015). **Hydrologic and Water Quality Models: Performance Measures and Evaluation Criteria**. *Transactions of the ASABE*, 58(6), 1763–1785.

- DOI：[10.13031/trans.58.10715](https://doi.org/10.13031/trans.58.10715)；[作者公开论文](https://web.ics.purdue.edu/~mgitau/pdf/Moriasi%20et%20al%202015.pdf)。
- 用途：`DATA_USAGE_AND_RESEARCH_ASSESSMENT.md` 的逐站科研指标核查。相对 RMSE、MAPE、PBIAS 是不同指标，不共用门槛；少量年份不能支持强稳健性结论。

### M24 — 率定目标与用途相匹配

Vis, M., Knight, R., Pool, S., Wolfe, W., & Seibert, J. (2015). **Model Calibration Criteria for Estimating Ecological Flow Characteristics**. *Water*, 7(5), 2358–2381.

- DOI：[10.3390/w7052358](https://doi.org/10.3390/w7052358)。
- 用途：归档 `FINE_MAP_PAIRWISE_EXPERIMENT.md` 的目标函数背景。USGS 收录页日期不替代原论文的 2015 年发表年份；项目的相对误差权重和最优站年计数规则仍为自身定义。

### M25 — 黄河干流自然补损

李春晖，杨志峰（2003）．**黄河干流河道水量自然补损分析**．《地理学报》，58(4)，527–533。

- DOI：[10.11821/xb200304006](https://doi.org/10.11821/xb200304006)；[期刊原页](https://www.geog.com.cn/CN/abstract/article/0375-5444/19027)。
- 用途：归档干旱区连通性诊断中单列河道净补损的背景。没有把论文历史多年净损失量作为本项目有效损失 L 的数值来源。

### M26 — 黄河流域范围与分区口径

左其亭，吴青松，姜龙，张伟（2022）．**黄河流域多尺度区域界定及其应用选择**．《水利水运工程学报》，(5)，12–20。

- DOI：[10.12170/20210923001](https://doi.org/10.12170/20210923001)；[期刊原文](https://slsy.nhri.cn/cn/article/pdf/preview/10.12170/20210923001.pdf)。
- 用途：长序列核查、上中下游及研究区域口径说明；不是本地边界 shp 的数据出处。

### M27 — 水资源分区与土地利用背景

张冉，王义民，畅建霞，黎云云（2019）．**基于水资源分区的黄河流域土地利用变化对人类活动的响应**．《自然资源学报》，34(2)，274–287。

- DOI：[10.31497/zrzyxb.20190205](https://doi.org/10.31497/zrzyxb.20190205)；[期刊原页](https://www.jnr.ac.cn/CN/abstract/article/1000-3037/43066)。
- 用途：归档 `LONG_RECORD_VALIDATION.md` 的八个水资源二级区背景；不能替代缺失的官方分区矢量。

### M28 — 共和盆地水系演化

Wang, S., Zheng, D., Pang, J., Yang, H., Wu, Y., & Ma, Y. (2026). **Drainage system evolution in Gonghe Basin, Northwest Tibetan Plateau: A story of tectonic geomorphology**. *Geomorphology*, 504, 110310.

- DOI：[10.1016/j.geomorph.2026.110310](https://doi.org/10.1016/j.geomorph.2026.110310)。已通过出版方提交的 Crossref 元数据核对原 PII `S0169555X26001534`。
- 用途：`audit_guide_connectivity.py` 和归档共和盆地连通性诊断的地貌背景；不据古水系演化断言当前地表或地下水连接，也不移植参数。

### M29 — 拉丁超立方采样（LHS）

McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). **Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code**. *Technometrics*, 21(2), 239–245.

- DOI：[10.1080/00401706.1979.10489755](https://doi.org/10.1080/00401706.1979.10489755)。元数据已由 Crossref 核对。
- 用途：为已有数据使用核查报告中记录的历史月模型约束试验补列 LHS 方法出处。报告记载使用过 8192 个内部 LHS 样本；相关专用程序随后归档，不能把它写成当前主链采样算法。
- 样本预算和边界最优解追加是项目设置；有限采样未找到其他可行解，不证明连续参数空间只有唯一解。

## 5. 计算软件与实现参考

实际环境版本以 [pyproject.toml](../pyproject.toml)、[环境配置](../environment.yml) 和 `locks/conda-win-64.lock` 为准。软件论文 DOI 与某一安装版本的 DOI 不一定相同。

| 编号 | 文献／软件 | DOI 或官方出处 | 项目用途 |
| --- | --- | --- | --- |
| S01 | Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). **SciPy 1.0: fundamental algorithms for scientific computing in Python**. *Nature Methods*, 17, 261–272. | [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2) | `least_squares` 有界最小二乘、滤波和插值；具体[优化器文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) |
| S02 | Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). **Array programming with NumPy**. *Nature*, 585, 357–362. | [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2) | 分块数组、向量化公式、面积和水量聚合 |
| S03 | GDAL/OGR contributors. **GDAL/OGR Geospatial Data Abstraction software Library**. Open Source Geospatial Foundation. | [10.5281/zenodo.5884351](https://doi.org/10.5281/zenodo.5884351)；[官方 CITATION](https://github.com/OSGeo/gdal/blob/master/CITATION) | 栅格／矢量读写、重投影、最近邻／双线性重采样；这是软件通用引用，具体 3.10 系列版本见锁文件 |
| S04 | McKinney, W. (2010). **Data Structures for Statistical Computing in Python**. *Proceedings of the 9th Python in Science Conference*, 56–61. | [10.25080/Majora-92bf1922-00a](https://doi.org/10.25080/Majora-92bf1922-00a) | pandas 水文表整理、配对比较和结果汇总 |
| S05 | Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R. (2017). **Google Earth Engine: Planetary-scale geospatial analysis for everyone**. *Remote Sensing of Environment*, 202, 18–27. | [10.1016/j.rse.2017.06.031](https://doi.org/10.1016/j.rse.2017.06.031) | 历史 TerraClimate TIFF 的导出平台背景及 MOD16 导出脚本；不是本地水文计算引擎 |
| S06 | Natural Capital Project. **PyGeoprocessing**；**TaskGraph**. | 未核实独立 DOI；[PyGeoprocessing](https://github.com/natcap/pygeoprocessing)、[TaskGraph](https://github.com/natcap/taskgraph) | 水文栅格处理、InVEST 依赖和任务执行；不套用其他 GIS 算法的 DOI |
| S07 | scikit-learn 开发者．**TimeSeriesSplit** 与 **Nested versus non-nested cross-validation** 文档． | 文档未提供独立 DOI；[时间顺序分割](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)、[嵌套交叉验证](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) | 旧文档引用的设计说明；不能由这两个链接推断项目实际调用了 scikit-learn |

分块缓存、受控多进程、空间抽样后完整像元复核、解析雅可比，以及具体随机种子和采样预算均为项目实现；未找到各自需要冒名套用的专门外部论文。通用 Python／netCDF4／YAML 等运行依赖由环境文件记录，不把每个依赖包当作独立水文方法。

## 6. 已引用或计划中的后续数据与方法

本节与“已使用数据”分开。它们可以用于研究设计或讨论，只有将来确实执行相应实验后，才能写入该实验的数据与方法部分。

### P01 — MOD16A2GF Collection 6.1

Running, S., Mu, Q., Zhao, M., & Moreno, A. (2021). **MODIS/Terra Net Evapotranspiration Gap-Filled 8-Day L4 Global 500m SIN Grid V061**. NASA EOSDIS Land Processes DAAC.

- DOI：[10.5067/MODIS/MOD16A2GF.061](https://doi.org/10.5067/MODIS/MOD16A2GF.061)。类型：数据集；[NASA 数据记录](https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod16a2gf-061)、[GEE 目录](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD16A2GF)。
- 已登记算法说明：[MOD16 ATBD](https://lpdaac.usgs.gov/documents/93/MOD16_ATBD.pdf)，未核实其独立 DOI。数据 DOI 不冒充算法文档 DOI。
- 状态：现有导出脚本和研究计划；没有已使用 MOD16 观测约束模型的证据。

### P02 — ET／土壤水分辅助率定

López López, P., Sutanudjaja, E. H., Schellekens, J., Sterk, G., & Bierkens, M. F. P. (2017). **Calibration of a large-scale hydrological model using satellite-based soil moisture and evapotranspiration products**. *Hydrology and Earth System Sciences*, 21, 3125–3144.

- DOI：[10.5194/hess-21-3125-2017](https://doi.org/10.5194/hess-21-3125-2017)。
- 用途：严格比较方案中独立过程约束的设计背景，不代表本项目已经完成多目标遥感率定。

### P03 — CMIP6 总体实验框架

Eyring, V., Bony, S., Meehl, G. A., et al. (2016). **Overview of the Coupled Model Intercomparison Project Phase 6 (CMIP6) experimental design and organization**. *Geoscientific Model Development*, 9, 1937–1958.

- DOI：[10.5194/gmd-9-1937-2016](https://doi.org/10.5194/gmd-9-1937-2016)。
- 状态：为项目第二项研究目标补列的框架文献；当前未识别具体 GCM、成员、网格、变量和下载版本。它不是某个 CMIP6 数据集的 DOI。

### P04 — ScenarioMIP 情景框架

O'Neill, B. C., Tebaldi, C., van Vuuren, D. P., et al. (2016). **The Scenario Model Intercomparison Project (ScenarioMIP) for CMIP6**. *Geoscientific Model Development*, 9, 3461–3482.

- DOI：[10.5194/gmd-9-3461-2016](https://doi.org/10.5194/gmd-9-3461-2016)。
- 状态：未来情景框架补充。正式模拟时还需逐模式数据 DOI、SSP、成员编号、时间段及实际采用的偏差订正／降尺度方法来源；目前不虚列某种订正方法为已用方法。

## 7. 无 DOI、原始出处待补或仅作背景的资料

### U01 — 自制精细地表分类及其大类

- 资料：`vegetation_model_20261010_100411ImperviousIce.tif`、`vegetation_model_20261010_100411.csv`，用户确认代表 2023 年；文件名时间不是制图年份。
- DOI：**未提供／未发现公开发布记录**。
- 待补：作者或制作机构、正式题名、制作方法、数据发布日期、版本、验证资料以及公开论文／仓储 DOI（若有）。大类图引用同一原始精细数据，并单独说明项目聚合规则，不伪造另一个数据来源。

### U02 — 研究范围及湖泊矢量

- 资料：`黄河流域.shp`、`黄河流域湖泊.shp` 及配套文件。
- DOI：**来源未确定，无法分配**。
- 已查 XML：有 ArcGIS 编辑／坐标处理记录，没有足以确定原始发布者和产品版本的引文。不能仅凭文件名或字段结构认定为某个公开湖泊数据库。

### U03 — 实测水文、管理账户与面积表

- 主链：`实测年径流量2011-2024.csv`、`地表水耗水2011-2024.csv`、`大中型水库年蓄水变化量2011-2024.csv`。
- 核查／情景：`地下水耗水2011-2024.csv`、`降水量2011-2024.csv`、`测站控制面积.csv`、`水资源二级区面积.csv`；历史已用的旧长表及行业表按归档记录引用，不宣称仍在当前原始目录。
- DOI：**本地整理表未提供公开 DOI**。应保留“用户提供并整理”以及原始公报／年鉴名、年份、表号、页码、统计口径和整理人。
- 已核实的 2022—2024 地表水账户可追溯至 U04；其他年份、变量不能据此一概宣称已逐页核对。实测径流是受调控流量，不自动等于天然径流；耗水不等于取水／供水，库容变化也不是全部陆地储量变化。

### U04 — 黄河水资源公报

水利部黄河水利委员会．**2022 年黄河水资源公报**；**2023 年黄河水资源公报**；**2024 年黄河水资源公报**［年度公报］。

- 本地原件：[2022 年](../data/Hydrology/2022年黄河水资源公报.pdf)、[2023 年](../data/Hydrology/2023年黄河水资源公报.pdf)、[2024 年](../data/Hydrology/2024年黄河水资源公报.pdf)。题名年份是统计年；正式书目中的出版年份应查封面或版权页，本文不推定。
- DOI：**未发现可确认的公开 DOI**。年度政府公报可以正常引用，无须为了补 DOI 改引一篇无关论文。
- 本地可追溯位置：`scripts/audit_bulletin_accounts.py` 和 `project/references/water_bulletins/surface_account_split.csv`。2022 年为 PDF 第 28 页／印刷第 20 页表 11；2023 年为第 29／21 页表 11；2024 年为第 30／24 页表 10。页号按现有转录脚本记录，本次未重新逐页转录。

### U05 — 中国土壤保持数据集的 P、C 因子资料

Li, J., He, H., Zeng, Q., Chen, L., & Sun, R. (2023). **A Chinese soil conservation dataset preventing soil water erosion from 1992 to 2019**. *Scientific Data*, 10, 319.

- DOI：[10.1038/s41597-023-02246-4](https://doi.org/10.1038/s41597-023-02246-4)；[出版方页面](https://www.nature.com/articles/s41597-023-02246-4)。
- 对应本地资料：`data/LCLU/P_300.tif`、`data/LCLU/nature_table_2_cropland_c_factor.csv`、`data/LCLU/nature_table_3_non_cropland_c_factor.csv`。
- 状态：**新增、尚未实际使用**。论文描述的是基于 RUSLE 建立的 1992—2019 年、300 m 中国土壤保持数据集，其中包括覆盖管理 C 因子和水土保持措施 P 因子；当前三个本地文件的精确年份、表格抽取方式和是否经过再处理，仍应以获取记录和文件元数据为准。
- 引用边界：该 DOI 是数据论文 DOI。若本地文件来自论文配套数据仓储，还应补记仓储记录、下载 URL、版本和文件校验值；不能仅凭论文引用推断本地文件与原始发布文件逐字节一致。

### U06 — 中国大陆 1991—2020 降雨侵蚀力 R 因子

Yue, T., Yin, S., Xie, Y., Yu, B., & Liu, B. (2022). **Rainfall erosivity mapping over mainland China based on high-density hourly rainfall records**. *Earth System Science Data*, 14, 665–682.

- 数据集 DOI：[10.12275/bnu.clicia.rainfallerosivity.CN.001](https://doi.org/10.12275/bnu.clicia.rainfallerosivity.CN.001)。
- 数据论文 DOI：[10.5194/essd-14-665-2022](https://doi.org/10.5194/essd-14-665-2022)。
- 提供方与下载说明：[北京师范大学 CLICIA 数据页](https://clicia.bnu.edu.cn/data/erosivity.html)。
- 对应本地文件：`data/Soil/R-factor1991-2020.tif`。状态：**新增、尚未实际使用**。
- 数据页说明该文件属于 1991—2020 气候态多年平均年 R 因子，原数据为 0.01°×0.01° TIFF，单位为 `MJ mm / (ha h a)`；由两千多个站点的逐日和逐时降水资料生成，并用 62 个站点的逐分钟资料验证。后续使用前须核对本地栅格的 CRS、分辨率、单位、NoData 和是否重采样。

### U07 — 全球 EPIC 土壤可侵蚀性 K 因子

Natural Capital Alliance. **Global Soil Erodibility — EPIC K-Factor**. NatCap Data Hub.

- 数据页面：[Global Soil Erodibility — EPIC K-Factor](https://data.naturalcapitalalliance.stanford.edu/dataset/sts-c2b5e353d6ff11ee45519e5b795306e298d2d3fcf22b0f64ad4124a654717bda)（访问：2026-10-18）。
- 对应本地文件：`data/Soil/epic-k-factor.tif`。状态：**新增、尚未实际使用**。
- DOI：**数据页面未提供可确认的数据集 DOI**。页面说明该全球栅格使用 Sharpley & Williams (1990) 的 EPIC K 因子公式和 ISRIC SoilGrids 的砂粒、粉粒、黏粒及土壤有机碳栅格计算，并使用 0.1317 将美国惯用单位转换为公制；许可为 CC BY 4.0。
- 引用边界：当前应引用数据页面并保留访问日期、下载记录和文件校验值。EPIC 方法书目和 SoilGrids 输入文献属于数据血缘，可在实际使用该 K 因子时依据页面的 metadata YML 进一步补齐；它们不能替代该 NatCap 数据产品自身缺失的 DOI。

### U08 — 输沙量表

`data/Hydrology/输沙量2011-2024.csv`：现有文件，尚未发现进入年产水计算的代码引用。来源、统计口径及公开 DOI 尚未提供；正式使用前需补充原始公报／年鉴、表号、单位和整理过程。

### U09 — 青海湖与共和盆地的官方背景资料

| 原始来源 | 项目引用位置／用途 | DOI 状态 |
| --- | --- | --- |
| FAO. **People's Republic of China. A management plan for the Naked Carp Fishery of Qinghai Lake**，第 2 章 [The physical setting](https://www.fao.org/4/ag186e/AG186E03.htm) | `config/m2.yaml`：青海湖闭流背景 | 未查到独立 DOI；具体出版年待核对全书版权页 |
| Walker, K. F., & Yang, H. Z. **Fish and fisheries in western China**，收录于 FAO Fisheries Technical Paper 385，[章节原文](https://www.fao.org/4/X2614E/x2614e12.htm) | `audit_guide_connectivity.py`：区域水系背景 | 未查到独立 DOI |
| 共和县人民政府．[共和县地理环境（2024年）](https://www.gonghe.gov.cn/lnb/zjgh1/ghnj1__zjgh/ghnj/content_1013653549)，页面发布于 2026-06-03 | 共和盆地、茶卡／沙珠玉内流水系识别背景 | 政府网页，无已确认 DOI |

### U10 — 其他归档背景网页

| 原始来源 | 原引用文档 | DOI 状态与用途 |
| --- | --- | --- |
| 山东地情档案，黄河库：[第一辑 河道](https://shandong-chorography.org/database/a/section/21/article/4/) | `LONG_RECORD_VALIDATION.md` | 未查到 DOI；河段与地域背景，不是水文输入表来源证明 |
| 中国天气网（2019）．[黄河上游今年已发生两次洪水 卫星揭秘洪水从哪儿来](https://news.weather.com.cn/2019/07/3216437.shtml) | `MAIN_PROCESS_DIAGNOSIS.md` | 新闻网页，无已确认 DOI；仅作洪水年份背景，不作率定数据或方法依据 |

## 8. 论文中引用时需要保留的边界

1. **数据论文 DOI、数据集 DOI、软件 DOI、技术手册 DOI 分别使用。** 例如 D04 是 HWSD 技术报告 DOI，D09 是官方推荐论文 DOI，不能声称它们是本地派生文件的 DOI。
2. **当前主链与历史／计划分开。** CMFD、轮作月 Kc、月模型、MOD16 和 CMIP6 的状态不同，不能合并写成“本研究均采用”。
3. **参数值与方法来源分开。** 年度 Kc 先验、限制层代理、净损失 L、储留系数、分区方式、目标函数权重、抽样阈值与多数敏感性范围属于项目假设或率定结果；文献不能代替本地验证。
4. **自制数据和水文资料保留真实来源。** 无 DOI 不妨碍引用；作者、版本、年份、原始表页及获取方式比给出错误 DOI 更重要。当前最需补齐的是 U01—U03，以及原始 ALOS／Copernicus 下载批次。
5. **不能用这份参考文献表证明模型精度或地类优势。** 目标期是否参与拟合、对照参数预算、独立验证与未来外推范围，仍以实际实验协议和结果记录为准。
