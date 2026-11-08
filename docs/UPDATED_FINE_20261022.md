# 20261022 精细分类图全流程重算

执行日期：2026-10-24。输入为 `data/LCLU/vegetation_model_20261022_150606.tif` 及同名 CSV。
2026-10-25 完成报告核验与交付：[综合图表](../experiments/updated_fine_20261022/report.html)。
本轮沿用当前 48 类参数库，不新增类别专属参数，不根据目标期排名追加候选。
旧实验、旧缓存和人工维护的 AGENTS.md 保留。

## 最终结果

| 比较方式 | 2022 误差 | 2023 误差 | 2024 误差 | 三年整体 | 年度及整体最优 | 站年最优 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 精细图独立率定 | 7.625607% | 6.681956% | 18.343934% | 12.100941% | 0/4 | 5/21 |
| 精细图共用参数 | 8.577525% | 10.329034% | 14.679757% | 11.485616% | 2/4 | 8/21 |

未达到多数最优目标。独立率定其他图整体误差：大类 11.689203%、Copernicus 11.873565%、
WorldCover 11.390706%。本轮候选未使用目标期排名改选。
两组均选择弱空间约束候选；50 个历史折拟合约 695.45 秒，最终率定约 139.52 秒。
全像元阶段约 811.42 秒；大类后半段年份增加两个并行进程，总并行数最高 4。
补充计算保留独立执行清单，2011 年与原计算逐值一致，9 个补充年份与最终检查点全部核对通过。
原主进程树监测峰值约 7.23 GiB（不含独立补充进程树，不作为总峰值）。
420 条目标期预测与水量账户一致，最大闭合误差 1.71e-13 亿 m³。
47 项针对性测试通过；修正一项测试对已删除旧图例的硬编码，改为读取当前配置。

## 输入与计算口径

- 新建 `project/comparison_2011_2024_fine_20261022/`，重新对齐新图、应用既定湖泊掩膜，并重建分层样本。
- 验证原有气候、其他土地覆盖产品和计算网格指纹后复用；不重复下载气候或重建不变的地形。
- 重新计算精细图的大类混合比例，四图样本统一使用当前参数库；不能沿用旧实验的 29 类对应关系。
- 水量使用 TerraClimate 2011—2024，7 个具备完整账户的测站；泥沙使用 CMFD 月降水近似年度 R，14 个控制流域、10 个实测站。
- 250 米等面积网格。水文两进程、SDR 两工作进程，年度泥沙两线程分块汇总，适配 32 GiB 内存。
- 新图固定用于所有年份，不代表历史逐年地类。水量与泥沙并行计算，尚不是动态河库水沙耦合模型。

## 水文流程

采用既有 `historical_refinement` 的五个固定候选、两个历史时间折和相同初值。
先历史选型与最终率定、封存，再对 2022—2024 年做全像元核算。
独立率定和共用参数两组均运行，保留年度误差、21 个站年误差及 4×4 参数互换表。
目标年份过去已被查看，因此属于回顾性评价，不能写成新的独立盲测。
与旧基线的差异同时涉及地图和已更新的参数库，不应全部归因为这一次地图修改。

水量结果区分 InVEST 产水量与账户核算后的测站径流。概念储留、耗水、库容变化和有效损失
均单独记录；有效损失是残差参数，不是独立观测到的耗水或渗漏。
2014 年头道拐及下游因上游蓄水变化缺测、2018 年因管理账户缺测，核算后的径流留空。
这 12 个站年不补零；14 年产水计算完整，2022—2024 年 21 个评价站年完整。

## 泥沙流程与已完成检查

完整重跑官方 InVEST SDR，新基准位于
`experiments/sediment/20261024T081800_172503Z/`，约 272.84 秒。
重新计算 C/P 对应关系及与新图有关的 K 缺测处理，不复用旧图 SDR 输出。
K 已是公制，换算乘数保持 1，不重复乘 0.1317。

年度结果位于 `experiments/updated_fine_20261022/sediment_annual/`，约 110.82 秒。
固定地类、C/P、K 与连通性，利用 R 的线性关系生成年度 VRT，避免重复 14 次地形运算。
`R_y = R_1991_2020 × MFI_y / mean(MFI_1991_2020)` 是未验证年际精度的项目近似。
196 条模型记录、140 条实测参考记录齐全；2024 年完整栅格和 14 站总量复核通过。
最大站点相对差为 8.88e-16，R 乘积差为零。

利津多年平均坡面入河基准约 1.676887 亿吨/年。全域仍有 22,395 个非河道像元
没有有效输移结果，未填零；利津非河道输出覆盖率约 99.8087%。
SDR 不包含沟岸侵蚀和库河冲淤；实测只作参考，不能将两者差值当作库坝拦沙观测。

## 复现入口

所有 Python 命令在项目环境中执行，可用
`powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python ...`。

```text
scripts/prepare_updated_fine.py
scripts/prepare_updated_experiment.py --output experiments/<新的运行目录>
scripts/experiment_historical_refinement.py --config experiments/<新的运行目录>/historical_refinement.yaml
scripts/report_historical_refinement.py experiments/<新的运行目录>/water/<运行时间>
scripts/run_sediment.py
scripts/run_annual_sediment.py --sdr-run experiments/sediment/<运行时间> --output experiments/<新的运行目录>/sediment_annual --workers 2
scripts/verify_annual_sediment.py experiments/<新的运行目录>/sediment_annual
scripts/report_updated_water_sediment.py --water <水文目录> --sediment <SDR目录> --annual <年度泥沙目录> --output <综合目录>
```

本次样本准备在压缩写入尚未结束时曾提前调用，失败后等待写入完成再重新准备。
新增样本指纹检查防止未完成缓存被使用；失败的准备目录保留，不参与最终实验。
环境启动日志中的 micromamba 用户目录警告不影响实际计算；成功以模型完成清单及数值检查为准。

## 方法参考

- [InVEST Annual Water Yield](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/annual_water_yield.html)
- [InVEST SDR](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/sdr.html)
- [Li et al. (2023)，C/P 数据参考](https://doi.org/10.1038/s41597-023-02246-4)
- [Yin et al. (2015)，降水分辨率与侵蚀力](https://doi.org/10.5194/hess-19-4113-2015)
- 项目原方法与假设：[历史期选型](HISTORICAL_REFINEMENT.md)、[SDR](SEDIMENT_SDR.md)、[年度泥沙](STATION_SEDIMENT.md)。
