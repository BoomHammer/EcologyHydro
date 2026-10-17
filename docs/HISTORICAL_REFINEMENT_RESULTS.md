# 2026-10-13 历史期选型改进：部分降误差，尚未多数最优

用户明确选择继续历史期选型、优先客观性。已按 [固定方法](HISTORICAL_REFINEMENT.md) 完成五候选、四图等机会的历史率定，并在目标期计算前封存。原始报告和 AGENTS.md 未改。

- [完整三张表](../experiments/historical_refinement/20261013T150526_240357Z/three_tables.html)
- [结果诊断与审计](../experiments/historical_refinement/20261013T150526_240357Z/analysis.md)
- [全部候选的历史成绩](../experiments/historical_refinement/20261013T150526_240357Z/candidate_scores.csv)

| 精细分类的比较方式 | 原整体相对 RMSE | 新整体相对 RMSE | 汇总最优数 | 站年最优数 |
| --- | ---: | ---: | ---: | ---: |
| 四图独立率定 | 12.796501% | 12.805157% | 0/4 | 7/21 |
| 四图共用参数 | 12.205820% | 9.813288% | 1/4 | 5/21 |

独立率定选择轻度 log Z 约束，历史两折四图合并误差由 14.198245% 降至 13.866668%。但早期选型后的后期前推误差由 17.085522% 升至 17.132491%，目标期整体误差也未改善，不能将合并历史成绩改善等同于可靠外推提升。

共同参数组选中较强约束，历史合并误差由 20.992306% 降至 19.539358%，后期前推从 19.233645% 降至 18.139149%。精细图目标期整体误差确有下降，但大类图为 9.307422%、Copernicus 为 9.483441%，都仍低于精细图；精细图仅 2024 年汇总误差最小。站年严格最优从原 7/21 降至 5/21，不满足至少 11/21 的多数定义。未按目标期结果改选其他候选。

参数诊断：精细图仍有分区 Z 触及边界；加权样本中约 53.68% 植被面积的 Kc 被既有 1.3 上限截断。该现象会压缩部分 Kc 差异，不能据此断言全部分类信息丢失或认定它是主因。有效净损失仍为残差参数，不能当成独立观测耗水。当前结果提示需要独立蒸散或水量过程资料约束参数；本次没有为获得有利排名取消上限、扩大边界或新增类别参数。

审计与效率：

- 420 条目标期预测与逐区水量账户重算一致，净损失只在一个区间扣除，详细闭合误差见 `water_balance_audit.json`。
- 全部主表用全像元结果，20 组地图×参数来源组合均通过高分辨率抽样与全像元差异不超过观测量 1% 的检查；这只是数值检查，不是水文过程验证。
- 历史折拟合耗时 680.78 秒，最终历史参数拟合 92.49 秒，全像元复算 822.11 秒；两个进程，全像元阶段峰值约 3.58 GiB。
- 226 项测试通过，新增拟合梯度、未来数据隔离及等权检查通过；Ruff 检查和格式检查通过。原两份实验目录和 AGENTS.md 共 227 个文件的哈希保持不变。
- 目标年份早已查看，本轮由既有诊断发起，是回顾性方法开发，不是独立盲测。未达成目标的结果完整保留。

复现（项目根目录）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/experiment_historical_refinement.py --stage select
# 将上一条打印的 OUTPUT 目录传给 --run，先封存、后评价。
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/experiment_historical_refinement.py --run <OUTPUT目录> --stage evaluate
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run.ps1 python scripts/report_historical_refinement.py <OUTPUT目录>
```

方法来源与文献保存在 [HISTORICAL_REFINEMENT.md](HISTORICAL_REFINEMENT.md)。本次没有取得或引入新的外部数据。
