# 当前实验输出字段

更新：2026-10-12。当前运行目录为
`experiments/fine_2023_focus/20261011T164652_986860Z`。
方法见 [FINE_FOCUS_ROUTING.md](FINE_FOCUS_ROUTING.md)，结论和局限见
[实验记录](FINE_2023_FOCUS.md)。历史字段字典已完整保存于
[清理前快照](../experiments/maintenance/20261012_experiment_cleanup/source_before_cleanup.zip)
的 `docs/OUTPUT_FIELDS.md`，不能用旧阶段口径解释当前表格。

## 预测与三张表

`full_predictions.csv` 包含四种分类各 21 个站年，共 84 行。
`transfer_predictions.csv` 包含四种参数来源 × 四种输入地类 × 21 个站年，共 336 行。

| 字段 | 定义 |
| --- | --- |
| `product` | 输入地类：fine、fine_level1、copernicus、worldcover |
| `parameter_source` | 参数来源；互换时不重新率定 |
| `mode` / `fit` | 完整模型候选及拟合标识，不能逐站逐年混用 |
| `year` / `station` | 自然年及测站；固定三年七站 |
| `predicted` | 已经处理储留、已知耗水、水库蓄变和单个净损失后的模拟径流，亿 m³ |
| `observed` | 原实测年径流，亿 m³；不是天然化径流 |
| `eligible` | 该站年观测及账户是否可用，缺失不能补零 |

表二站年误差为 `100 * abs(predicted - observed) / observed`。
表一年度误差为当年七站相对误差的均方根；三年整体为 21 个站年相对误差的均方根。
不能将年度总误差解释为“把七站水量相加再计算的流域水量误差”，嵌套测站水量不能直接相加。
表三各格为对应参数来源/输入地类组合的三年整体相对 RMSE。
所有胜负使用全精度，平局不算胜出；详见 [验收口径](TWO_ACCURACY_GOALS.md)。

## 显式水量账

`water_balance.csv` 保存 2011—2024 的 98 行站/区间记录，以下水量均为亿 m³。
本地量指不重叠区间；累计量指截至该测站的全部上游，不能把两者混用。

| 字段 | 定义 |
| --- | --- |
| `natural` / `local_natural` | 累计 / 本地区间的 AWY 年产水 |
| `delivered` / `local_delivered` | 应用概念储留后、扣管理账户之前的累计 / 本地输送水量 |
| `storage` | 本地区间概念储留的年末状态 |
| `storage_change` | 本地区间当年概念储量变化，增加为正 |
| `consumed` | 本地区间原公报地表耗水 |
| `reservoir_change` | 本地区间原水库年末减年初蓄变，增加为正 |
| `loss` | 有效净损失，仅兰州—头道拐区间非零，其他区为零 |
| `predicted` / `local_predicted` | 累计模拟径流 / 相邻测站模拟径流差 |
| `observed` | 测站累计实测径流 |
| `closure` | 本地产水减去本地径流、耗水、水库蓄变、概念蓄变及净损失之和 |

`local_predicted` 可因干流沿程扣水而为负，不等于测站累计径流为负。
概念储量和实测水库蓄变是不同账户；代数闭合不能证明两者的物理解释正确或充分。
`loss` 不是独立测得的渗漏，也不是再次分配公报耗水。

## 候选、诊断和审计

- `candidate_metrics.csv`、`candidate_predictions.csv` 与 `candidate_*.json`：全部候选及初值收敛信息。
- `selection.json`：选择规则、选中候选，以及目标年份已用于拟合的明确标识。
- `current_parameters.json`：唯一完整冻结参数，跨站跨年统一使用。
- `cv_predictions.csv`、`cv_metrics.csv`：原训练折分别重拟合路由的历史诊断；不等同于最终三年拟合模型的独立验证。
- `all_year_changes.csv`：完整站年改善及退化，`change_percentage_points > 0` 表示误差上升。
- `quadrature.json`：抽样与全像元差值除以观测，保存的是比例，乘 100 才是百分数。
- `sources.json`：输入文件 SHA256；`executed_*.py`：部分实际执行代码快照。
- `balance_check.json`、`performance_full.json`、`status.json`：闭合、计算资源和运行状态。

原协议配置中的 `validation_years` 为兼容保留的字段名。本轮对应年份已经参与开发期目标拟合，不能因字段名包含 validation 就称为独立验证。
