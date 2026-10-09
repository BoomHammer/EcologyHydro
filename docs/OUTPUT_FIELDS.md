# 测站输出口径与字段字典

适用于官方 AWY 运行的 `stations.csv`。历史字段名保留，当前实验见 `config/runoff_experiment.json`。

| 字段 | 定义与单位 |
| --- | --- |
| `natural_yield_m3` / `natural_yield_1e8_m3` | 历史名称；未显式修正人工水量及储量的 AWY 年产水，m³ / 亿 m³，不等于严格天然径流 |
| `yield_depth_mm` | 产水除以实际计算上游面积的等效深度，mm |
| `mean_flow_m3_s` | 年产水除以当年秒数，m³/s，是等效产水流率 |
| `observed_1e8_m3` | 受调控实测年径流，亿 m³ |
| `observation_raw/status` | 原记录与有效、缺失、异常状态，缺失不补零 |
| `area_km2/valid_area_km2` | 实际计算上游面积 / 有效产水面积，km² |
| `human_adjusted_m3` | 尚未实现的逐站同口径修正，目前为空，空值不等于零 |
| `human_adjustment_status` | `not_available` 表示没有完成核验的修正，不表示没有管理资料 |
| `run_id/year/climate/landcover/quality` | 运行、年份、方案和质量追溯 |

当前率定直接将 AWY 累计产水作为实测径流代理，误差指标不表示完成天然化或管理修正。AWY 已扣 AET，不重复扣相同蒸散。

历史 `station_comparison_status.csv` 的 `eligible_for_calibration=false` 指部分还原账户不能作为严格天然化目标，与后来直接拟合实测径流的选择不矛盾。`Q+地表水耗水+水库蓄变量` 未解决地下水、调水、边界及蒸散重叠，不能改称天然径流。

训练汇总为 `project/calibration/major_bias/official_comparison.csv`（两套共 88 条），测试为 `holdout_2023/stations.csv`（共 22 条）。每套训练 44 个站年、测试 11 站；嵌套测站不是独立样本。

- MAPE：各有效站年绝对相对误差的平均，%。
- 相对 RMSE：各站年相对误差平方平均后开根，%。
- RMSE：水量误差平方平均后开根，亿 m³。
- 总体偏差：模拟总量减实测总量后除以实测总量，%；嵌套站总量仅作指标计算，不是流域总水量。

小负值舍入修正见 [预处理](M2_PREPROCESSING.md)。真实负值、缺失及面积丢失仍须报错。
