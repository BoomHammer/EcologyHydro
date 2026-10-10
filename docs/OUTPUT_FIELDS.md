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

上述为旧冻结实验与通用运行接口。新增 `regional_water_v1/v2/stations_*.csv` 使用独立字段：`awy_yield` 为累计 AWY 产水、`consumption_cumulative` 为条件情景扣除的累计地表耗水、`storage_cumulative` 为累计水库年末减年初蓄变、`predicted` 为两项扣除后的径流代理、`observed` 为实测，水量均为亿 m³；`relative_error=predicted/observed−1`。未加管理的对照行两项扣除为零，仅表示未启用算子，不表示实际无耗水或蓄变。新结果仅覆盖六个可对应的站，不能当作所有站均已实现 `human_adjusted_m3`。详见 [新诊断](REGIONAL_WATER_DIAGNOSIS.md)。

`reach_loss_v1/stations_*.csv` 另有 `net_loss_cumulative`：兰州及以上为 0，头道拐及以下为同一个拟合年度 L，单位亿 m³；每个累计站扣一次，不逐站叠加。`predicted=awy_yield−consumption_cumulative−storage_cumulative−net_loss_cumulative`。L 是未识别账户差额，不能当作实测地下水通量。

空间核查 `yield_by_zone_category.csv` 的 `zone` 为 1 起算增量区，类别 `all/mapped_inland/desert_proxy/inland_or_desert` 存在包含关系，不能相加；`yield_1e8_m3` 为该增量范围的局地产水。`reach_attribution.csv` 中“移除产水后”均是固定原参数的反事实上限，不是重新率定或已验证的边界修复。裸地敏感性中的 `yield_change_1e8_m3` 为相对原 Kc=0.15 的累计产水变化。详见 [空间与净损失诊断](DRYLAND_CONNECTIVITY_DIAGNOSIS.md)。

`connected_water_v1/`、`reference_rainfall_v1/stations_*.csv` 中，`known_adjustment_cumulative` 已包括已知耗水、水库蓄变，以及启用时的土壤／积雪蓄变；`soil_snow_delta_cumulative` 是其中的解释项，**不能再扣一遍**。恒等式为 `predicted=awy_yield−known_adjustment_cumulative−net_loss_cumulative`，水量均为亿 m³。蓄变原表 `storage_changes.csv` 为非重叠增量区的年度差，`variable=soil/swe` 可相加，正值表示蓄水增加。新降水 `factor` 无量纲，不是径流订正系数。

历史 `station_comparison_status.csv` 的 `eligible_for_calibration=false` 指部分还原账户不能作为严格天然化目标，与后来直接拟合实测径流的选择不矛盾。`Q+地表水耗水+水库蓄变量` 未解决地下水、调水、边界及蒸散重叠，不能改称天然径流。

训练汇总为 `project/calibration/major_bias/official_comparison.csv`（两套共 88 条），测试为 `holdout_2023/stations.csv`（共 22 条）。每套训练 44 个站年、测试 11 站；嵌套测站不是独立样本。

- MAPE：各有效站年绝对相对误差的平均，%。
- 相对 RMSE：各站年相对误差平方平均后开根，%。
- RMSE：水量误差平方平均后开根，亿 m³。
- 总体偏差：模拟总量减实测总量后除以实测总量，%；嵌套站总量仅作指标计算，不是流域总水量。

`stability_v1/` 仅评价 2019—2022 年。`cv_stations_*.csv` 每行是该年从率定中排除后的预测，`refit_predictions_*.csv` 另将每折模型作用于相同四年的气象输入，用于比较删年敏感性；后者中训练年份的预测不能当作额外留出样本。`stations_*.csv` 为全四年训练方案的官方内核全像元计算，不是验证预测。

- `worst_year_relative_rmse_pct`：分别计算四个留出年份六站相对 RMSE 后取最大。
- `delete_year_prediction_rms_change_pct`：每折预测减全四年训练方案预测，再除以同站同年实测；对 4 折×4 年×6 站求均方根，单位 %。它表示对训练资料删年的敏感性，不是置信区间，也不是另一套 96 个独立验证样本。
- `year_mape_std_pct`：四个留出年 MAPE 的总体标准差，单位百分点。
- `negative_heldout_predictions`：负模拟径流个数，保留原值，不裁零美化指标。
- `parameters_*.json` 的组合模型记录成员列表；预测逐成员算水量再等权平均，不平均 Z/Kc 后另算。
- `climate_sensitivity_*.csv` 的 `factor=0.95/1.05` 是固定参数下的假设性气象扰动。响应除以实测做尺度归一化，不是气候数据实测不确定度，也不意味着对扰动情景有观测真值。

小负值舍入修正见 [预处理](M2_PREPROCESSING.md)。真实负值、缺失及面积丢失仍须报错。
