# 历史期选型的有限改进实验

2026-10-13：用户希望在尽量客观比较的情况下降低精细分类误差，并在多数情况略优即可；随后明确继续历史期选型，优先客观性。

本轮保留原始报告和源码，新增独立实验目录。将多数站年操作性解释为至少 11/21 严格最优；年度及整体四项报告实际最优数，以至少 3/4 作为多数指标的辅助检查。同时要求精细分类三年整体误差低于原独立率定的 12.796501%。这些是目标，不能保证达成，不能用于历史期候选排序。旧 4/4、9/21 指标仅作为历史记录。

原结果各产品 2024 年共同失准、Z 多次触及边界，为检验参数补偿与时间外推提供了诊断线索，并没有证明地类图错误。本轮不增加类别专属参数，原类别 Kc、根深、灌溉处理、裸地系数、Kc 上限、水量账户均保持。模型仍是七区 Z、一个共同植被倍率、一个净损失、一个储留参数。

固定五个候选：原 regional；log Z 围绕七区均值的惩罚强度 0.001、0.01；历史观测权重按三年半衰期衰减；三年半衰期与 0.001 惩罚的组合。惩罚项为 λ × mean((log Z − mean(log Z))²)。半衰期和惩罚网格是本项目预先固定的实验假设，不是文献证明的黄河流域最优值。时间加权只改变历史拟合损失，各评价期仍使用原等权指标，状态传播保留所有连续年份。每图仍是一套跨年固定参数。

原 regional 候选从三个原固定初值重拟合，不借用旧最优参数；全部候选边界和求解预算相同。沿用两组前推历史折，以四图合并的未加权相对平方误差选择。独立率定组与共同参数组分别选型，不因精细图排名挑选模型。参数与协议封存后才计算完整 2022—2024 年七站全像元预测和 4×4 互换矩阵。若未达目标，保留全部不利结果，不追加针对目标期的候选搜索。

计算使用解析梯度、已有加权样本、两个单线程进程、24 GiB 上限；最终统一分块全像元复算。原目标年份曾被查看，本轮亦由已有误差诊断发起，不能称独立盲测。

依据与参考文献：

- InVEST 官方 [Annual Water Yield](https://storage.googleapis.com/releases.naturalcapitalproject.org/invest/3.19.0/userguide/en/annual_water_yield.html)：年产水量与可达水量、耗水及模型简化。现有净损失和储留为模型扩展，不是新增观测过程。
- Jay-Allemand et al. (2020), [On the potential of variational calibration for a fully distributed hydrological model](https://hess.copernicus.org/articles/24/5519/2020/), HESS 24:5519–5538：分布式率定的参数过多、等效参数及正则化问题；本轮惩罚公式为项目设计。
- Gharari et al. (2013), [An approach to identify time consistent model parameters: sub-period calibration](https://hess.copernicus.org/articles/17/149/2013/), HESS 17:149–161：子时期率定与参数时间一致性；本轮不采用时变目标期参数。
- Cawley & Talbot (2010), [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://jmlr.org/papers/v11/cawley10a.html)：选型与性能评价分离。
