### 1. 项目介绍 (Project Introduction)
这是一个基于InVEST Annual Water Yield，对黄河流域进行水文模拟的项目，输出数据为几个重要测站的年径流量。需要做的实验有3个：
1. 分别用精细的地表划分数据和现成的地表覆盖产品计算径流量，证明精细的地表划分数据比现成的产品能更准确地模拟水文（测站流量）。
2. 根据CMIP6不同气候场景模拟水文变化。
3. 改变地表覆盖类别后模拟水文变化。
### 2. 项目结构 (Project Structure)
 
├── AGENTS.md  
├── DEVELOPMENT.md  
├── config.yaml  
├── .gitignore  
├── data/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Climate/  # 气象数据  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── DEM/  # DEM数据、河流湖泊数据  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── LCLU/  # 地表覆盖数据   
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Soil/  #土壤数据  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── Hydrology/  #实际的水文数据  
├── project/   # 存储计算后的基础数据，避免后续做实验时重复计算  
├── experiments/   # 存储每次实验的log、结果   
├── scripts/  # 程序入口（train.py, test.py）  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── test/  # 各种测试模块功能的脚本  
└── src/  # 源码  

### 3. 项目背景与约束 (Context & Constraints)
**个人背景:** 我对水文模拟一窍不通，需要你经常思考，并上网查阅相关资料。需要考虑水文站实测的“径流量”和InVEST算出来的产水量概念上的差异，考虑有哪些过程InVEST没有模拟到。
**硬件环境:** 本地单机部署，CPU：3700x，内存32Gb
**效率限制:** 每次模拟不能超过12h
### 4. 严格行为准则 (Strict Directives for Agent)
1. 必须考虑InVEST的计算速度，执行并行计算或计算加速，优化算法，保证效率。
2. 所有提供的 Python 代码必须遵循 Ruff 的规范。
3. 本AGENTS.md文档只能由人工修改，AI Agent禁止修改。