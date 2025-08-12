数据集 `data/` 

- > `raw/` 原始数据 (自行下载，如参考 [STD-MAE](https://github.com/Jimmy-7664/STD-MAE) 的 `raw_data/`)
- `processed/` 参考 [BasicTS](https://github.com/GestaltCogTeam/BasicTS) datasets，行成如 `processed/PEMS03` 的目录结构。

数据处理等 `utils/`

- `datasets/` 数据集操作
  - `graphHandler.py` 图读取和基本操作
  - `timeHandler.py` 计算数据集的时间
  - `statHandler.py` 计算数据集统计量
  - `dataset.py` 数据集读取和划分
  - `handler.py` 上面内容的捆绑包
  - `data.py` 输入、输出、真实值、评估结果集成及其存取
- `metrics/` 评价指标实现参考 [BasicTS](https://github.com/GestaltCogTeam/BasicTS)
  - `evaluation.py` 进行准确率评估
  - `plot.py` 绘图可视化对比结果
- `common/` 基础通用代码
  - `log.py` 输出和日志
- `baselines/` 基准模型
  - `HA.py`，历史平均值，不同节点、不同时间片、星期的三维度的平均值用作预测


提示词工程实验：`prompt/`

- `query.py` 带缓存的封装 API 查询
- `prompt.py` 不同的提示词方案
- `model.py` 主体类，执行实验

单元测试、调试代码等：`unittest/` 

## 提示词工程

