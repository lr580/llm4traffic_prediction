**介绍：** 一个调用 LLM API 进行时空序列预测(交通流量预测)的框架。核心功能亮点如下：

1. 封装了常用操作，如生成数据子集、带缓存的 API 调用、日志和可视化等。
2. 支持快速扩展，可以方便添加多种不同的 API、提示词、数据集。
3. 实现了几种不同的常用提示词和 HA 基准模型。

> 版权所有，如您需要使用本项目(如使用本文档提供的实验结果数据)，请标记出处/引用。

## 项目结构

包依赖参见 `requirements.txt`。

数据集 `data/`

- > `raw/` 原始数据 (自行下载，如参考 [STD-MAE](https://github.com/Jimmy-7664/STD-MAE) 的 `raw_data/`，目前无使用)
  >
- `processed/` 参考 [BasicTS](https://github.com/GestaltCogTeam/BasicTS) datasets (`all_data.zip `)，形成如 `processed/PEMS03` 的目录结构。

数据处理等 `utils/`

- `datasets/` 数据集操作

  - `graphHandler.py` 图读取和基本操作
  - `timeHandler.py` 计算数据集的时间
  - `statHandler.py` 基于数据集计算数据集统计量(如历史平均)
  - `calcHandler.py` 基于数据计算数据集统计量(如均值、自相关)
  - `dataset.py` 数据集读取和划分
  - `handler.py` 上面内容的捆绑包
  - `data.py` 输入、输出、真实值、评估结果集成及其存取
  - `converter.py` 实现多数据源的格式转换与变换
- `metrics/` 评价指标实现参考 [BasicTS](https://github.com/GestaltCogTeam/BasicTS)

  - `evaluation.py` 进行准确率评估
  - `plot.py` 绘图可视化对比结果
- `common/` 基础通用代码

  - `log.py` 输出、日志、格式转换
  - `tex.py` 处理 LaTeX 格式转换的辅助函数
- `baselines/` 基准模型

  - `HA.py`，历史平均值，不同节点、不同时间片、星期的三维度的平均值用作预测
  - `results.py` 对基准模型结果进行多角度展示对比等
  - `citations.py` 维护 LaTeX 引用，给结果注入引用
  - `citations.json` 可自行更换的模型->引用映射表
  - `baselineResults.csv` 部分经典基准模型结果
  - `rawResultParser.py` 辅助中间函数，解析论文原始表格结果数据
  - `rawResults.py` 一些论文文本原始结果
  - `eda/` 对多个数据集的一些统计脚本和观察结论(详见文件夹内 `.md`说明，如 [此处](https://lr580.github.io/llm4traffic_prediction/) 在线可视化查看 LargeST 数据集图结构)

提示词工程实验：`prompt/`

- `query.py` 带缓存的封装 API 查询
- `prompt.py` 不同的提示词方案
- `model.py` 主体类，执行实验

单元测试、调试代码等：`unittest/`

- `prompt_test.py` 执行提示词工程小批量数据集实验



## 提示词工程

[提示词工程](docs/提示词工程.md)，点击查看。包含：

- 几种提示词 (普通 / +历史平均 / +邻居信息等)设计的例子

## 实验结果

[实验结果](docs/实验结果.md)，包含：

- 基准模型 (历史平均) 的全测试集结果 PEMS0x, LargeST
- PEMS0x 对随机样本的提示词结果对比



其它文档：

- [可行性分析](docs/可行性分析.md)，包括对 API 调用的时空开销估算；本地部署 Qwen 小模型的准确率



