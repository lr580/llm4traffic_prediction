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

考虑随机取测试集的一部分，用部分值代替整体。确定何时部分值收敛。经过测试，当取 32/64 个测试点(设一个测试点=随机 12 个区间1小时输入，单个探测器空间点)时，结果基本稳定，PEMS03 的 MAPE 误差在 1% 以内。所以这里按 32 个测试点，计算实验结果，如下：

> 其中，(T) 是使用深度思考。可见，与不使用相比，提升效果不明显。

```
PEMS03-64:
              mae      mape       rmse
Plain   23.368322  0.206827  34.952785
HA      19.761604  0.169685  30.892782
HA_Nei  16.206450  0.147683  24.311964

PEMS03-32:
                mae      mape       rmse
Plain     32.979980  0.249627  46.975353
Neighbor  26.727194  0.204145  36.791534
HA        22.045837  0.150311  33.868702
HA_Nei    19.058632  0.140031  26.726738
HA_Nei(T) 18.892052  0.123374  28.649897

PEMS04-32:
                mae      mape       rmse
Plain     32.174080  0.184957  61.973930
Neighbor  26.929720  0.167282  54.225113
HA        29.372223  0.149416  57.545109
HA_Nei    15.777431  0.123573  32.505707

PEMS07-32:
                mae      mape       rmse
Plain     38.792011  0.129565  57.948433
Neighbor  40.558704  0.144437  63.940178
HA        39.854847  0.135642  66.284691
HA_Nei    20.265905  0.075539  28.247744

PEMS08-32:
                mae      mape       rmse
Plain     22.445910  0.132522  34.405945
Neighbor  22.497286  0.132565  34.148689
HA        21.692184  0.134507  33.723923
HA_Nei    18.505394  0.109718  30.051735

PEMS03-16-3:
                mae      mape       rmse
Plain     25.073412  0.238844  42.445835
Neighbor  24.834314  0.224607  45.987045
HA        23.087145  0.234999  35.094261
HA_Nei    17.983538  0.189089  26.264658
```

PEMS-BAY, METR-LA 数据量数量级类似，不再尝试。BasicTS 里也没有显著特别小很多的其他合适数据集了。

其他模型对比：以 HA_Nei 和一个新的 PEMS03-32 为例：

```
                mae      mape       rmse
deepseek  21.228708  0.173750  31.899065
gpt5.2    18.718863  0.166928  28.815720
```

> 附：使用历史平均 (按星期(dow)、时间片(tod)、空间点三关键字分组)的对比如下
>
> ```
> PEMS03 MAE:26.1007, MAPE:0.2687, RMSE:47.4744
> PEMS04 MAE:26.4224, MAPE:0.1678, RMSE:43.4247
> PEMS07 MAE:30.3553, MAPE:0.1280, RMSE:56.7535
> PEMS08 MAE:23.2495, MAPE:0.1450, RMSE:40.5865
> SD     MAE:34.5474, MAPE:0.1958, RMSE:72.9972
> GBA    MAE:32.3985, MAPE:0.2477, RMSE:56.2634
> GLA    MAE:34.5100, MAPE:0.2289, RMSE:63.0197
> CA     MAE:31.8463, MAPE:0.2366, RMSE:59.0193
> ```
>



其它文档：

- [可行性分析](docs/可行性分析.md)，包括对 API 调用的时空开销估算；本地部署 Qwen 小模型的准确率



