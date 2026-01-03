from .results import Result, Results
from ..datasets.data import EvaluateResult
import csv
from io import StringIO
import re

def _parseLine_common(line: str, model: str, items: list, metrics: list[str],
                      dataset_getter, horizon_getter, item_name: str) -> Results:
    ''' 通用解析函数，用于解析按 metrics 顺序排列的数值字符串
    Args:
        line: 输入的数值字符串，按空格分隔
        model: 模型名称
        items: 要遍历的项目列表（可以是 horizons 或 datasets）
        metrics: 指标列表
        dataset_getter: 函数，接收 (item, idx) 返回 dataset 名称
        horizon_getter: 函数，接收 (item, idx) 返回 horizon 值
        item_name: 项目类型名称，用于错误提示（如 "horizon" 或 "dataset"）
    '''
    results = Results()
    values = line.strip().split()

    if not values:
        return results

    num_metrics = len(metrics)
    num_items = len(items)
    expected_values = num_metrics * num_items

    if len(values) < expected_values:
        print(f"警告: 期望 {expected_values} 个数值，但只找到 {len(values)} 个")
        return results

    for idx, item in enumerate(items):
        base_idx = idx * num_metrics
        try:
            metric_values = {}
            for m_idx, metric in enumerate(metrics):
                value_str = values[base_idx + m_idx].replace('%', '')
                metric_values[metric] = round(float(value_str), 2)

            eval_res = EvaluateResult(
                metric_values.get('MAE', 0.0),
                metric_values.get('MAPE', 0.0),
                metric_values.get('RMSE', 0.0)
            )

            dataset = dataset_getter(item, idx)
            horizon = horizon_getter(item, idx)
            split = getattr(item, 'split', "6:2:2") if hasattr(item, 'split') else "6:2:2"
            inLen = getattr(item, 'inLen', 12) if hasattr(item, 'inLen') else 12
            outLen = getattr(item, 'outLen', 12) if hasattr(item, 'outLen') else 12
            source = getattr(item, 'source', '') if hasattr(item, 'source') else ''
            tags = getattr(item, 'tags', 'survey') if hasattr(item, 'tags') else 'survey'

            result = Result(model, dataset, eval_res, split, inLen, outLen, horizon, source, tags)
            results.append(result)
        except (ValueError, IndexError) as e:
            print(f"解析 {item_name} {item} 时出错: {e}")
            continue

    return results

def parseLine_horizons(line: str, model: str, datasets: str, split="6:2:2", inLen=12, outLen=12,
                       tags="survey", source='', horizons:list[str]=[3,6,12,-1],
                       metrics:list[str]=['MAE', 'RMSE', 'MAPE']) -> Results:
    ''' 输入一行字符串 line，按空格隔开每个数值。顺序是 metrics 顺序；前三个数值是 horizons[0]，第 4-6 个是 horizons[1]，依此类推。
    usage: unittest/addLineToBaselineResults_test.py '''

    class Item:
        def __init__(self, horizon_val):
            self.horizon = horizon_val
            self.split = split
            self.inLen = inLen
            self.outLen = outLen
            self.source = source
            self.tags = tags

    items = [Item(h) for h in horizons]
    return _parseLine_common(
        line, model, items, metrics,
        dataset_getter=lambda item, idx: datasets,
        horizon_getter=lambda item, idx: item.horizon,
        item_name="horizon"
    )

def parseLine_datasets(line: str, model: str, datasets: list[str]=['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'],
                       split="6:2:2", inLen=12, outLen=12, tags="survey", source='', horizon=-1,
                       metrics:list[str]=['MAE', 'RMSE', 'MAPE']) -> Results:
    ''' 输入一行字符串 line，按空格隔开每个数值。顺序是 metrics 顺序；前三个数值是 datasets[0]，第 4-6 个是 datasets[1]，依此类推。
    usage: unittest/addLineToBaselineResults_test.py '''

    class Item:
        def __init__(self, dataset_name):
            self.dataset = dataset_name
            self.split = split
            self.inLen = inLen
            self.outLen = outLen
            self.source = source
            self.tags = tags

    items = [Item(d) for d in datasets]
    return _parseLine_common(
        line, model, items, metrics,
        dataset_getter=lambda item, idx: item.dataset,
        horizon_getter=lambda item, idx: horizon,
        item_name="dataset"
    )

class ParserBasicTS:
    ''' 从 BasicTS 格式的 checkpoints 的 .log 里提取结果
    BasicTS https://github.com/GestaltCogTeam/BasicTS 
    usage: unittest/rawResultParser_test.py '''
    @staticmethod
    def parse_horizon(path: str, dataset: str, model: str, horizons:str=[3,6,12], split="6:2:2", inLen=12, outLen=12, tags="survey", source='') -> Results:
        ''' 取 log 里最后的结果，就是 test horizon '''
        results = Results()
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            pattern = r'Evaluate best model on test data for horizon (\d+), Test MAE: ([\d.]+), Test RMSE: ([\d.]+), Test MAPE: ([\d.]+)'
            matches = re.findall(pattern, content)
            
            horizon_matches = {}
            for match in matches:
                horizon = int(match[0])
                if horizon in horizons:
                    horizon_matches[horizon] = EvaluateResult(round(float(match[1]), 2), round(float(match[3]) * 100, 2), round(float(match[2]), 2))
            for horizon in horizons:
                if horizon in horizon_matches:
                    result = Result(model, dataset, horizon_matches[horizon], split, inLen, outLen, horizon, source, tags)
                    results.append(result)
                else:
                    results[horizon] = None  
        except FileNotFoundError:
            print(f"文件 {path} 不存在")
        except Exception as e:
            print(f"读取文件时出错: {e}")
        # print(results.to_dataframe().to_csv(None, index=False, header=False))
        return results


def _parse_largest_like(input_str: str, dataset: str, split: str, inLen: int, outLen: int, tags: str, source: str, has_param: bool) -> Results:
    """公共解析器：按顺序消费 token；支持一行包含多个模型块。
    has_param=True 时：模型名以 Param（98K/1.2M/3B/-/–/—）结束；
    has_param=False 时：模型名后直接是 12 个数值指标（遇到数字即进入指标区）。
    usage: unittest/rawResultParser_test.py  """
    results = Results()
    text = input_str.strip()
    if not text:
        return results
    tokens = [t for t in re.split(r'\s+', text) if t]
    n = len(tokens)
    if n == 0:
        return results
    param_pattern = re.compile(r'^\d+(?:\.\d+)?[KMB]$', re.IGNORECASE)
    dash_params = {'-', '–', '—'}
    def is_param(tok: str) -> bool:
        return tok in dash_params or bool(param_pattern.match(tok))
    def is_numeric_or_percent(tok: str) -> bool:
        return bool(re.fullmatch(r'[\d.+-]+%?', tok))
    def clean_token(token: str) -> str:
        return re.sub(r'[^\d.%+-]', '', token)
    i = 0
    while i < n:
        # 解析模型名
        model_tokens = []
        if has_param:
            while i < n and not is_param(tokens[i]):
                model_tokens.append(tokens[i])
                i += 1
            if i < n and is_param(tokens[i]):
                i += 1  # 跳过 param
        else:
            while i < n and not is_numeric_or_percent(tokens[i]):
                model_tokens.append(tokens[i])
                i += 1
        if len(model_tokens) == 0:
            # 防止死循环
            i += 1
            continue
        model_name = ' '.join(model_tokens)
        # 收集 12 个指标
        metrics = []
        while i < n and len(metrics) < 12:
            ct = clean_token(tokens[i])
            if ct:
                metrics.append(ct)
            i += 1
        if len(metrics) < 12:
            break
        horizons = [3, 6, 12, -1]
        for h_idx, hz in enumerate(horizons):
            base = h_idx * 3
            try:
                mae = round(float(metrics[base]), 2)
                rmse = round(float(metrics[base + 1]), 2)
                mape = round(float(metrics[base + 2].replace('%', '')), 2)
            except Exception:
                continue
            eval_res = EvaluateResult(mae, mape, rmse)
            results.append(Result(model_name, dataset, eval_res, split, inLen, outLen, hz, source, tags))
    return results

class ParserLargeST:
    ''' LargeST 论文：LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting 解析器'''
    @staticmethod
    def parse(input_str: str, dataset: str, split: str = "6:2:2", inLen: int = 12, outLen: int = 12, tags: str = "survey", source: str = '') -> Results:
        ''' 解析 LargeST 风格汇总表2的多行或单行长文本；支持一行内多模型连续出现。
        每个模型块形如：Model Param MAE RMSE MAPE MAE RMSE MAPE MAE RMSE MAPE MAE RMSE MAPE
        分别对应 Horizon 3、6、12 与 Average（Average 记为 horizon=-1）'''
        return _parse_largest_like(input_str, dataset, split, inLen, outLen, tags, source, has_param=True)

class ParserPatchSTG:
    ''' PatchSTG 风格：每个模型块没有 Param，格式：Model MAE RMSE MAPE x3 + Average '''
    @staticmethod
    def parse(input_str: str, dataset: str, split: str = "6:2:2", inLen: int = 12, outLen: int = 12, tags: str = "survey", source: str = '') -> Results:
        ''' 解析 PatchSTG 风格汇总表2的多行或单行长文本；支持一行内多模型连续出现。
        每个模型块形如：Model MAE RMSE MAPE MAE RMSE MAPE MAE RMSE MAPE MAE RMSE MAPE
        分别对应 Horizon 3、6、12 与 Average（Average 记为 horizon=-1）'''
        return _parse_largest_like(input_str, dataset, split, inLen, outLen, tags, source, has_param=False)
    
class ParserD2STGNN:
    ''' Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting 
    对论文里的表格进行提取和转换格式。早期史山，已经完成使命懒得改了，规范性不如其他的类'''
    def parse_D2STGNN_horizon(input_str: str, dataset: str, split="6:2:2", inLen=12, outLen=12, tags="survey", source=''):
        ''' 对文章 Table 3 文本格式解析为 csv baselineResults 格式)，其结果可以手动复制粘贴合并或使用代码操作 '''
        # CodeBunny 写的感觉有点史
        headers = ["model", "dataset", "mae", "mape", "rmse", "split", "inLen", "outLen", "tags", "horizon", "source"]
        
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(headers)
        
        # 使用正则表达式分割模型块，处理多个空格和特殊字符
        model_blocks = re.split(r'\s{2,}', input_str.strip())
        
        for block in model_blocks:
            if not block.strip():
                continue
            
            # 分割每个单词，处理特殊字符
            parts = re.split(r'\s+', block.strip())
            model_name = parts[0]
            
            # 清理指标数据，移除*号等特殊字符
            metrics = []
            for part in parts[1:]:
                # 移除*号，保留数字和百分号
                cleaned = re.sub(r'[^\d.%]', '', part)
                if cleaned:
                    metrics.append(cleaned)
            
            # 每个模型应该有9个指标（3个horizon * 3个指标）
            if len(metrics) >= 9:
                horizons = [3, 6, 12]
                
                for i, horizon in enumerate(horizons):
                    start_idx = i * 3
                    if start_idx + 2 < len(metrics):
                        mae = metrics[start_idx]
                        rmse = metrics[start_idx + 1]
                        mape = metrics[start_idx + 2].replace("%", "")
                        
                        writer.writerow([model_name, dataset, mae, mape, rmse, split, inLen, outLen, tags, horizon, source])
        
        return output.getvalue()
    
# class ParserLargeST:
#     ''' LargeST 标准数据，6-2-2 12-12 2019 15 见论文 LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting '''
#     # TODO
