from .results import Result, Results
from ..datasets.data import EvaluateResult
import csv
from io import StringIO
import re

class ParserBasicTS:
    ''' 从 BasicTS 格式的 checkpoints 的 .log 里提取结果
    BasicTS https://github.com/GestaltCogTeam/BasicTS '''
    @staticmethod
    def parse_horizon(path: str, dataset: str, model: str, horizons:str=[3,6,12], split="6:2:2", inLen=12, outLen=12, tags="survey", source=''):
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
    has_param=False 时：模型名后直接是 12 个数值指标（遇到数字即进入指标区）。"""
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
    
class ParserLargeST:
    ''' LargeST 标准数据，6-2-2 12-12 2019 15 见论文 LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting '''

    # TODO
