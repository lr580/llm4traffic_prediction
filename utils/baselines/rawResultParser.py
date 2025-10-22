from .baselineResults import Result, Results
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

    RAW_HORIZONS_PEMS04 = 'HA 28.92 42.69 20.31% 33.73 49.37 24.01% 46.97 67.43 35.11%  VAR 21.94 34.30 16.42% 23.72 36.58 18.02% 26.76 40.28 20.94%  SVR 22.52 35.30 14.71% 27.63 42.23 18.29% 37.86 56.01 26.72%  FC-LSTM 21.42 33.37 15.32% 25.83 39.10 20.35% 36.41 50.73 29.92%  DCRNN 20.34 31.94 13.65% 23.21 36.15 15.70% 29.24 44.81 20.09%  STGCN 19.35 30.76 12.81% 21.85 34.43 14.13% 26.97 41.11 16.84%  Graph WaveNet 18.15 29.24 12.27% 19.12 30.62 13.28% 20.69 33.02 14.11%  ASTGCN 20.15 31.43 14.03% 22.09 34.34 15.47% 26.03 40.02 19.17%  STSGCN 19.41 30.69 12.82% 21.83 34.33 14.54% 26.27 40.11 14.71%  MTGNN 18.22 30.13 12.47% 19.27 32.21 13.09% 20.93 34.49 14.02%  GMAN 18.28 29.32 12.35% 18.75 30.77 12.96% 19.95 30.21 12.97%  DGCRN 18.27 28.97 12.36% 19.39 30.86 13.42% 21.09 33.59 14.94%  D2STGNN 17.44∗ 28.64∗ 11.64%∗ 18.28∗ 30.10∗ 12.10%∗ 19.55∗ 31.99 12.82%∗'
    RAW_HORIZONS_PEMS08 = 'HA 23.52 34.96 14.72% 27.67 40.89 17.37% 39.28 56.74 25.17%  VAR 19.52 29.73 12.54% 22.25 33.30 14.23% 26.17 38.97 17.32%  SVR 17.93 27.69 10.95% 22.41 34.53 13.97% 32.11 47.03 20.99%  FC-LSTM 17.38 26.27 12.63% 21.22 31.97 17.32% 30.69 43.96 25.72%  DCRNN 15.64 25.48 10.04% 17.88 27.63 11.38% 22.51 34.21 14.17%  STGCN 15.30 25.03 9.88% 17.69 27.27 11.03% 25.46 33.71 13.34%  Graph WaveNet 14.02 22.76 8.95% 15.24 24.22 9.57% 16.67 26.77 10.86%  ASTGCN 16.48 25.09 11.03% 18.66 28.17 12.23% 22.83 33.68 15.24% STSGCN 15.45 24.39 10.22% 16.93 26.53 10.84% 19.50 30.43 12.27%  MTGNN 14.24 22.43 9.02% 15.30 24.32 9.58% 16.85 26.93 10.57%  GMAN 13.80 22.88 9.41% 14.62 24.02 9.57% 15.72 25.96 10.56%  DGCRN 13.89 22.07 9.19% 14.92 23.99 9.85% 16.73 26.88 10.84%  D2STGNN 13.14∗ 21.42∗ 8.55%∗ 14.21∗ 23.65∗ 9.12%∗ 15.69∗ 26.41 10.17%∗'
    RAW_HORIZONS_PEMSBAY = 'HA 1.89 4.30 4.16% 2.50 5.82 5.62% 3.31 7.54 7.65%  VAR 1.74 3.16 3.60% 2.32 4.25 5.00% 2.93 5.44 6.50%  SVR 1.85 3.59 3.80% 2.48 5.18 5.50% 3.28 7.08 8.00%  FC-LSTM 2.05 4.19 4.80% 2.20 4.55 5.20% 2.37 4.96 5.70%  DCRNN 1.38 2.95 2.90% 1.74 3.97 3.90% 2.07 4.74 4.90%  STGCN 1.36 2.96 2.90% 1.81 4.27 4.17% 2.49 5.69 5.79%  Graph WaveNet 1.30 2.74 2.73% 1.63 3.70 3.67% 1.95 4.52 4.63%  ASTGCN 1.52 3.13 3.22% 2.01 4.27 4.48% 2.61 5.42 6.00%  STSGCN 1.44 3.01 3.04% 1.83 4.18 4.17% 2.26 5.21 5.40%  MTGNN 1.32 2.79 2.77% 1.65 3.74 3.69% 1.94 4.49 4.53%  GMAN 1.34 2.91 2.86% 1.63 3.76 3.68% 1.86 4.32 4.37%  DGCRN 1.28 2.69 2.66% 1.59 3.63 3.55% 1.89 4.42 4.43%  D2STGNN 1.24∗ 2.60∗ 2.58%∗ 1.55∗ 3.52∗ 3.49%∗ 1.85∗ 4.30∗ 4.37%'
    RAW_HORIZONS_METR_LA = 'HA 4.79 10.00 11.70% 5.47 11.45 13.50% 6.99 13.89 17.54%  VAR 4.42 7.80 13.00% 5.41 9.13 12.70% 6.52 10.11 15.80%  SVR 3.39 8.45 9.30% 5.05 10.87 12.10% 6.72 13.76 16.70%  FC-LSTM 3.44 6.30 9.60% 3.77 7.23 10.09% 4.37 8.69 14.00%  DCRNN 2.77 5.38 7.30% 3.15 6.45 8.80% 3.60 7.60 10.50%  STGCN 2.88 5.74 7.62% 3.47 7.24 9.57% 4.59 9.40 12.70%  Graph WaveNet 2.69 5.15 6.90% 3.07 6.22 8.37% 3.53 7.37 10.01%  ASTGCN 4.86 9.27 9.21% 5.43 10.61 10.13% 6.51 12.52 11.64%  STSGCN 3.31 7.62 8.06% 4.13 9.77 10.29% 5.06 11.66 12.91%  MTGNN 2.69 5.18 6.88% 3.05 6.17 8.19% 3.49 7.23 9.87%  GMAN 2.80 5.55 7.41% 3.12 6.49 8.73% 3.44 7.35 10.07%  DGCRN 2.62 5.01 6.63% 2.99 6.05 8.02% 3.44 7.19 9.73%  D2STGNN 2.56∗ 4.88∗ 6.48%∗ 2.90∗ 5.89∗ 7.78%∗ 3.35∗ 7.03∗ 9.40%∗'
    RAW_HORIZONS = [RAW_HORIZONS_PEMS04, RAW_HORIZONS_PEMS08, RAW_HORIZONS_PEMSBAY, RAW_HORIZONS_METR_LA]
    '''文章 Table 3 文本格式 '''

class ParserLargeST:
    ''' LargeST 论文：LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting 解析器'''
    @staticmethod
    def parse_table2(input_str: str, dataset: str, split: str = "6:2:2", inLen: int = 12, outLen: int = 12, tags: str = "survey", source: str = '') -> Results:
        ''' 解析 LargeST 风格汇总表2的多行或单行长文本；支持一行内多模型连续出现。
        每个模型块形如：Model Param MAE RMSE MAPE MAE RMSE MAPE MAE RMSE MAPE MAE RMSE MAPE
        分别对应 Horizon 3、6、12 与 Average（Average 记为 horizon=-1）'''
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
        def clean_token(token: str) -> str:
            return re.sub(r'[^\d.%+-]', '', token)
        i = 0
        while i < n:
            # 解析 model 名称（直到遇到 param）
            model_tokens = []
            while i < n and not is_param(tokens[i]):
                model_tokens.append(tokens[i])
                i += 1
            if i >= n:
                break
            if len(model_tokens) == 0:
                # 异常：没有模型名，跳过该 token
                i += 1
                continue
            model_name = ' '.join(model_tokens)
            # 跳过 param token
            i += 1
            # 收集 12 个指标（清洗百分号）
            metrics = []
            while i < n and len(metrics) < 12:
                ct = clean_token(tokens[i])
                if ct:
                    metrics.append(ct)
                i += 1
            if len(metrics) < 12:
                # 指标不足，停止
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

    TABLE2_SD = r'''HL – 33.61 50.97 20.77% 57.80 84.92 37.73% 101.74 140.14 76.84% 60.79 87.40 41.88% LSTM 98K 19.03 30.53 11.81% 25.84 40.87 16.44% 37.63 59.07 25.45% 26.44 41.73 17.20% DCRNN 373K 17.14 27.47 11.12% 20.99 33.29 13.95% 26.99 42.86 18.67% 21.03 33.37 14.13% AGCRN 761K 15.71 27.85 11.48% 18.06 31.51 13.06% 21.86 39.44 16.52% 18.09 32.01 13.28% STGCN 508K 17.45 29.99 12.42% 19.55 33.69 13.68% 23.21 41.23 16.32% 19.67 34.14 13.86% GWNET 311K 15.24 25.13 9.86% 17.74 29.51 11.70% 21.56 36.82 15.13% 17.74 29.62 11.88% ASTGCN 2.2M 19.56 31.33 12.18% 24.13 37.95 15.38% 30.96 49.17 21.98% 23.70 37.63 15.65% STTN 114K 16.22 26.22 10.63% 18.76 30.98 12.80% 22.62 39.09 16.14% 18.69 31.11 12.82% STGODE 729K 16.75 28.04 11.00% 19.71 33.56 13.16% 23.67 42.12 16.58% 19.55 33.57 13.22% DSTAGNN 3.9M 18.13 28.96 11.38% 21.71 34.44 13.93% 27.51 43.95 19.34% 21.82 34.68 14.40% DGCRN 243K 15.34 25.35 10.01% 18.05 30.06 11.90% 22.06 37.51 15.27% 18.02 30.09 12.07% D2STGNN 406K 14.92 24.95 9.56% 17.52 29.24 11.36% 22.62 37.14 14.86% 17.85 29.51 11.54%'''
    TABLE2_GBA = r'''HL – 32.57 48.42 22.78% 53.79 77.08 43.01% 92.64 126.22 92.85% 56.44 79.82 48.87% LSTM 98K 20.38 33.34 15.47% 27.56 43.57 23.52% 39.03 60.59 37.48% 27.96 44.21 24.48% DCRNN 373K 18.71 30.36 14.72% 23.06 36.16 20.45% 29.85 46.06 29.93% 23.13 36.35 20.84% AGCRN 777K 18.31 30.24 14.27% 21.27 34.72 16.89% 24.85 40.18 20.80% 21.01 34.25 16.90% STGCN 1.3M 21.05 34.51 16.42% 23.63 38.92 18.35% 26.87 44.45 21.92% 23.42 38.57 18.46% GWNET 344K 17.85 29.12 13.92% 21.11 33.69 17.79% 25.58 40.19 23.48% 20.91 33.41 17.66% ASTGCN 22.3M 21.46 33.86 17.24% 26.96 41.38 24.22% 34.29 52.44 32.53% 26.47 40.99 23.65% STTN 218K 18.25 29.64 14.05% 21.06 33.87 17.03% 25.29 40.58 21.20% 20.97 33.78 16.84% STGODE 788K 18.84 30.51 15.43% 22.04 35.61 18.42% 26.22 42.90 22.83% 21.79 35.37 18.26% DSTAGNN 26.9M 19.73 31.39 15.42% 24.21 37.70 20.99% 30.12 46.40 28.16% 23.82 37.29 20.16% DGCRN 374K 18.02 29.49 14.13% 21.08 34.03 16.94% 25.25 40.63 21.15% 20.91 33.83 16.88% D2STGNN 446K 17.54 28.94 12.12% 20.92 33.92 14.89% 25.48 40.99 19.83% 20.71 33.65 15.04%'''
    TABLE2_GLA = r'''HL – 33.66 50.91 19.16% 56.88 83.54 34.85% 98.45 137.52 71.14% 59.58 86.19 38.76% LSTM 98K 20.02 32.41 11.36% 27.73 44.05 16.49% 39.55 61.65 25.68% 28.05 44.38 17.23% DCRNN 373K 18.41 29.23 10.94% 23.16 36.15 14.14% 30.26 46.85 19.68% 23.17 36.19 14.40% AGCRN 792K 17.27 29.70 10.78% 20.38 34.82 12.70% 24.59 42.59 16.03% 20.25 34.84 12.87% STGCN 2.1M 19.86 34.10 12.40% 22.75 38.91 14.11% 26.70 45.78 17.00% 22.64 38.81 14.17% GWNET 374K 17.28 27.68 10.18% 21.31 33.70 13.02% 26.99 42.51 17.64% 21.20 33.58 13.18% ASTGCN 59.1M 21.89 34.17 13.29% 29.54 45.01 19.36% 39.02 58.81 29.23% 28.99 44.33 19.62% STGODE 841K 18.10 30.02 11.18% 21.71 36.46 13.64% 26.45 45.09 17.60% 21.49 36.14 13.72% DSTAGNN 66.3M 19.49 31.08 11.50% 24.27 38.43 15.24% 30.92 48.52 20.45% 24.13 38.15 15.07%'''
    TABLE2_CA = r'''HL – 30.72 46.96 20.43% 51.56 76.48 37.22% 89.31 125.71 76.80% 54.10 78.97 41.61% LSTM 98K 19.04 31.28 13.19% 26.49 42.63 19.57% 38.22 60.29 30.28% 26.89 43.11 20.16% DCRNN 373K 17.55 28.21 12.68% 21.79 34.27 16.67% 28.56 44.34 23.84% 21.87 34.41 17.06% STGCN 4.5M 18.99 32.37 14.84% 21.37 36.46 16.27% 24.94 42.59 19.74% 21.33 36.39 16.53% GWNET 469K 17.14 27.81 12.62% 21.68 34.16 17.14% 28.58 44.13 24.24% 21.72 34.20 17.40% STGODE 1.0M 17.57 29.91 13.91% 20.98 36.62 16.88% 25.46 45.99 21.00% 20.77 36.60 16.80%'''
    TABLE2 = {'SD': TABLE2_SD, 'GBA': TABLE2_GBA, 'GLA': TABLE2_GLA, 'CA': TABLE2_CA}

if __name__ == "__main__":
    dataset_names = ['PEMS04', 'PEMS08', 'PEMS-BAY', 'METR-LA']
    for i, raw_horizon in enumerate(ParserD2STGNN.RAW_HORIZONS):
        dataset_name = dataset_names[i]
        csv_output = ParserD2STGNN.parse_D2STGNN_horizon(raw_horizon, dataset_name)
        output_filename = f'd:\\_lr580_desktop\\research\\llm_tfp\\utils\\baselines\\horizons_D2STGNN_{dataset_name}.csv'
        with open(output_filename, 'w', newline='', encoding='utf-8') as f:
            f.write(csv_output)
        print(f"结果已保存到 {dataset_name}_horizon_results.csv")
    
    print("\n所有数据集处理完成！")