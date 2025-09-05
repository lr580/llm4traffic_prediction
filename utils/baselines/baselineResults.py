'''整理汇总基准模型的实验结果；数据在同目录同名baselineResults.csv下，目前只收录了部分结果，大部分是新论文的'''
from __future__ import annotations
from dataclasses import dataclass, asdict
from ..datasets.data import EvaluateResult
from typing import List
import pandas as pd

@dataclass
class Result:
    '''包含了构成基准模型结果；实际上也可以是自己跑的模型的结果'''
    model: str
    '''模型名字'''
    
    dataset: str
    '''数据集名字'''
    
    evaluateResult: EvaluateResult
    '''评估结果；目前只考虑这三个指标'''
    
    split: str = '6:2:2'
    '''数据集划分(训练、验证测试比例)，如"6:2:2"'''
    
    inLen: int = 12
    '''输入序列长度'''
    
    outLen: int = 12
    '''输出序列长度'''
    
    source: str = ''
    '''论文来源等备注信息，可选'''
    
    tags: str = ''
    ''' 标签，便于分类，可选；如 "BasicTS, self" \n
    属性逗号分隔，其中，self 表示自己的实验结果, BasicTS 表示用该框架运行的结果，survey 表示从论文中整理的结果，
    12 表示并非使用 average 计算的评价指标(而是只看第12个未来区间的结果)
    '''
    
    def __post_init__(self):
        if pd.isna(self.source):
            self.source = ''
        if pd.isna(self.tags):
            self.tags = ''
            
    def tags_satisfy(self, subtags: str):
        need = subtags.split(',')
        if len(need) == 1 and not need[0]:
            return True
        ownedTags = self.tags.split(',')
        return all([tag in ownedTags for tag in need])
    
    def dataset_satisfy(self, datasets: list):
        return len(datasets) == 0 or (self.dataset in datasets)
    
    def input_satisfy(self, inLen: int):
        return inLen <= 0 or inLen == self.inLen
    
    def output_satisfy(self, outLen: int):
        return outLen <= 0 or outLen == self.outLen
    
    def modell_name_satisfy(self, models: list):
        return len(models) == 0 or (self.model in models)
    
@dataclass
class Results:
    results: List[Result]
    '''结果'''
    
    def __iter__(self):
        return iter(self.results)
    
    def __len__(self):
        return len(self.results)

    def to_dataframe(self, **kwargs) -> pd.DataFrame:
        data = []
        for result in self.results:
            row_data = asdict(result)
            eval_data = asdict(result.evaluateResult)
            row_data.update(eval_data)
            del row_data['evaluateResult']
            data.append(row_data)
        return pd.DataFrame(data, **kwargs)
    
    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> List[Result]:
        # 可以设计为动态的，让 Result 的字段可以动态变化而不必修改该函数，略。
        results = []
        for _, row in df.iterrows():
            eval_data = {
                'mae': float(row['mae']),
                'mape': float(row['mape']),
                'rmse': float(row['rmse'])
            }
            evaluate_result = EvaluateResult(**eval_data)
            baseline_data = {
                'model': row['model'],
                'dataset': row['dataset'],
                'split': row['split'],
                'inLen': int(row['inLen']),
                'outLen': int(row['outLen']),
                'evaluateResult': evaluate_result,
                'source': row['source'],
                "tags": row['tags'],
            }
            results.append(Result(**baseline_data))
        return Results(results=results)
    
    def to_csv(self, filepath: str, **kwargs):
        ''' 感觉不如 self.sort().to_csv(...) 好用 '''
        self.to_dataframe().to_csv(filepath, index=False, **kwargs)
    
    @staticmethod
    def from_csv(filepath: str, **kwargs) -> Results:
        df = pd.read_csv(filepath, **kwargs)
        return Results.from_dataframe(df)
    
    def printSOTA(self):
        df = self.to_dataframe() # 数据不多，耗不了几个复杂度；懒得缓存 df 避免一致性问题
        grouped = df.groupby('dataset')
        for dataset, group in grouped: # 懒得泛化到任意指标了，需要再改吧
            print(f"数据集: {dataset}")
            mae_best = group.loc[group['mae'].idxmin()]
            mape_best = group.loc[group['mape'].idxmin()]
            rmse_best = group.loc[group['rmse'].idxmin()]
            print(f"  MAE 最优: {mae_best['model']} (值为 {mae_best['mae']:.4f})")
            print(f"  MAPE最优: {mape_best['model']} (值为 {mape_best['mape']:.4f})")
            print(f"  RMSE最优: {rmse_best['model']} (值为 {rmse_best['rmse']:.4f})")
            print()
            
    @staticmethod
    def merge(*results: Results) -> Results:
        merged_results = Results(results=[])
        for result in results:
            merged_results.results.extend(result.results)
        return merged_results
    
    def flit(self, datasets=[], split='6:2:2', tags='', inLen=-1, outLen=-1, models=[], inner=True):
        results = []
        for result in self.results:
            ok = True
            if not(result.dataset_satisfy(datasets) and result.split == split and result.tags_satisfy(tags)):
                ok = False
            if not(result.input_satisfy(inLen) and result.output_satisfy(outLen)):
                ok = False
            if not result.modell_name_satisfy(models):
                ok = False
            if ok:
                results.append(result)
        if inner:
            self.results = results
            return self
        else:
            return Results(results=results)
        
    def sort(self, key='mape') -> pd.DataFrame:
        '''注意返回 DataFrame'''
        return self.to_dataframe().sort_values(by=key)
    
    def rank(self, key='mape') -> pd.DataFrame:
        df = self.to_dataframe()
        best_rmse = df.groupby(['model', 'dataset'])[key].min().reset_index()
        pivot_df = best_rmse.pivot(index='model', columns='dataset', values=key)
        pivot_df[f'Avg {key}'] = pivot_df.mean(axis=1, skipna=True)
        ranked_df = pivot_df.sort_values(by=f'Avg {key}', ascending=True)      
        return ranked_df