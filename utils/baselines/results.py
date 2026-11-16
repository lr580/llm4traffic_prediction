'''整理汇总基准模型的实验结果；数据在同目录同名baselineResults.csv下，目前只收录了部分结果，大部分是新论文的；
出表用法参见 main.py；筛选、视图等示例参见 unittest/baselineResults_test.py 和 baselineResults_test2.py'''
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from ..datasets.data import EvaluateResult
from ..common import pdDataFrame2str, bold_column_min_in_latex_table
from typing import List
from copy import deepcopy
import pandas as pd
from pathlib import Path

@dataclass
class Paper:
    '''模型对应的论文条目：模型名、引用键与元数据；快照查不到的字段留空。

    纯数据类，由 citations.CitationHandler.paper() 查表构建，本模块不依赖 citations。
    '''
    model: str
    '''模型名'''

    cite: str = ''
    '''引用键，如 'T-28'；无引用键的基线模型（如 HA）为空'''

    venue: str = ''
    '''venue 缩写（CCF 目录口径），如 'IJCAI'；未知为空'''

    year: int | None = None
    '''发表年份；未知为 None'''

    ccf: str = ''
    '''CCF 等级，如 'A'；未知为空'''

    def unknown(self) -> bool:
        '''识别 HA / OLS 等不来自任何论文的对照基线（三字段皆空）：筛选时默认剔除，include_unknown=True 时保留作对照。'''
        return not (self.venue or self.year or self.ccf)

    def display(self, withCite: bool = True) -> str:
        '''显示名：Model (VENUE YEAR) \\cite{T-xxx}；标注与引用各自缺失时跳过。

        例：Paper('STGCN', 'T-28', 'IJCAI', 2018, 'A').display() -> 'STGCN (IJCAI 2018) \\cite{T-28}'
            Paper('HA').display()                            -> 'HA'
        '''
        name = f'{self.model} ({self.venue} {self.year})' if self.venue and self.year else self.model
        return f'{name} \\cite{{{self.cite}}}' if withCite and self.cite else name

    def display_cite(self) -> str:
        '''cite 风格显示名：Model \\cite{T-xxx}；无引用键原样返回。

        与 display 的区别：不加 (VENUE YEAR) 标注，只注入引用。
        '''
        return f'{self.model} \\cite{{{self.cite}}}' if self.cite else self.model

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
    '''输出序列长度 (最终)'''

    horizon: int = -1
    ''' 预测区间统计方式：-1表示平均；3/6/12表示不同的 horizons (参见论文) '''
    
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
    
    def model_satisfy(self, models: list):
        return len(models) == 0 or (self.model in models)
    
    def horizon_satisfy(self, horizons: int):
        return (len(horizons) == 0 and self.horizon == -1) or (self.horizon in horizons)
    
    def paper_satisfy(self, paper: Paper, ccf: list, venues: list, year_range: tuple | None,
                      include_unknown: bool = False):
        '''论文元数据条件；不带条件恒 True，三字段皆空按 include_unknown 决定。'''
        if not (ccf or venues or year_range is not None):
            return True
        if paper.unknown():
            return include_unknown
        if ccf and paper.ccf not in ccf:
            return False
        if venues and paper.venue not in venues:
            return False
        if year_range is not None:
            return isinstance(paper.year, int) and year_range[0] <= paper.year <= year_range[1]
        return True
    
@dataclass
class Results:
    results: List[Result] = field(default_factory=list)
    '''结果'''
    
    def __iter__(self):
        return iter(self.results)
    
    def __len__(self):
        return len(self.results)
    
    def append(self, result: Result):
        self.results.append(result)

    def deduplicate(self):
        """去重结果集，保留插入顺序。 usage: unittest/baselinesUnique_test.py """
        unique_results = []
        for result in self.results:
            if result not in unique_results:
                unique_results.append(result)
        self.results = unique_results
        return self

    def to_dataframe(self, **kwargs) -> pd.DataFrame:
        data = []
        for result in self.results:
            row_data = asdict(result)
            eval_data = asdict(result.evaluateResult)
            row_data.update(eval_data)
            del row_data['evaluateResult']
            data.append(row_data)
        return pd.DataFrame(data, columns=Results.ORDERS, **kwargs)
    
    ORDERS = ['model', 'dataset', 'mae', 'mape', 'rmse', 'split', 'inLen', 'outLen', 'tags', 'horizon', 'source']
    
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
                'horizon': int(row['horizon']) if not pd.isna(row['horizon']) else -1
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
    
    @staticmethod
    def get_baseline_results() -> Results:
        current_file = Path(__file__).resolve()
        current_dir = current_file.parent
        csv_path = current_dir / 'baselineResults.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"找不到文件: {csv_path}")
        return Results.from_csv(str(csv_path))
    
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
    def merge(*results: Results, remove_duplicates: bool = True) -> Results:
        merged_results = Results(results=[])
        for result in results:
            merged_results.results.extend(result.results)
        if remove_duplicates:
            df = merged_results.to_dataframe()
            merged_results = Results.from_dataframe(df.drop_duplicates())
        return merged_results
    
    def flit(self, datasets=[], split='6:2:2', tags='', inLen=-1, outLen=-1, models=[], inner=True, horizons=[],
             ccf=None, year_range=None, venues=None, include_unknown=False, paper_meta={}):
        '''唯一的筛选入口：实验条件之外，也可按论文元数据 ccf / venues / year_range 筛。

        论文条件由 Result.paper_satisfy 判定；元数据由调用方经 paper_meta 传入
        （{model: Paper}，缺键视为全空的 Paper），本模块不查表、不依赖 citations。
        带论文条件但不传 paper_meta 时各模型元数据视为空，默认会被全部剔除；
        不带论文条件时不需要它，行为与旧版 flit 完全一致，
        即仍套用默认实验条件（split='6:2:2'、horizons=[] 只看 horizon=-1）。
        inner=False 返回深拷贝的新 Results，render 就地改写 model 不会串改原集合。
        '''
        results = []
        for result in self.results:
            ok = True
            if not(result.dataset_satisfy(datasets) and result.split == split and result.tags_satisfy(tags)):
                ok = False
            if not(result.input_satisfy(inLen) and result.output_satisfy(outLen)):
                ok = False
            if not (result.model_satisfy(models) and result.horizon_satisfy(horizons)):
                ok = False
            if not result.paper_satisfy(paper_meta.get(result.model, Paper(result.model)),
                                        ccf, venues, year_range, include_unknown):
                ok = False
            if ok:
                results.append(result)
        if inner:
            self.results = results
            return self
        else:
            return Results(results=[deepcopy(result) for result in results])

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
    
    def multi_index_view(self, group_by: str, groups: list = None, metrics: list = None) -> pd.DataFrame:
        """ 通用的多级索引视图函数
        
        Args:
            group_by: 分组字段，如 'dataset', 'horizon'
            groups: 指定的分组值列表，如果为None则使用所有存在的值
            metrics: 指定的指标列表，如果为None则使用 ['mae', 'mape', 'rmse']
            
        Returns:
            pd.DataFrame: 多级列索引的DataFrame"""
        df = self.to_dataframe()
        if metrics is None:
            metrics = ['mae', 'mape', 'rmse']
        if groups is not None:
            df = df[df[group_by].isin(groups)]
        melted = df.melt(
            id_vars=['model', group_by],
            value_vars=metrics,
            var_name='metric',
            value_name='value'
        )
        result = melted.pivot_table(
            index='model',
            columns=[group_by, 'metric'],
            values='value',
            aggfunc='mean'
        )
        result.columns.names = [group_by, 'metric']
        if groups is not None:
            available_groups = [g for g in groups if g in result.columns.get_level_values(0)]
            result = result.reindex(columns=available_groups, level=0)
        return result

    def horizon_view(self, horizons: list = [3, 6, 12]) -> pd.DataFrame:
        """根据给定的horizons参数生成多级表头的DataFrame。
        索引是model，第一级列是不同horizons，第二级列为各个horizon的mae、mape、rmse指标。"""
        return self.multi_index_view(group_by='horizon', groups=horizons)

    def dataset_view(self, datasets: list = [f'PEMS0{x}' for x in '3478']) -> pd.DataFrame:
        """根据给定的datasets参数生成多级表头的DataFrame。
        索引是model，第一级列是不同数据集，第二级列为各个数据集的mae、mape、rmse指标。"""
        return self.multi_index_view(group_by='dataset', groups=datasets)

    def to_latex_table(self, datasets: list = None, sort_by=None,
                       bold_min: bool = True, nan: str = '-', **kwargs) -> str:
        """把结果整理成 LaTeX 表格行（不含表头/表尾），index 列即模型名。

        datasets/sort_by 控制取哪个数据集视图与排序；bold_min 加粗每列最小值；
        kwargs 透传给 pdDataFrame2str（如 suffix、cell_decorator）。用法见 main.py。
        """
        df = self.dataset_view(datasets) if datasets else self.dataset_view()
        if sort_by is not None:
            df = df.sort_values(by=sort_by, ascending=False)
        kwargs.setdefault('prefix', '')
        tex = pdDataFrame2str(df, **kwargs)
        if bold_min:
            tex = bold_column_min_in_latex_table(tex)
        return tex.replace('nan', nan) if nan else tex
    
