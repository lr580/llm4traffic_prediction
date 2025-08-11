from dataclasses import dataclass, asdict, is_dataclass
import numpy as np
from numpy.typing import NDArray
import json
import os
from ..common import NumpyEncoder
from ..metrics import evaluate_average
# from typing import Union
@dataclass
class SingleInput:
    X : NDArray[np.float32]
    ''' 输入数据(历史时间区间数据) '''
    
    i : int 
    ''' 时间区间编号 '''
    
    j : int
    ''' 空间点编号 '''
    
    def __post_init__(self):
        if isinstance(self.X, list):
            self.X = np.array(self.X, dtype=np.float32)

@dataclass
class EvaluateResult:
    mae : float
    '''Mean Absolute Error (MAE)'''
    
    mape: float
    '''Mean Absolute Percentage Error (MAPE)'''
    
    rmse: float
    '''Root Mean Squared Error (RMSE) '''

@dataclass
class SingleData:
    input : SingleInput
    ''' 输入数据(历史时间区间数据) '''
    
    y_true : NDArray[np.float32] = None
    ''' 答案数据(未来时间区间数据) '''

    y_pred: NDArray[np.float32] = None
    ''' 输出数据(未来时间区间数据) '''
    
    result : EvaluateResult = None
    ''' 评估结果 '''
    
    def __post_init__(self):
        if isinstance(self.input, dict):
            self.input = SingleInput(**self.input)
        if isinstance(self.result, dict):
            self.result = EvaluateResult(**self.result)
        if isinstance(self.y_true, list):
            self.y_true = np.array(self.y_true, dtype=np.float32)
        if isinstance(self.y_pred, list):
            self.y_pred = np.array(self.y_pred, dtype=np.float32)
            
    def evaluate(self):
        self.result = EvaluateResult(*evaluate_average(self.y_pred, self.y_true))
    
@dataclass
class DataList:
    data : list[SingleData]
    ''' 数据列表 '''
    
    totalResult: EvaluateResult = None
    ''' 整体评估结果 '''
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def evaluate(self):
        y_preds, y_trues = [], []
        for data in self.data:
            y_preds.append(data.y_pred)
            y_trues.append(data.y_true)
        y_preds = np.stack(y_preds, axis=0)
        y_trues = np.stack(y_trues, axis=0)
        self.totalResult = EvaluateResult(*evaluate_average(y_preds, y_trues))
    
    def save(self, path:str): # 用 json 而不是 pickle 等原因：便于人阅读(可以当日志看)；且数据规模不大
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, cls=NumpyEncoder)

    @classmethod
    def load(cls, path:str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
    
    def __post_init__(self):
        self.data = [SingleData(**item) if not is_dataclass(item) else item for item in self.data]
        if self.totalResult and isinstance(self.totalResult, dict):
            self.totalResult = EvaluateResult(**self.totalResult)
    
    
class EnhancedNumpyEncoder(json.JSONEncoder): # not used
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, np.ndarray):
            return {
                '__ndarray__': True,
                'dtype': str(obj.dtype),
                'data': obj.tolist()
            }
        return super().default(obj)