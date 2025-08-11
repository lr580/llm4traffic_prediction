import datetime
import string
import random
import json
import numpy as np
from typing import Any
def prints_and_returns(verbose=True, **kwargs):
    if verbose:
        strs = []
        for name, val in kwargs.items():
            strs.append(f'{name}:{val:.4f}')
        print(', '.join(strs))
    return kwargs.values()

def date2str(datetime:datetime.datetime, format='general'):
    if format == 'general':
        return datetime.strftime('%Y-%m-%d %H:%M:%S')
    elif format == 'file':
        return datetime.strftime('%Y%m%d%H%M%S')
    
def now2str(format='file'):
    return date2str(datetime.datetime.now(), format)

ASCII_BIN = string.ascii_letters + string.digits
def randomStr(n:int):
    return ''.join(random.choices(ASCII_BIN, k=n))

class CompactListEncoder(json.JSONEncoder): # 无用
    def iterencode(self, o: Any, _one_shot: bool = False):
        if isinstance(o, list) and not any(isinstance(x, (dict, list)) for x in o):
            return super().iterencode(o, _one_shot=True)
        return super().iterencode(o, _one_shot=_one_shot)

class NumpyEncoder(CompactListEncoder): # (json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
if __name__ == '__main__':
    import torch
    a,b,c = prints_and_returns(mae=1., mape=2/3, rmse=torch.tensor(3.))
    print(a,b,c)