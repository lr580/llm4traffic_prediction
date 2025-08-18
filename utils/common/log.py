import datetime
import string
import random
import numpy as np
from typing import cast
def prints_and_returns(verbose=True, **kwargs):
    if verbose:
        strs = []
        for name, val in kwargs.items():
            strs.append(f'{name}:{val:.4f}')
        print(', '.join(strs))
    return kwargs.values()

def vec2str(X:np.ndarray):
    strs = []
    for x in X:
        if x == int(x):
            strs.append('%d' % x)
        else:
            strs.append('%.4f' % x)
    return ', '.join(strs)

def date2str(datetime:datetime.datetime, format='general'):
    if format == 'general':
        return datetime.strftime('%Y-%m-%d %H:%M:%S')
    elif format == 'file':
        return datetime.strftime('%Y%m%d%H%M%S')
    
def str2date(date_string: str, format='general') -> datetime.datetime:
    if format == 'general':
        res = datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    elif format == 'file':
        res = datetime.datetime.strptime(date_string, '%Y%m%d%H%M%S')
    return cast('datetime.datetime', res)
    
def now2str(format='file'):
    return date2str(datetime.datetime.now(), format)

ASCII_BIN = string.ascii_letters + string.digits
def randomStr(n:int):
    return ''.join(random.choices(ASCII_BIN, k=n))
    
if __name__ == '__main__':
    import torch
    a,b,c = prints_and_returns(mae=1., mape=2/3, rmse=torch.tensor(3.))
    print(a,b,c)