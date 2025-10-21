# 所有输出与格式化相关通用函数
import datetime
import string
import random
import numpy as np
from typing import cast, Optional, Callable
from pandas import Series, DataFrame
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

def non_decorate(s:str):
    return s

def latex_add_decorator(s: str):
    return f"\\added{{{s}}}"
    
def pdRow2str(row: Series, prefix: str = "& ", separator: str = " & ", suffix: str = r" \\ \hline", include_index: bool = False, cell_decorator: Callable[[str], str] = non_decorate) -> str:
    """格式化pandas Series (row)为自定义字符串"""
    values = row.astype(str).tolist()
    values = list(map(cell_decorator, values))
    if include_index:
        values.insert(0, cell_decorator(str(row.name)))
    joined_values = separator.join(values)
    return f"{prefix}{joined_values}{suffix}"

def pdDataFrame2str(df: DataFrame, prefix: str = "& ", separator: str = " & ", suffix: str = r" \\ \hline", line_separator: str = "\n", include_index: bool = True, include_header: bool = False, header_prefix: Optional[str] = None, header_separator: Optional[str] = None, header_suffix: Optional[str] = None, index_header: str = "Index", cell_decorator: Callable[[str], str] = non_decorate) -> str:
    """格式化整个DataFrame的每一行数据 返回包含所有格式化行的字符串"""
    formatted_rows = []
    if include_header:
        header_values = df.columns.astype(str).tolist()
        if include_index:
            header_values.insert(0, index_header)
        header_str = header_separator.join(header_values)
        formatted_rows.append(f"{header_prefix}{header_str}{header_suffix}")
    for i in range(len(df)):
        row = df.iloc[i]
        formatted_rows.append(pdRow2str(row, prefix, separator, suffix, include_index, cell_decorator))
    return line_separator.join(formatted_rows)

if __name__ == '__main__':
    import torch
    a,b,c = prints_and_returns(mae=1., mape=2/3, rmse=torch.tensor(3.))
    print(a,b,c)