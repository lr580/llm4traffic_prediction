import datetime
import string
import random
def prints_and_returns(verbose=True, **kwargs):
    if verbose:
        strs = []
        for name, val in kwargs.items():
            strs.append(f'{name}:{val:.4f}')
        print(', '.join(strs))
    return kwargs.values()

def date2str(datetime:datetime.datetime):
    return datetime.strftime('%Y-%m-%d %H:%M:%S')

ASCII_BIN = string.ascii_letters + string.digits
def randomStr(n:int):
    return ''.join(random.choices(ASCII_BIN, k=n))
    
if __name__ == '__main__':
    import torch
    a,b,c = prints_and_returns(mae=1., mape=2/3, rmse=torch.tensor(3.))
    print(a,b,c)