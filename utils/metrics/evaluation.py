from .mae import masked_mae
from .mape import masked_mape
from .rmse import masked_rmse
from ..common import prints_and_returns
import numpy as np
import torch

def transfer_format(y_pred, y_true):
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred).float()
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true).float()
    return y_pred, y_true

def evaluate_average(y_pred, y_true, verbose=True):
    y_pred, y_true = transfer_format(y_pred, y_true)
    mae = masked_mae(y_pred, y_true)
    mape = masked_mape(y_pred, y_true)
    rmse = masked_rmse(y_pred, y_true)
    return prints_and_returns(verbose, MAE=mae, MAPE=mape, RMSE=rmse)
