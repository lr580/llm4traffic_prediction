# code adapted from https://github.com/GestaltCogTeam/BasicTS
from .mae import masked_mae
from .mape import masked_mape
from .rmse import masked_rmse
from .evaluation import evaluate_average

ALL_METRICS = {
            'MAE': masked_mae,
            'RMSE': masked_rmse,
            'MAPE': masked_mape,
            }

__all__ = [
    'masked_mae',
    'masked_rmse',
    'masked_mape',
    'evaluate_average',
    'ALL_METRICS'
]