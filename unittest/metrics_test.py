import sys
import os
rootpath = os.path.abspath(os.path.join(os.getcwd()))
# rootpath = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(rootpath)
# print(sys.path)

from utils.metrics import masked_mae, masked_mape, masked_rmse
import torch
y_pred = torch.tensor([1., 2, 3, 4, 5])
y_true = torch.tensor([1., 2, 3, 6, 4])
print(masked_mae(y_pred, y_true)) # tensor(0.6000)
print(masked_rmse(y_pred, y_true))
print(masked_mape(y_pred, y_true))
'''
MAE = (0+0+0+2+1)/5
MAPE = (0+0+0+2/6+1/4)/5
RMSE = ( (0+0+0+4+1)/5 )**0.5
'''

mae = torch.abs(y_pred - y_true).mean()
print("MAE:", mae.item())  # 0.6
mape = (torch.abs((y_true - y_pred) / y_true).mean()) * 100
print("MAPE:", mape.item())  # ≈11.666%
rmse = torch.sqrt(((y_pred - y_true) ** 2).mean())
print("RMSE:", rmse.item())  # 1.0

y_pred = torch.tensor([[[1.,2,3],[4,5,6],[7,8,9]],[[1.,2,3],[4,5,6],[7,8,9]]])
y_true = torch.tensor([[[1.,2,3],[4,5,6],[7,8,9]],[[1.,2,3],[4,5,6],[7,8,9+18]]])
print(masked_mae(y_pred, y_true))