import numpy as np
import torch
import sys, os
import tqdm
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.datasets import PEMSDataset, PEMS_HAstat
from utils.metrics import masked_mae, masked_mape, masked_rmse
def main(x):
    dataset = PEMSDataset(x)
    ha = PEMS_HAstat(dataset)
    test_data = dataset.get_test()
    y_pred = np.zeros(test_data.shape[:2])
    for i in range(test_data.shape[0]):
        for j in range(dataset.n):
            y_pred[i][j] = ha.calc(j, test_data[i, j, 1], test_data[i, j, 2])
    y_pred = torch.tensor(y_pred).float()
    y_true = torch.tensor(np.squeeze(test_data[:,:,0])).float()
    print(f'Evaluation results {x}:')
    print(f'MAE: {masked_mae(y_pred, y_true).item()}, RMSE: {masked_rmse(y_pred, y_true).item()}, MAPE: {masked_mape(y_pred, y_true).item()}')
    
for x in tqdm.tqdm('8347'):
    main(x)