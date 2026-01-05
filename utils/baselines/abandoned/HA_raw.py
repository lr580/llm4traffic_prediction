# 使用历史平均值预测，检验结果是否与一般模型在同一个数量级，验证原始数据量纲
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.metrics import masked_mae, masked_mape, masked_rmse
import torch
def main(filepath):
    # data prepare
    data = np.load(filepath)['data']
    data = np.squeeze(data[:, :, 0])
    
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    T = data.shape[0]
    train_end = int(T * train_ratio)
    val_end = train_end + int(T * val_ratio)
    train_data = data[:train_end, :] 
    val_data = data[train_end:val_end, :]
    test_data = data[val_end:, :] 
    
    # sub = train_data[:5,:5]
    # for r in range(5):
    #     for c in range(5):
    #         print(f'{sub[r,c]:.2f}', end=' ')
    #     print()
    # print()
    
    TOD = 1 # time of day
    N = data.shape[1]
    N_sam = N
    samples = [[[] for i in range(N_sam)] for j in range(TOD)]
    for i in range(train_data.shape[0]):
        for j in range(N):
            samples[i % TOD][j % N_sam].append(train_data[i, j])
    avg = [[np.mean(samples[i][j]) for j in range(N_sam)] for i in range(TOD)]
    y_pred = np.zeros_like(test_data)
    for i in range(test_data.shape[0]):
        for j in range(N):
            y_pred[i, j] = avg[(i+train_end+val_end) % TOD][j % N_sam]
            
    for i in range(5):
        for j in range(5):
            print(f'{y_pred[i,j]:.2f} : {test_data[i,j]:.2f}', end='    ')
        print()
    print()
    # print(y_pred.dtype, test_data.dtype)
    y_pred = torch.tensor(y_pred).float()
    y_true = torch.tensor(test_data).float()
    print(f'MAE: {masked_mae(y_pred, y_true).item()}, RMSE: {masked_rmse(y_pred, y_true).item()}, MAPE: {masked_mape(y_pred, y_true).item()}')

for x in '3':
    main(f'data/raw/PEMS0{x}/PEMS0{x}.npz')