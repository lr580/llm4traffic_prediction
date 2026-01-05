# 使用历史平均值预测，检验结果是否与一般模型在同一个数量级，验证原始数据量纲
'''更低耦合的实现： utils/unittest/HA_test.py'''
import numpy as np
import sys, os
# import pickle
import json
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.metrics import masked_mae, masked_mape, masked_rmse
import torch
def main(filepath, shape):
    # data prepare
    # with open(filepath, 'rb') as f:
    #     data = pickle.load(f)['processed_data']
    data = np.memmap(filepath, dtype=np.float32, mode='r', shape=shape)
    
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    T = data.shape[0]
    train_end = int(T * train_ratio)
    val_end = train_end + int(T * val_ratio)
    train_data = data[:train_end, :, :] 
    val_data = data[train_end:val_end, :, :]
    test_data = data[val_end:, :, :] 
    
    # train
    # TOD = np.unique(data[:, :, 1]).shape[0] # time of day, 288
    # DOW = np.unique(data[:, :, 2]).shape[0] # day of week, 7
    N = data.shape[1]
    # avg = defaultdict(defaultdict(defaultdict(list)))
    avg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    res = defaultdict(lambda: defaultdict(dict))
    for i in range(train_data.shape[0]):
        for j in range(N):
            avg[j][train_data[i, j, 1]][train_data[i, j, 2]].append(train_data[i, j, 0])
    for i in avg.keys():
        for j in avg[i].keys():
            for k in avg[i][j].keys():
                res[i][j][k] = np.mean(avg[i][j][k])
                # avg[i][j][k] = np.median(avg[i][j][k])

    # test
    y_pred = np.zeros(test_data.shape[:2])
    for i in range(test_data.shape[0]):
        for j in range(N):
            y_pred[i, j] = res[j][test_data[i, j, 1]][test_data[i, j, 2]]
    
    # evaluate
    # for i in range(5):
    #     for j in range(5):
    #         print(f'{y_pred[i,j]:.2f} : {test_data[i,j,0]:.2f}', end='    ')
    #     print()
    # print()
    y_pred = torch.tensor(y_pred).float()
    y_true = torch.tensor(np.squeeze(test_data[:,:,0])).float()
    print(f'Evaluation results {filepath}:')
    print(f'MAE: {masked_mae(y_pred, y_true).item()}, RMSE: {masked_rmse(y_pred, y_true).item()}, MAPE: {masked_mape(y_pred, y_true).item()}')

for x in '3478':
    with open(f'data/processed/PEMS0{x}/desc.json', encoding='utf-8') as f:
        desc = json.load(f)
    shape = desc['shape']
    main(f'data/processed/PEMS0{x}/data.dat', shape)
    # main(f'data/processed/PEMS0{x}/data_in12_out12.pkl')
    
''' np.mean
Evaluation results data/processed/PEMS03/data.dat:
MAE: 26.100666046142578, RMSE: 47.474395751953125, MAPE: 0.2687479853630066
Evaluation results data/processed/PEMS04/data.dat:
MAE: 26.422449111938477, RMSE: 43.42472457885742, MAPE: 0.1678064614534378
Evaluation results data/processed/PEMS07/data.dat:
MAE: 30.35526466369629, RMSE: 56.753475189208984, MAPE: 0.12799006700515747
Evaluation results data/processed/PEMS08/data.dat:
MAE: 23.249540328979492, RMSE: 40.586490631103516, MAPE: 0.14498291909694672

如果用 np.median，差不太多
Evaluation results data/processed/PEMS03/data.dat:
MAE: 26.120023727416992, RMSE: 48.109710693359375, MAPE: 0.2639146149158478
Evaluation results data/processed/PEMS04/data.dat:
MAE: 25.938827514648438, RMSE: 44.14350891113281, MAPE: 0.1621645987033844
Evaluation results data/processed/PEMS07/data.dat:
MAE: 30.468910217285156, RMSE: 59.47943878173828, MAPE: 0.1260502189397812
Evaluation results data/processed/PEMS08/data.dat:
MAE: 22.373567581176758, RMSE: 41.640342712402344, MAPE: 0.13811932504177094'''