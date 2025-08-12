import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.datasets import DataList
dataList = DataList.load('results/plain/PEMS03/16-2/results_tiny_test.json')
for data in dataList:
    data.plotResult(savepath=f'results/plain/PEMS03/16-2/{data.input.i}_{data.input.j}.png', show=False)