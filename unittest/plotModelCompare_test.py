import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.datasets import DataList, plotCompareDataResult
list1 = DataList.load('results/HA/PEMS03/32-1/results_tiny_test.json')
list2 = DataList.load('results/plain/PEMS03/32/results_tiny_test.json')
plotCompareDataResult([list1, list2], ['HA', 'plain'], 'mape')
plotCompareDataResult([list1, list2], ['HA', 'plain'], 'mape')
plotCompareDataResult([list1, list2], ['HA', 'plain'], 'mae')