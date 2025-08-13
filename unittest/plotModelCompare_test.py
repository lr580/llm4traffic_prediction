import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.datasets import DataList, plotCompareDataResult, getDataResults
list1 = DataList.load('results/HA/PEMS03/32-1/results_tiny_test.json')
list2 = DataList.load('results/plain/PEMS03/32/results_tiny_test.json')
list3 = DataList.load('results/neighbor/PEMS03/32-2/results_tiny_test.json')
list4 = DataList.load('results/HA_neighbor/PEMS03/32/results_tiny_test.json')
lists = [list1, list2, list3, list4]
names = ['HA', 'Plain', 'Neighbor', 'HA_Nei']
print(getDataResults(lists, names))
plotCompareDataResult(lists, names, 'mape')
plotCompareDataResult(lists, names, 'mape')
plotCompareDataResult(lists, names, 'mae')