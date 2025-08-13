import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.datasets import DataList, getDataResults
list1 = DataList.load('results/HA/PEMS03/64-1/results_tiny_test.json')
list2 = DataList.load('results/plain/PEMS03/64/results_tiny_test.json')
list3 = DataList.load('results/HA_neighbor/PEMS03/64/results_tiny_test.json')
lists = [list1, list2, list3]
names = ['HA', 'Plain', 'HA_Nei']
print(getDataResults(lists, names))