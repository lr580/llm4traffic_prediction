import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.datasets import DataList, PEMSDatasetHandler
# print('start')
# path = 'results/HA/PEMS03/32-1/results_tiny_test.json'
# path = 'results/HA/PEMS03/64-1/results_tiny_test.json'
# path = 'data/tiny/PEMS03/1-1.json'
# path = 'data/tiny/PEMS03/16-1.json'
# path = 'data/tiny/PEMS03/32-1.json'
path = 'data/tiny/PEMS03/64-1.json'
dataList = DataList.load(path)
handler = PEMSDatasetHandler(3)
handler.rebuildBatchInput(dataList)
dataList.save(path)