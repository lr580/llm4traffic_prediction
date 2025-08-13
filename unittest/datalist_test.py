import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.datasets import PEMSDatasetHandler, DataList
"""
handler = PEMSDatasetHandler(3, loadData=True)
datalist = handler.buildBatchInput()
path = 'data/tiny/PEMS03/16-3.json'
# datalist = handler.dataset.buildBatchInput(1)
# path = 'data/tiny/PEMS03/1-1.json'
# datalist = handler.dataset.buildBatchInput(32)
# path = 'data/tiny/PEMS03/32-1.json'
# datalist = handler.dataset.buildBatchInput(64)
# path = 'data/tiny/PEMS03/64-1.json'
datalist.save(path)
datalist2 = DataList.load(path)
print(datalist2)
"""
for x in [4,7,8]: # 3 already generated
    handler = PEMSDatasetHandler(x, loadData=True)
    datalist = handler.buildBatchInput()
    path = f'data/tiny/PEMS0{x}/32-1.json'
    datalist.save(path)
    datalist2 = DataList.load(path)
    print(datalist2)