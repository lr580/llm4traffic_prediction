import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.datasets import PEMSDatasetHandler, DataList
handler = PEMSDatasetHandler(3)
# datalist = handler.dataset.buildBatchInput()
# path = 'data/tiny/PEMS03/16-1.json'
datalist = handler.dataset.buildBatchInput(1)
path = 'data/tiny/PEMS03/16-2.json'
datalist.save(path)
datalist2 = DataList.load(path)
print(datalist2)