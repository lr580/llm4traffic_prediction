import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.datasets import PEMSDatasetHandler

for x in [3,4,7,8]:
    handler = PEMSDatasetHandler(x, loadData=True)
    data = handler.dataset.data
    means = data[:, :, 0].mean(axis=0)
    print(f'\nPEMS0{x}')
    print(means)
    ''' 结论： PEMS04, 08 BasicTS 与 ASTGCN 一样，都没有做减去 mean 的预处理
     其中，ASCTCN 的参见 pems0x_preview_raw.py '''