import sys, os, tqdm
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.datasets import PEMSDatasetHandler, BasicTS_Shrink
for x in tqdm.tqdm((3, )):
    handler = PEMSDatasetHandler(x, loadData=True, stat=False)
    BasicTS_Shrink.shrink(handler, 'data.dat', 'desc.json', 'PEMS03_small')