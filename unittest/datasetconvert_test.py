import sys, os, tqdm
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.datasets import PEMSDatasetHandler, BasicTS_TSLib_Converter
for x in tqdm.tqdm((3, 4, 7, 8)):
    converter = BasicTS_TSLib_Converter(PEMSDatasetHandler(x, loadData=True, stat=False))
    converter.toTSLib('data/PEMS0' + str(x) + '.csv')