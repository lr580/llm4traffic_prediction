import sys, os, tqdm
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.datasets import DataList, PEMSDatasetHandler
dataList = DataList.load('results/HA/PEMS03/64-1/results_tiny_test.json')
handler = PEMSDatasetHandler(3, loadData=True, stat=True)
for data in dataList:
# for data in tqdm.tqdm(dataList, 'Plotting'):
    handler.plotResultWithHA(data, savepath=f'results/HA/PEMS03/64-1/{data.input.i}_{data.input.j}.png', show=False)