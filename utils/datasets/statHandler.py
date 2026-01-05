import numpy as np
from .dataset import PEMSDataset, Dataset
from tqdm import tqdm
from typing import cast
class HAstat():
    '''see utils/baselines/HA.py for detail'''
    def __init__(self, data : np.ndarray, metric=np.mean):
        self.tod = np.unique(data[:, :, 1])
        self.dow = np.unique(data[:, :, 2])
        self.itod = {t: i for i, t in enumerate(self.tod)}
        self.idow = {d: i for i, d in enumerate(self.dow)}
        self.n = data.shape[1]
        self.n_tod = len(self.tod)
        self.n_dow = len(self.dow)
        self.data = data # pointer
        t = data.shape[0]
        
        group = [[[[] for k in range(self.n_dow)] for j in range(self.n_tod)] for i in range(self.n)]
        for i in tqdm(range(t), 'HA grouping'):
            for j in range(self.n):
                group[j][self.itod[data[i, j, 1]]][self.idow[data[i, j, 2]]].append(data[i, j, 0])
        self.avg = [[[0. for k in range(self.n_dow)] for j in range(self.n_tod)] for i in range(self.n)]
        for i in tqdm(range(self.n), 'HA calculating'):
            for j in range(self.n_tod):
                for k in range(self.n_dow):
                    self.avg[i][j][k] = metric(group[i][j][k])
        # return
        # group = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # for i in range(t):
        #     for j in range(self.n):
        #         group[j][data[i, j, 1]][data[i, j, 2]].append(data[i, j, 0])
        # self.avg = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # for i in group.keys():
        #     for j in group[i].keys():
        #         for k in group[i][j].keys():
        #             self.avg[i][j][k] = metric(group[i][j][k])
        
    
    def get(self, i:int, j:int)->np.float32:
        # return self.avg[j][self.data[i,j,1]][self.data[i,j,2]]
        res = self.avg[j][self.itod[self.data[i, j, 1]]][self.idow[self.data[i, j, 2]]]
        return cast("np.float32", res)
    
    def getRange(self, l:int, r:int, j:int, data:np.ndarray): 
        ''' 保证没有使用真实数据 (data[i,j,0])，没有数据泄露'''
        return np.array([self.calc(j, data[i,j,1], data[i,j,2]) for i in range(l, r)])
    
    def calc(self, n:int, tod:np.float32, dow:np.float32)->np.float32:
        # return self.avg[n][tod][dow]
        res = self.avg[n][self.itod[tod]][self.idow[dow]]
        return cast("np.float32", res)
    
class BasicTSHAstat(HAstat):
    def __init__(self, dataset:Dataset, train_ratio=0.6, metric=np.mean, l=0, r=0, customInterval=False):
        fulldata = dataset.data
        if not customInterval:
            l = 0
            r = int(dataset.t * train_ratio)
        data = fulldata[l:r, :, :]
        super().__init__(data, metric)
    
class PEMS_HAstat(BasicTSHAstat): # 向前兼容
    def __init__(self, dataset:PEMSDataset, train_ratio=0.6, metric=np.mean, l=0, r=0, customInterval=False):
        super().__init__(dataset, train_ratio, metric, l, r, customInterval)
