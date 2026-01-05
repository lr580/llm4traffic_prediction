import numpy as np
from .dataset import PEMSDataset, Dataset
from typing import cast
class HAstat():
    ''' usage: unittest/HA_test.py; 
    see utils/baselines/HA.py for detail '''
    def __init__(self, data : np.ndarray, metric=np.mean):
        self.tod = np.unique(data[:, :, 1])
        self.dow = np.unique(data[:, :, 2])
        self.itod = {t: i for i, t in enumerate(self.tod)}
        self.idow = {d: i for i, d in enumerate(self.dow)}
        self.n = data.shape[1]
        self.n_tod = len(self.tod)
        self.n_dow = len(self.dow)
        self.data = data # pointer

        # updates: 优化了计算，使用 numpy，提速至少几倍；优化前：
        if metric in (np.mean, np.average):
            self._build_mean_stat(data)
        else:
            self._build_stat_with_metric(data, metric)
    
    def _flatten_indices(self, data: np.ndarray):
        """将 (node, tod, dow) 映射到扁平数组索引"""
        t = data.shape[0]
        values = np.asarray(data[:, :, 0], dtype=np.float32).reshape(-1)
        node_idx = np.tile(np.arange(self.n), t)
        tod_idx = np.searchsorted(self.tod, data[:, :, 1].reshape(-1))
        dow_idx = np.searchsorted(self.dow, data[:, :, 2].reshape(-1))
        return values, node_idx, tod_idx, dow_idx
    
    def _combine_indices(self, node_idx, tod_idx, dow_idx):
        """组合多个索引为单个整数标识"""
        return node_idx * (self.n_tod * self.n_dow) + tod_idx * self.n_dow + dow_idx
    
    def _decode_combo_id(self, combo_id: int):
        """将组合索引还原为 (node, tod, dow)"""
        tod_dow = self.n_tod * self.n_dow
        node_idx = combo_id // tod_dow
        rem = combo_id % tod_dow
        tod_idx = rem // self.n_dow
        dow_idx = rem % self.n_dow
        return node_idx, tod_idx, dow_idx
    
    def _build_mean_stat(self, data: np.ndarray):
        """针对均值统计使用纯 numpy 汇总"""
        values, node_idx, tod_idx, dow_idx = self._flatten_indices(data)
        combos = self._combine_indices(node_idx, tod_idx, dow_idx)
        total = self.n * self.n_tod * self.n_dow
        sums = np.zeros(total, dtype=np.float64)
        counts = np.zeros(total, dtype=np.int64)
        np.add.at(sums, combos, values)
        np.add.at(counts, combos, 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            averages = sums / counts
        averages[counts == 0] = np.nan
        self.avg = averages.reshape(self.n, self.n_tod, self.n_dow).astype(np.float32)
    
    def _build_stat_with_metric(self, data: np.ndarray, metric):
        """针对自定义 metric 逐组统计结果"""
        values, node_idx, tod_idx, dow_idx = self._flatten_indices(data)
        combos = self._combine_indices(node_idx, tod_idx, dow_idx)
        order = np.argsort(combos)
        combos_sorted = combos[order]
        values_sorted = values[order]
        unique_ids, counts = np.unique(combos_sorted, return_counts=True)
        split_points = np.cumsum(counts)[:-1]
        grouped_values = np.split(values_sorted, split_points) if counts.size else []
        avg = np.full((self.n, self.n_tod, self.n_dow), np.nan, dtype=np.float32)
        for combo_id, group_vals in zip(unique_ids, grouped_values):
            node_idx, tod_idx, dow_idx = self._decode_combo_id(combo_id)
            avg_val = metric(group_vals)
            avg[node_idx, tod_idx, dow_idx] = float(avg_val)
        self.avg = avg
    
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
