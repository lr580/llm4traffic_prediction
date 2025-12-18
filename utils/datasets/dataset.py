import numpy as np
import json
import random
import typing
from .data import SingleInput, SingleData, DataList
class Dataset():
    def __init__(self, path:typing.Union[int, str]=''):
        self.desc_json = dict()
        self.data = self.load_data(str(path))
        ''' shape [T, N, C] '''
        self.train_ratio = 0.6
        self.val_ratio = 0.2
        self.test_ratio = 1 - self.train_ratio - self.val_ratio
        self.t = self.data.shape[0]
        self.n = self.data.shape[1]
        self.train_end = int(self.t*self.train_ratio)
        self.val_end = self.train_end + int(self.t*self.val_ratio)
        self.T_input = 12
        self.T_output = 12
        
    def load_data(self, path:str): # to be implemented by subclass
        return np.zeros((1,1,3), dtype=np.float32)
        
    def get_train(self):
        return self.data[:self.train_end,:,:]
    
    def get_val(self):
        return self.data[self.train_end:self.val_end,:,:]
    
    def get_test(self):
        return self.data[self.val_end:,:,:]
    
    # Not used
    # def get_time_of_day(self, i:int, j:int):
    #     '''可以用 timeCalc 计算的，懒得算了, BasicTS only'''
    #     return self.data[i,j,1] 
    # def get_day_of_week(self, i:int, j:int):
    #     '''可以用 timeCalc 计算的，懒得算了, BasicTS only'''
    #     return self.data[i,j,2] 
    
    def get_data(self, i:int, j:int, copy=False):
        '''只取数据，不取time of day, day of week；返回长为T的俩一维向量'''
        X = np.squeeze(self.data[i-self.T_input:i,j,0])
        y = np.squeeze(self.data[i:i+self.T_output,j,0])
        if copy: # memmap -> numpy
            X = np.array(X)
            y = np.array(y)
        return X, y
    
    def get_random_index(self):
        i = random.randint(self.val_end, self.t-self.T_output-1)
        j = random.randint(0, self.n-1)
        return i, j
    
    def get_random_data(self):
        i, j = self.get_random_index()
        return *self.get_data(i, j), i, j
    
    def get_random_batch(self, batch_size:int=16):
        batch_X = []
        batch_y = []
        batch_index = []
        for _ in range(batch_size):
            X, y, i, j = self.get_random_data()
            batch_X.append(X)
            batch_y.append(y)
            batch_index.append((i, j))
        batch_X = np.stack(batch_X, axis=0)
        batch_y = np.stack(batch_y, axis=0)
        return batch_X, batch_y, batch_index
    
class PEMSDataset(Dataset):
    ''' Data from BasicTS (v0.x) https://github.com/GestaltCogTeam/BasicTS '''
    def load_data(self, x:int):
        with open(f'data/processed/PEMS0{x}/desc.json', encoding='utf-8') as f:
            desc = json.load(f)
        shape = tuple(desc['shape']) # 要用 tuple 而不是 list，否则一些版本会出问题
        self.desc_json = desc # for future usage / reflection
        filepath = f'data/processed/PEMS0{x}/data.dat'
        return np.memmap(filepath, dtype=np.float32, mode='r', shape=shape)
    
    def __init__(self, x):
        self.x = x
        super().__init__(x)