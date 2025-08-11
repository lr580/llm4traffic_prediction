import numpy as np
import json
import random
import typing
from .data import SingleInput, SingleData, DataList
class Dataset():
    def __init__(self, path:typing.Union[int, str]=''):
        self.data = self.load_data(path)
        self.space = self.load_space()
        self.name = self.load_name()
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
    
    def load_space(self): # to be implemented by subclass
        return 'undefined'
    
    def load_name(self): # to be implemented by subclass
        return 'undefined'
        
    def get_train(self):
        return self.data[:self.train_end,:,:]
    
    def get_val(self):
        return self.data[self.train_end:self.val_end,:,:]
    
    def get_test(self):
        return self.data[self.val_end:,:,:]

    def buildSingleInput(self, i:int, j:int, copy:bool):
        X, y = self.get_data(i, j, copy)
        singleInput = SingleInput(X=X, i=i, j=j)
        return SingleData(input=singleInput, y_true=y)
    
    def buildBatchInput(self, batch_size:int=16, copy=True):
        batch = []
        for _ in range(batch_size):
            i, j = self.get_random_index()
            batch.append(self.buildSingleInput(i, j, copy))
        return DataList(data=batch)
    
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
        i, j = self.get_random_index(i, j)
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
    def load_data(self, x:int):
        with open(f'data/processed/PEMS0{x}/desc.json', encoding='utf-8') as f:
            desc = json.load(f)
        shape = desc['shape']
        filepath = f'data/processed/PEMS0{x}/data.dat'
        return np.memmap(filepath, dtype=np.float32, mode='r', shape=shape)
    
    def load_space(self):
        # the information from https://www.sciencedirect.com/science/article/pii/S0957417422011654
        if self.x == 3:
            return '中北部区域(North Central Area)'
        elif self.x == 4:
            return '旧金山湾区(San Francisco Bay Area)'
        elif self.x == 7:
            return '洛杉矶区域(Los Angeles Area)'
        elif self.x == 8:
            return '圣贝纳迪诺区(San Bernardino Area)'
        
    def load_name(self):
        return f'PEMS0{self.x}'
    
    def __init__(self, x):
        self.x = x
        super().__init__(x)