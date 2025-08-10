import numpy as np
import json
class Dataset():
    def __init__(self, path=''):
        self.data = self.load_data(path)
        self.train_ratio = 0.6
        self.val_ratio = 0.2
        self.test_ratio = 1 - self.train_ratio - self.val_ratio
        self.t = self.data.shape[0]
        self.n = self.data.shape[1]
        self.train_end = int(self.t*self.train_ratio)
        self.val_end = self.train_end + int(self.t*self.val_ratio)
        
    def load_data(self, path):
        return np.zeros((1,1,3), dtype=np.float32)
        
    def get_train(self):
        return self.data[:self.train_end,:,:]
    
    def get_val(self):
        return self.data[self.train_end:self.val_end,:,:]
    
    def get_test(self):
        return self.data[self.val_end:,:,:]
    
class PEMSDataset(Dataset):
    def load_data(self, x):
        with open(f'data/processed/PEMS0{x}/desc.json', encoding='utf-8') as f:
            desc = json.load(f)
        shape = desc['shape']
        filepath = f'data/processed/PEMS0{x}/data.dat'
        return np.memmap(filepath, dtype=np.float32, mode='r', shape=shape)
    
    def __init__(self, x):
        super().__init__(x)