import pickle
import numpy as np
import pandas as pd
class Graph():
    def load_graph(self, filepath:str):
        with open(filepath, 'rb') as f:
            self.g:np.ndarray = pickle.load(f)
    def __init__(self, filepath:str=''):
        self.load_graph(filepath)
        
    def get_neighbors(self, node:int):
        return np.nonzero(self.g[node])[0]
    
class BasicTSGraph(Graph):
    def __init__(self, dataset):
        path = f'data/processed/{dataset}/adj_mx.pkl'
        super().__init__(path)
    
class PEMSGraph(BasicTSGraph): # 向下兼容
    def __init__(self, x):
        assert str(x) in '3478' and len(str(x)) == 1
        super().__init__(f'PEMS0{x}')

class LargeSTMeta():
    def __init__(self, dataset: str):
        path = f'data/processed/{dataset}/meta.csv'
        self.meta = pd.read_csv(path)
        print(self.meta)

if __name__ == '__main__':
    g = PEMSGraph(3)
    print(g.get_neighbors(0))