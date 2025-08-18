import pickle
import numpy as np
class Graph():
    def load_graph(self, filepath:str):
        with open(filepath, 'rb') as f:
            self.g:np.ndarray = pickle.load(f)
    def __init__(self, filepath:str=''):
        self.load_graph(filepath)
        
    def get_neighbors(self, node:int):
        return np.nonzero(self.g[node])[0]
    
class PEMSGraph(Graph):
    def __init__(self, x):
        assert str(x) in '3478' and len(str(x)) == 1
        path = f'data/processed/PEMS0{x}/adj_mx.pkl'
        super().__init__(path)

if __name__ == '__main__':
    g = PEMSGraph(3)
    print(g.get_neighbors(0))