import pickle
import numpy as np
for x in '3478':
    path = f'data/processed/PEMS0{x}/adj_mx.pkl'
    with open(path, 'rb') as f:
        g = pickle.load(f) # <class 'numpy.ndarray'>
    print(g.shape) # (358, 358)
    print(np.array_equal(g, g.T)) # 对称矩阵
    print(g)
    print(g.sum(axis=1))