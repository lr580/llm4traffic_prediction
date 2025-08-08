import os
import pickle
for x in '3478':
    folder = f'data/processed/PEMS0{x}'
    with open(os.path.join(folder, 'data_in12_out12.pkl'), 'rb') as f:
        data = pickle.load(f)['processed_data'] 
        # print(type(data)) # <class 'numpy.ndarray'>
    print(data.shape) # pems03 (26208, 358, 3), 第三维+1
    sub = data[:5,:5,:]
    for r in range(5):
        for c in range(5):
            for k in range(sub.shape[2]):
                print(f'{sub[r,c,k]:.2f}', end=' ')
            print(' ', end='')
        print()
    print()
    '''
    -1.12 0.00 0.00  -1.12 0.00 0.00  0.00 0.00 0.00  0.00 0.00 0.00  -0.63 0.00 0.00  
    -1.10 0.00 0.00  -1.10 0.00 0.00  -0.05 0.00 0.00  -0.05 0.00 0.00  -0.65 0.00 0.00
    -1.10 0.01 0.00  -1.10 0.01 0.00  0.01 0.01 0.00  0.01 0.01 0.00  -0.62 0.01 0.00
    -0.91 0.01 0.00  -0.92 0.01 0.00  -0.31 0.01 0.00  -0.29 0.01 0.00  -0.84 0.01 0.00
    -1.00 0.01 0.00  -1.01 0.01 0.00  -0.37 0.01 0.00  -0.40 0.01 0.00  -0.88 0.01 0.00'''
    
    with open(os.path.join(folder, 'index_in12_out12.pkl'), 'rb') as f:
        index = pickle.load(f)
        # print(index.keys()) # dict_keys(['train', 'valid', 'test'])
    # print(type(index['train'])) # list
    print(index['train'][:10]) # [(0, 12, 24), (1, 13, 25), (2, 14, 26), (3, 15, 27), (4, 16, 28), (5, 17, 29), (6, 18, 30), (7, 19, 31), (8, 20, 32), (9, 21, 33)]
    
    with open(os.path.join(folder, 'scaler_in12_out12.pkl'), 'rb') as f:
        scaler = pickle.load(f)
        # print(scaler.keys()) # func, args
        print(scaler['func'])
        print(scaler['args'])
        #re_standard_transform
        #{'mean': np.float64(181.37324561746377), 'std': np.float64(144.4023243785617)}  