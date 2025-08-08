# 读取数据集进行查看
import numpy as np
for x in '3478':
    data = np.load(f'data/raw/PEMS0{x}/PEMS0{x}.npz')
    # print(data.files) # ['data']
    print(data['data'].shape)
    # print(data['data'].dtype) # float64
    sub = data['data'][:5,:5,:]
    for r in range(5):
        for c in range(5):
            for k in range(sub.shape[2]):
                print(f'{sub[r,c,k]:.2f}', end=' ')
            print(' ', end='')
        print()
    print()