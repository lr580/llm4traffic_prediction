# 读取数据集进行查看
# PEMS04,8 datasets from https://github.com/wanhuaiyu/ASTGCN/tree/master
import numpy as np
for x in '48':
    data = np.load(f'data/ASTGCN/PEMS0{x}/pems0{x}.npz')
    print(data.files) # ['data']
    data = data['data']
    print(data.shape) # [T, N, C]
    means = data[:, :, 0].mean(axis=0)
    print(f'PEMS0{x}')
    print(means)

    for r in range(5):
        for c in range(5):
            for k in range(data.shape[2]):
                print(f'{data[r,c,k]:.2f}', end=' ')
            print(' ', end='')
        print()
    print()
