# run in BasicTS rootpath
import numpy as np
data = np.memmap('datasets/PEMS03/data.dat', dtype=np.float32, mode='r', shape=(26208, 358, 3))
sub = data[:5,:5,:]
for r in range(5):
    for c in range(5):
        for k in range(sub.shape[2]):
            print(f'{sub[r,c,k]:.2f}', end=' ')
        print(' ', end='')
    print()
print()