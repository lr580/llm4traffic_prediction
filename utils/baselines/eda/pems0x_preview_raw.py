# 读取数据集进行查看，对边权列进行统计分析
# PEMS04,8 datasets from https://github.com/wanhuaiyu/ASTGCN/tree/master
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import csv

# 设置中文字体（Windows 下一般有 SimHei 黑体），以及正常显示负号
rcParams['font.sans-serif'] = ['SimHei']  # 可根据本机情况改成其他中文字体
rcParams['axes.unicode_minus'] = False


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('PEMS数据集distance.csv cost分布', fontsize=16)
axes = axes.flatten()

subplot_idx = 0

# 用于收集统计信息
stats_list = []

for x in '3748':
    if int(x) % 2 == 0:
        data = np.load(f'data/ASTGCN/PEMS0{x}/pems0{x}.npz')
        dist_path = f'data/ASTGCN/PEMS0{x}/distance.csv'
        d_name = 'cost'
    else:
        data = np.load(f'data/STSGCN/PEMS0{x}/PEMS0{x}.npz')
        dist_path = f'data/STSGCN/PEMS0{x}/PEMS0{x}.csv'
        # PEMS03 使用 'distance' 列，PEMS07 使用 'cost' 列
        d_name = 'distance' if x == '3' else 'cost'

    print(data.files)  # ['data']
    data = data['data']
    print(data.shape)  # [T, N, C]
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

    distances = np.genfromtxt(dist_path, delimiter=',', names=True)
    # print(distances.dtype.names)
    cost = distances[d_name]
    frm = distances['from']
    to = distances['to']

    nodes = np.unique(np.concatenate([frm, to]))
    num_nodes = nodes.size

    num_edges = len(distances)

    print(f'PEMS0{x} distance.csv cost 列统计特征：')
    print(f'  样本数: {cost.size}')
    print(f'  均值: {cost.mean():.4f}')
    print(f'  标准差: {cost.std():.4f}')
    print(f'  最小值: {cost.min():.4f}')
    print(f'  最大值: {cost.max():.4f}')
    print(f'  图节点数: {num_nodes}')
    print(f'  图边数: {num_edges}')

    # 收集统计信息到列表
    stats_list.append({
        'dataset': f'PEMS0{x}',
        # 'sample_count': cost.size,
        'mean': f'{cost.mean():.4f}',
        'std': f'{cost.std():.4f}',
        'min': f'{cost.min():.4f}',
        'max': f'{cost.max():.4f}',
        'num_nodes': num_nodes,
        'num_edges': num_edges
    })

    # 在对应的子图上绘制直方图
    ax = axes[subplot_idx]
    ax.hist(cost, bins=30, edgecolor='black')
    ax.set_title(f'PEMS0{x} distance.csv cost 分布')
    ax.set_xlabel('cost')
    ax.set_ylabel('频数')
    ax.grid(alpha=0.3)

    # 更新子图索引
    subplot_idx += 1

# 调整子图之间的间距
plt.tight_layout()
# 调整顶部标题与子图的间距
plt.subplots_adjust(top=0.92)

# 保存统计信息到 CSV 文件
csv_filename = 'utils/baselines/eda/pems0x_statistics.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['dataset', 'mean', 'std', 'min', 'max', 'num_nodes', 'num_edges']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for stats in stats_list:
        writer.writerow(stats)

print(f'\n统计信息已保存到: {csv_filename}')

plt.show()
