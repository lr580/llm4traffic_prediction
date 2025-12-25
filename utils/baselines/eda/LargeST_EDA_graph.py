''' LargeST 数据集：https://github.com/liuxu77/LargeST 下载放在 data/LargeST/ '''
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False 

adj = np.load('data/LargeST/ca_rn_adj.npy')
print(f"数据类型: {type(adj)}")
print(f"数组形状: {adj.shape}")
print(f"数据类型: {adj.dtype}")
print(f"最小值: {adj.min()}") # 0
print(f"最大值: {adj.max()}") # 1
print(f"平均值: {adj.mean()}")
uni = np.unique(adj)
print(f'{uni.size} {uni}') # 0.01000184, ....  5933 distinct values
'''数组形状: (8600, 8600)
数据类型: float64
最小值: 0.0
最大值: 1.0
平均值: 0.0010241728174695383
5933 [0.         0.01000184 0.01001734 ... 0.99999948 0.99999987 1.        ]'''

plt.figure(figsize=(12, 5))
# 绘制直方图
plt.subplot(1, 2, 1)
plt.hist(adj.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('邻接矩阵值分布')
plt.xlabel('值')
plt.ylabel('频数')
plt.grid(True, alpha=0.3)

# 如果有很多0值，可以单独查看非零值的分布
non_zero_values = adj[adj != 0]
plt.subplot(1, 2, 2)
plt.hist(non_zero_values.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
plt.title('非零值分布')
plt.xlabel('值')
plt.ylabel('频数')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 添加更多统计分析
print("\n=== 更多统计信息 ===")
print(f"零值比例: {(adj == 0).sum() / adj.size:.4f}")
print(f"非零值数量: {(adj != 0).sum()}")
print(f"非零值的最小值: {adj[adj != 0].min() if (adj != 0).sum() > 0 else 'N/A'}")
print(f"非零值的平均值: {adj[adj != 0].mean() if (adj != 0).sum() > 0 else 'N/A'}")

# 检查是否为对称矩阵（如果是无向图）
if adj.shape[0] == adj.shape[1]:
    is_symmetric = np.allclose(adj, adj.T)
    print(f"是否对称: {is_symmetric}")
    
# 检查稀疏性
sparsity = 1.0 - (adj != 0).sum() / adj.size
print(f"稀疏度: {sparsity:.6f}")

# 对角线信息
if adj.shape[0] == adj.shape[1]:
    diag_values = np.diag(adj)
    print(f"对角线平均值: {diag_values.mean():.6f}")
    print(f"对角线非零值数量: {(diag_values != 0).sum()}")
'''零值比例: 0.9972
非零值数量: 209963
非零值的最小值: 0.010001835317713963
非零值的平均值: 0.3607674760793427
是否对称: False
稀疏度: 0.997161
对角线平均值: 1.000000
对角线非零值数量: 8600'''

if adj.shape[0] == adj.shape[1]:
    print("对称性详细统计")
    # 获取上三角索引（i < j）
    n = adj.shape[0]
    i_idx, j_idx = np.triu_indices(n, k=1)  # k=1 排除对角线
    d_ij = adj[i_idx, j_idx]
    d_ji = adj[j_idx, i_idx]
    
    # 1. 统计对称元素的情况
    total_pairs = len(i_idx)
    
    # 情况1: d[i,j]和d[j,i] 都非零且相等
    both_nonzero = (d_ij != 0) & (d_ji != 0)
    equal_both_nonzero = (both_nonzero) & (np.abs(d_ij - d_ji) < 1e-10)  # 考虑浮点误差
    count_case1 = np.sum(equal_both_nonzero)
    
    # 情况2: d[i,j]和d[j,i] 都非零但不相等
    both_nonzero_not_equal = (both_nonzero) & (np.abs(d_ij - d_ji) >= 1e-10)
    count_case2 = np.sum(both_nonzero_not_equal)
    
    # 情况3: 只有一个为零
    one_zero = ((d_ij != 0) & (d_ji == 0)) | ((d_ij == 0) & (d_ji != 0))
    count_case3 = np.sum(one_zero)
    
    # 情况4: 都为零
    both_zero = (d_ij == 0) & (d_ji == 0)
    count_case4 = np.sum(both_zero)
    
    print(f"上三角元素对数 (i<j): {total_pairs:,}")
    print("\n四种情况统计:")
    print(f"1. 都非零且相等: {count_case1:,} 对 ({count_case1/total_pairs*100:.2f}%)")
    print(f"2. 都非零但不相等: {count_case2:,} 对 ({count_case2/total_pairs*100:.2f}%)")
    print(f"3. 只有一个为零: {count_case3:,} 对 ({count_case3/total_pairs*100:.2f}%)")
    print(f"4. 都为零: {count_case4:,} 对 ({count_case4/total_pairs*100:.2f}%)")
'''上三角元素对数 (i<j): 36,975,700
四种情况统计:
1. 都非零且相等: 440 对 (0.00%)
2. 都非零但不相等: 79,164 对 (0.21%)
3. 只有一个为零: 42,155 对 (0.11%)
4. 都为零: 36,853,941 对 (99.67%)'''

# 都非零但不相等的作差绝对值，与任意两点作差绝对值，没有显著统计学差异，如下：
if False:
    #情况A: 都非零但不相等的情况
    if count_case2 > 0:
        diff_case2 = np.abs(d_ij[both_nonzero_not_equal] - d_ji[both_nonzero_not_equal])
        print(f"\n情况2 (都非零但不相等) 的绝对值差统计:")
        print(f"  元素对数量: {len(diff_case2):,}")
        print(f"  绝对值差最小值: {diff_case2.min():.6f}")
        print(f"  绝对值差最大值: {diff_case2.max():.6f}")
        print(f"  绝对值差平均值: {diff_case2.mean():.6f}")
        print(f"  绝对值差中位数: {np.median(diff_case2):.6f}")
        print(f"  绝对值差标准差: {diff_case2.std():.6f}")
        
        # 计算百分位数
        percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        print(f"  绝对值差百分位数:")
        for p in percentiles:
            value = np.percentile(diff_case2, p)
            print(f"    {p:3d}%: {value:.6f}")
        
        # 相对差异 (以较大值为分母)
        max_vals = np.maximum(np.abs(d_ij[both_nonzero_not_equal]), np.abs(d_ji[both_nonzero_not_equal]))
        rel_diff_case2 = diff_case2 / max_vals
        rel_diff_case2 = rel_diff_case2[max_vals > 0]  # 避免除零
        
        print(f"\n  相对差异统计 (|d_ij - d_ji| / max(|d_ij|, |d_ji|)):")
        if len(rel_diff_case2) > 0:
            print(f"    相对差异最小值: {rel_diff_case2.min():.6f}")
            print(f"    相对差异最大值: {rel_diff_case2.max():.6f}")
            print(f"    相对差异平均值: {rel_diff_case2.mean():.6f}")
            print(f"    相对差异中位数: {np.median(rel_diff_case2):.6f}")
            for p in [50, 75, 90, 95, 99, 100]:
                value = np.percentile(rel_diff_case2, p)
                print(f"    相对差异 {p:3d}% 分位数: {value:.6f}")
        else:
            print(f"    无法计算相对差异 (全零值)")
    else:
        diff_case2 = np.array([])
        rel_diff_case2 = np.array([])
        print("\n情况2: 没有都非零但不相等的元素对")

    # 情况B: 所有都非零的情况 (包括相等和不相等)
    if np.sum(both_nonzero) > 0:
        # 所有都非零的情况
        diff_all_both_nonzero = np.abs(d_ij[both_nonzero] - d_ji[both_nonzero])
        print(f"\n情况B (所有都非零的情况) 的绝对值差统计:")
        print(f"  元素对数量: {len(diff_all_both_nonzero):,}")
        print(f"  绝对值差最小值: {diff_all_both_nonzero.min():.6f}")
        print(f"  绝对值差最大值: {diff_all_both_nonzero.max():.6f}")
        print(f"  绝对值差平均值: {diff_all_both_nonzero.mean():.6f}")
        print(f"  绝对值差中位数: {np.median(diff_all_both_nonzero):.6f}")
        print(f"  绝对值差标准差: {diff_all_both_nonzero.std():.6f}")
        
        # 计算百分位数
        print(f"  绝对值差百分位数:")
        for p in percentiles:
            value = np.percentile(diff_all_both_nonzero, p)
            print(f"    {p:3d}%: {value:.6f}")
        
        # 相对差异
        max_vals_all = np.maximum(np.abs(d_ij[both_nonzero]), np.abs(d_ji[both_nonzero]))
        rel_diff_all = diff_all_both_nonzero / max_vals_all
        rel_diff_all = rel_diff_all[max_vals_all > 0]
        
        print(f"\n  相对差异统计 (|d_ij - d_ji| / max(|d_ij|, |d_ji|)):")
        if len(rel_diff_all) > 0:
            print(f"    相对差异最小值: {rel_diff_all.min():.6f}")
            print(f"    相对差异最大值: {rel_diff_all.max():.6f}")
            print(f"    相对差异平均值: {rel_diff_all.mean():.6f}")
            print(f"    相对差异中位数: {np.median(rel_diff_all):.6f}")
            for p in [50, 75, 90, 95, 99, 100]:
                value = np.percentile(rel_diff_all, p)
                print(f"    相对差异 {p:3d}% 分位数: {value:.6f}")
        else:
            print(f"    无法计算相对差异 (全零值)")
        
        # 比较相等和不相等的情况
        equal_indices = equal_both_nonzero[both_nonzero]  # 在都非零的索引中，哪些是相等的
        not_equal_indices = ~equal_indices
        
        print(f"\n  在都非零的情况中:")
        print(f"    相等对数量: {np.sum(equal_indices):,} ({np.sum(equal_indices)/len(diff_all_both_nonzero)*100:.2f}%)")
        print(f"    不相等对数量: {np.sum(not_equal_indices):,} ({np.sum(not_equal_indices)/len(diff_all_both_nonzero)*100:.2f}%)")
        
        if np.sum(equal_indices) > 0:
            diff_equal = diff_all_both_nonzero[equal_indices]
            print(f"    相等对的绝对值差统计:")
            print(f"      最小值: {diff_equal.min():.6f}")
            print(f"      最大值: {diff_equal.max():.6f}")
            print(f"      平均值: {diff_equal.mean():.6f}")
        
        if np.sum(not_equal_indices) > 0:
            diff_not_equal = diff_all_both_nonzero[not_equal_indices]
            print(f"    不相等对的绝对值差统计:")
            print(f"      最小值: {diff_not_equal.min():.6f}")
            print(f"      最大值: {diff_not_equal.max():.6f}")
            print(f"      平均值: {diff_not_equal.mean():.6f}")
    else:
        diff_all_both_nonzero = np.array([])
        rel_diff_all = np.array([])
        print("\n情况B: 没有都非零的元素对")

    # 3. 绘制对比图
    print("\n" + "=" * 50)
    print("可视化对比")
    print("=" * 50)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 子图1: 绝对值差分布直方图对比
    ax1 = axes[0, 0]
    if len(diff_case2) > 0 and len(diff_all_both_nonzero) > 0:
        # 使用对数y轴
        ax1.hist(diff_case2, bins=50, alpha=0.7, color='red', edgecolor='black', 
                label=f'都非零不相等 (n={len(diff_case2):,})', density=True)
        ax1.hist(diff_all_both_nonzero, bins=50, alpha=0.5, color='blue', edgecolor='black',
                label=f'所有都非零 (n={len(diff_all_both_nonzero):,})', density=True)
        ax1.set_yscale('log')
        ax1.set_xlabel('绝对值差 |d[i,j] - d[j,i]|')
        ax1.set_ylabel('频率密度 (对数尺度)')
        ax1.set_title('绝对值差分布对比 (直方图)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, '数据不足\n无法绘制对比图', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)

    # 子图2: 累积分布函数对比
    ax2 = axes[0, 1]
    if len(diff_case2) > 0 and len(diff_all_both_nonzero) > 0:
        # 对差值进行排序
        sorted_diff_case2 = np.sort(diff_case2)
        sorted_diff_all = np.sort(diff_all_both_nonzero)
        
        # 计算累积分布
        y_case2 = np.arange(1, len(sorted_diff_case2) + 1) / len(sorted_diff_case2)
        y_all = np.arange(1, len(sorted_diff_all) + 1) / len(sorted_diff_all)
        
        ax2.plot(sorted_diff_case2, y_case2, 'r-', linewidth=2, 
                label=f'都非零不相等 (n={len(diff_case2):,})')
        ax2.plot(sorted_diff_all, y_all, 'b-', linewidth=2, 
                label=f'所有都非零 (n={len(diff_all_both_nonzero):,})')
        
        ax2.set_xlabel('绝对值差阈值')
        ax2.set_ylabel('累积概率')
        ax2.set_title('绝对值差累积分布函数对比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
    else:
        ax2.text(0.5, 0.5, '数据不足\n无法绘制CDF', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)

    # 子图3: 相对差异分布
    ax3 = axes[0, 2]
    if len(rel_diff_case2) > 0 and len(rel_diff_all) > 0:
        ax3.hist(rel_diff_case2, bins=50, alpha=0.7, color='red', edgecolor='black',
                label=f'都非零不相等 (n={len(rel_diff_case2):,})', density=True)
        ax3.hist(rel_diff_all, bins=50, alpha=0.5, color='blue', edgecolor='black',
                label=f'所有都非零 (n={len(rel_diff_all):,})', density=True)
        ax3.set_xlabel('相对差异 |d[i,j] - d[j,i]| / max(|d[i,j]|, |d[j,i]|)')
        ax3.set_ylabel('频率密度')
        ax3.set_title('相对差异分布对比')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, '数据不足\n无法绘制相对差异图', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)

    # 子图4: 箱线图对比
    ax4 = axes[1, 0]
    if len(diff_case2) > 0 and len(diff_all_both_nonzero) > 0:
        data_to_plot = []
        labels = []
        
        if len(diff_case2) > 0:
            data_to_plot.append(diff_case2)
            labels.append('都非零不相等')
        
        if len(diff_all_both_nonzero) > 0:
            data_to_plot.append(diff_all_both_nonzero)
            labels.append('所有都非零')
        
        # 分离相等和不相等
        if np.sum(both_nonzero) > 0:
            equal_indices = equal_both_nonzero[both_nonzero]
            not_equal_indices = ~equal_indices
            
            if np.sum(equal_indices) > 0:
                diff_equal = diff_all_both_nonzero[equal_indices]
                data_to_plot.append(diff_equal)
                labels.append('都非零且相等')
            
            if np.sum(not_equal_indices) > 0:
                diff_not_equal = diff_all_both_nonzero[not_equal_indices]
                data_to_plot.append(diff_not_equal)
                labels.append('都非零但不相等\n(从所有中分离)')
        
        bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True, showfliers=False)
        
        # 设置颜色
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
        
        ax4.set_ylabel('绝对值差')
        ax4.set_title('绝对值差箱线图对比')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_yscale('log')
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, '数据不足\n无法绘制箱线图', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)

    # 子图5: 散点图 d[i,j] vs d[j,i] (随机采样)
    ax5 = axes[1, 1]
    if count_case2 > 0:
        # 随机采样最多1000个点
        sample_size = min(1000, count_case2)
        indices = np.where(both_nonzero_not_equal)[0]
        if len(indices) > sample_size:
            sampled_indices = np.random.choice(indices, size=sample_size, replace=False)
        else:
            sampled_indices = indices
        
        sampled_d_ij = d_ij[sampled_indices]
        sampled_d_ji = d_ji[sampled_indices]
        
        ax5.scatter(sampled_d_ij, sampled_d_ji, alpha=0.6, s=20, c='red', edgecolors='black', linewidths=0.5)
        
        # 添加对角线
        min_val = min(sampled_d_ij.min(), sampled_d_ji.min())
        max_val = max(sampled_d_ij.max(), sampled_d_ji.max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'g--', linewidth=2, alpha=0.7, label='对角线')
        
        ax5.set_xlabel('d[i,j]')
        ax5.set_ylabel('d[j,i]')
        ax5.set_title(f'都非零不相等元素散点图 (n={sample_size:,})')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_aspect('equal', 'box')
        
        # 添加相关系数
        correlation = np.corrcoef(sampled_d_ij, sampled_d_ji)[0, 1]
        ax5.text(0.05, 0.95, f'相关系数: {correlation:.4f}', 
                transform=ax5.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax5.text(0.5, 0.5, '没有都非零不相等的数据\n无法绘制散点图', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)

    # 子图6: 差值与平均值的关系
    ax6 = axes[1, 2]
    if count_case2 > 0:
        # 计算差值和平均值
        indices = np.where(both_nonzero_not_equal)[0]
        diff_vals = np.abs(d_ij[indices] - d_ji[indices])
        avg_vals = (d_ij[indices] + d_ji[indices]) / 2
        
        # 随机采样最多1000个点
        sample_size = min(1000, len(diff_vals))
        if len(diff_vals) > sample_size:
            sample_idx = np.random.choice(len(diff_vals), size=sample_size, replace=False)
            diff_vals_sampled = diff_vals[sample_idx]
            avg_vals_sampled = avg_vals[sample_idx]
        else:
            diff_vals_sampled = diff_vals
            avg_vals_sampled = avg_vals
        
        scatter = ax6.scatter(avg_vals_sampled, diff_vals_sampled, alpha=0.6, s=20, 
                            c=diff_vals_sampled/avg_vals_sampled, cmap='viridis', 
                            edgecolors='black', linewidths=0.5, vmin=0, vmax=2)
        
        ax6.set_xlabel('平均值 (d[i,j] + d[j,i]) / 2')
        ax6.set_ylabel('绝对值差 |d[i,j] - d[j,i]|')
        ax6.set_title(f'差值 vs 平均值 (n={sample_size:,})')
        ax6.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('相对差异 (差值/平均值)', rotation=270, labelpad=15)
        
        # 设置对数坐标轴
        ax6.set_xscale('log')
        ax6.set_yscale('log')
        
        # 添加趋势线
        if len(avg_vals_sampled) > 1:
            # 使用对数空间进行线性拟合
            log_avg = np.log10(avg_vals_sampled[avg_vals_sampled > 0])
            log_diff = np.log10(diff_vals_sampled[avg_vals_sampled > 0])
            
            if len(log_avg) > 1:
                coeffs = np.polyfit(log_avg, log_diff, 1)
                poly = np.poly1d(coeffs)
                
                x_fit = np.logspace(np.log10(avg_vals_sampled.min()), 
                                np.log10(avg_vals_sampled.max()), 100)
                y_fit = 10**poly(np.log10(x_fit))
                
                ax6.plot(x_fit, y_fit, 'r-', linewidth=2, alpha=0.7, 
                        label=f'趋势: y ∝ x^{coeffs[0]:.2f}')
                ax6.legend()
    else:
        ax6.text(0.5, 0.5, '没有都非零不相等的数据\n无法绘制差值-平均值图', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)

    plt.suptitle('对称性绝对值差分布对比分析', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 4. 统计显著性检验
    print("\n" + "=" * 50)
    print("统计显著性检验")
    print("=" * 50)

    if len(diff_case2) > 0 and len(diff_all_both_nonzero) > 0:
        from scipy import stats
        
        print(f"\n1. Kolmogorov-Smirnov 检验:")
        ks_stat, ks_p = stats.ks_2samp(diff_case2, diff_all_both_nonzero)
        print(f"   KS统计量: {ks_stat:.6f}")
        print(f"   P值: {ks_p:.6e}")
        if ks_p < 0.05:
            print(f"   结论: 两种分布的差异显著 (p < 0.05)")
        else:
            print(f"   结论: 两种分布的差异不显著 (p >= 0.05)")
        
        print(f"\n2. Mann-Whitney U 检验 (非参数检验):")
        mw_stat, mw_p = stats.mannwhitneyu(diff_case2, diff_all_both_nonzero, alternative='two-sided')
        print(f"   U统计量: {mw_stat:.1f}")
        print(f"   P值: {mw_p:.6e}")
        if mw_p < 0.05:
            print(f"   结论: 两种分布的中位数差异显著 (p < 0.05)")
        else:
            print(f"   结论: 两种分布的中位数差异不显著 (p >= 0.05)")
        
        print(f"\n3. 效应量 (Cohen's d):")
        mean_diff = diff_case2.mean() - diff_all_both_nonzero.mean()
        pooled_std = np.sqrt((diff_case2.std()**2 + diff_all_both_nonzero.std()**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        print(f"   均值差: {mean_diff:.6f}")
        print(f"   合并标准差: {pooled_std:.6f}")
        print(f"   Cohen's d: {cohens_d:.6f}")
        if abs(cohens_d) < 0.2:
            print(f"   效应量: 很小")
        elif abs(cohens_d) < 0.5:
            print(f"   效应量: 小")
        elif abs(cohens_d) < 0.8:
            print(f"   效应量: 中等")
        else:
            print(f"   效应量: 大")
        
        print(f"\n4. 描述性统计对比:")
        print(f"                    都非零不相等        所有都非零")
        print(f"  样本量:         {len(diff_case2):12,}  {len(diff_all_both_nonzero):12,}")
        print(f"  均值:          {diff_case2.mean():12.6f}  {diff_all_both_nonzero.mean():12.6f}")
        print(f"  中位数:        {np.median(diff_case2):12.6f}  {np.median(diff_all_both_nonzero):12.6f}")
        print(f"  标准差:        {diff_case2.std():12.6f}  {diff_all_both_nonzero.std():12.6f}")
        print(f"  最小值:        {diff_case2.min():12.6f}  {diff_all_both_nonzero.min():12.6f}")
        print(f"  最大值:        {diff_case2.max():12.6f}  {diff_all_both_nonzero.max():12.6f}")
        
        # 计算偏度和峰度
        from scipy.stats import skew, kurtosis
        
        skew_case2 = skew(diff_case2) if len(diff_case2) > 0 else 0
        skew_all = skew(diff_all_both_nonzero) if len(diff_all_both_nonzero) > 0 else 0
        kurt_case2 = kurtosis(diff_case2) if len(diff_case2) > 0 else 0
        kurt_all = kurtosis(diff_all_both_nonzero) if len(diff_all_both_nonzero) > 0 else 0
        
        print(f"  偏度:          {skew_case2:12.6f}  {skew_all:12.6f}")
        print(f"  峰度:          {kurt_case2:12.6f}  {kurt_all:12.6f}")
    else:
        print("数据不足，无法进行统计显著性检验")