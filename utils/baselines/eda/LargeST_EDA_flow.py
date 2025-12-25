''' LargeST 数据集：https://github.com/liuxu77/LargeST 下载放在 data/LargeST/ '''
import numpy as np
import h5py
import pandas as pd

path = 'data/LargeST/ca_his_raw_2019.h5'
with h5py.File(path, 'r') as f:
    # 查看文件结构
    print("文件结构：")
    def print_structure(name, obj):
        print(name, type(obj))
    f.visititems(print_structure)
    '''t <class 'h5py._hl.group.Group'> 主组
    t/axis0 <class 'h5py._hl.dataset.Dataset'> 数据集
    t/axis1 <class 'h5py._hl.dataset.Dataset'>
    t/block0_items <class 'h5py._hl.dataset.Dataset'>
    t/block0_values <class 'h5py._hl.dataset.Dataset'>'''

'''数据集: axis0
形状: (8600,)
数据类型: |S9
前10个元素:
[b'317802' b'312134' b'312133' b'313159' b'319767' b'319780' b'317830'
 b'314876' b'314886' b'314909']
 
数据集: axis1
形状: (105120,)
数据类型: int64
前10个元素:
[1546300800000000000 1546301100000000000 1546301400000000000
 1546301700000000000 1546302000000000000 1546302300000000000
 1546302600000000000 1546302900000000000 1546303200000000000
 1546303500000000000]
即纳秒时间戳，对应2019/1/1 ...

数据集: block0_items
形状: (8600,)
数据类型: |S9
前10个元素:
[b'317802' b'312134' b'312133' b'313159' b'319767' b'319780' b'317830'
 b'314876' b'314886' b'314909']

数据集: block0_values
形状: (105120, 8600)
数据类型: float64
预览数据形状: (5, 5)
[[15. 15. 56. 56. 56.]
 [16. 16. 57. 57. 57.]
 [16. 16. 57. 57. 57.]
 [40. 28. 43. 66. 68.]
 [42. 14. 31. 40. 48.]]

105120 = 粒度为 5 的共一年的时间片。
'''

def inspect_h5_dataset(file_path, group_name='t'):
    """
    查看H5文件中的数据集信息
    Args:
        file_path: H5文件路径
        group_name: 组名，默认为't'
    """
    with h5py.File(file_path, 'r') as f:
        print(f"=== 文件: {file_path} ===")
        print(f"组: {group_name}")
        print("-" * 50)
        
        # 检查组是否存在
        if group_name not in f:
            print(f"错误: 组 '{group_name}' 不存在于文件中")
            return
        
        group = f[group_name]
        
        # 遍历组中的所有数据集
        for name, obj in group.items():
            if isinstance(obj, h5py.Dataset):
                print(f"\n数据集: {name}")
                print(f"形状: {obj.shape}")
                print(f"数据类型: {obj.dtype}")

                # 仅抓取必要的切片，避免一次性加载巨大数组
                try:
                    if len(obj.shape) == 0:
                        preview = obj[()]
                        print("标量值:")
                        print(preview)
                    elif len(obj.shape) == 1:
                        count = min(10, obj.shape[0])
                        print(f"前{count}个元素:")
                        preview = obj[:count]
                        print(preview)
                    else:
                        # 默认展示前几行几列，其余维度仅取首个切片
                        slices = []
                        for axis, dim in enumerate(obj.shape):
                            if axis == 0:
                                limit = min(5, dim)
                            elif axis == 1:
                                limit = min(5, dim)
                            else:
                                limit = 1
                            slices.append(slice(0, limit))
                        preview = obj[tuple(slices)]
                        print(f"预览数据形状: {preview.shape}")
                        print(preview)
                except MemoryError:
                    print("⚠️ 数据集过大，切片时仍然触发内存限制。")
                except Exception as e:
                    print(f"⚠️ 读取数据时出错: {e}")

                print("-" * 30)

path = 'data/LargeST/ca_his_raw_2019.h5'
inspect_h5_dataset(path, 't')

# explore_h5_file(path)
# summary = get_h5_summary(path)
# if 'error' not in summary:
#     print(f"📁 文件: {summary['file_path']}")
#     print(f"📁 总组数: {summary['total_groups']}")
#     print(f"📊 总数据集数: {summary['total_datasets']}")
#     print("\n📁 组列表:")
#     for group in summary['groups']:
#         print(f"  - {group['name']} (包含 {group['num_items']} 个项目)")
#     print("\n📊 数据集列表:")
#     for dataset in summary['datasets']:
#         print(f"  - {dataset['name']}: 形状={dataset['shape']}, 类型={dataset['dtype']}")
# else:
#     print(f"错误: {summary['error']}")


        # # 尝试将数据组合成DataFrame查看（如果可能）
        # print("\n" + "="*50)
        # print("尝试组合成DataFrame:")
        # print("="*50)
        
        # # 检查是否有必要的数据集来创建DataFrame
        # needed_datasets = ['block0_items', 'block0_values', 'axis0', 'axis1']
        # if all(ds in group for ds in needed_datasets):
        #     try:
        #         # 读取各数据集
        #         columns = group['block0_items'][:].astype(str)
        #         values = group['block0_values'][:]
        #         axis0 = group['axis0'][:].astype(str)
        #         axis1 = group['axis1'][:].astype(str)
                
        #         print(f"列名 (block0_items): {columns}")
        #         print(f"行索引 (axis0): {axis0}")
        #         print(f"列索引 (axis1): {axis1}")
        #         print(f"\n数据形状 (block0_values): {values.shape}")
                
        #         # 创建DataFrame
        #         if len(values.shape) == 2:
        #             df = pd.DataFrame(values, index=axis0, columns=columns)
        #             print(f"\nDataFrame 形状: {df.shape}")
        #             print("\nDataFrame 前5行:")
        #             print(df.head())
        #             print("\nDataFrame 列名和数据类型:")
        #             print(df.dtypes)
        #         else:
        #             print(f"block0_values 的形状 {values.shape} 不适合直接转换为DataFrame")
                    
        #     except Exception as e:
        #         print(f"创建DataFrame时出错: {e}")

# def get_h5_summary(file_path):
#     """
#     获取HDF5文件的简要统计信息
#     """
#     try:
#         with h5py.File(file_path, 'r') as f:
#             summary = {
#                 'file_path': file_path,
#                 'total_groups': 0,
#                 'total_datasets': 0,
#                 'groups': [],
#                 'datasets': []
#             }
            
#             def collect_info(name, obj):
#                 if isinstance(obj, h5py.Group):
#                     summary['total_groups'] += 1
#                     summary['groups'].append({
#                         'name': name,
#                         'type': 'group',
#                         'num_items': len(obj)
#                     })
#                 elif isinstance(obj, h5py.Dataset):
#                     summary['total_datasets'] += 1
#                     summary['datasets'].append({
#                         'name': name,
#                         'type': 'dataset',
#                         'shape': obj.shape,
#                         'dtype': str(obj.dtype),
#                         'size': obj.size
#                     })
#                 return None
            
#             f.visititems(collect_info)
            
#             return summary
            
#     except Exception as e:
#         return {'error': str(e)}

# def explore_h5_file(file_path, max_items=5, max_values=10):
#     """
#     探索HDF5文件的结构和基本特征
    
#     参数:
#     - file_path: HDF5文件路径
#     - max_items: 每个数据集显示的最大项目数
#     - max_values: 显示的最大数值数量
#     """
    
#     def print_dataset_info(name, obj, indent=0):
#         """打印数据集或组的详细信息"""
#         indent_str = "  " * indent
        
#         if isinstance(obj, h5py.Group):
#             print(f"{indent_str}📁 组: {name}")
#             return True  # 继续遍历子项
            
#         elif isinstance(obj, h5py.Dataset):
#             print(f"{indent_str}📊 数据集: {name}")
#             print(f"{indent_str}  ├─ 形状: {obj.shape}")
#             print(f"{indent_str}  ├─ 数据类型: {obj.dtype}")
#             print(f"{indent_str}  ├─ 维度数: {len(obj.shape)}")
#             print(f"{indent_str}  ├─ 总元素数: {np.prod(obj.shape):,}")
            
#             # 计算内存大小
#             element_size = obj.dtype.itemsize
#             total_size = np.prod(obj.shape) * element_size
#             print(f"{indent_str}  ├─ 内存大小: {total_size:,} 字节 ({total_size/1024/1024:.2f} MB)")
            
#             # 显示属性
#             if obj.attrs:
#                 print(f"{indent_str}  ├─ 属性: {len(obj.attrs)} 个")
#                 for attr_name in list(obj.attrs.keys())[:max_items]:
#                     attr_value = obj.attrs[attr_name]
#                     print(f"{indent_str}  │    {attr_name}: {attr_value}")
            
#             # 显示部分数据
#             try:
#                 data = obj[:]
#                 if obj.size > 0:
#                     if len(obj.shape) == 1:  # 一维数据
#                         print(f"{indent_str}  └─ 前{min(max_values, len(data))}个值:")
#                         for i, val in enumerate(data[:max_values]):
#                             if i < 5:  # 只显示前5个值的完整信息
#                                 print(f"{indent_str}      [{i}]: {val}")
#                         if len(data) > max_values:
#                             print(f"{indent_str}      ... 还有 {len(data)-max_values} 个值")
                            
#                     elif len(obj.shape) == 2:  # 二维数据
#                         print(f"{indent_str}  └─ 数据预览:")
#                         rows_to_show = min(3, obj.shape[0])
#                         cols_to_show = min(5, obj.shape[1])
#                         for i in range(rows_to_show):
#                             row_preview = data[i, :cols_to_show]
#                             print(f"{indent_str}      行 {i}: {row_preview}")
#                         if obj.shape[0] > rows_to_show or obj.shape[1] > cols_to_show:
#                             print(f"{indent_str}      ... 形状: {obj.shape}")
                    
#                     # 显示统计信息
#                     if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
#                         print(f"{indent_str}  📈 统计信息:")
#                         print(f"{indent_str}      ├─ 最小值: {np.nanmin(data):.4f}")
#                         print(f"{indent_str}      ├─ 最大值: {np.nanmax(data):.4f}")
#                         print(f"{indent_str}      ├─ 平均值: {np.nanmean(data):.4f}")
#                         print(f"{indent_str}      └─ 标准差: {np.nanstd(data):.4f}")
                        
#             except Exception as e:
#                 print(f"{indent_str}  ⚠ 无法读取数据: {e}")
            
#             print()  # 空行分隔
#             return False  # 不继续遍历（已经是数据集）
    
#     print("=" * 60)
#     print(f"🔍 分析文件: {file_path}")
#     print("=" * 60)
    
#     try:
#         with h5py.File(file_path, 'r') as f:
#             # 打印文件基本信息
#             print(f"📁 文件: {file_path}")
#             print(f"├─ 文件模式: {f.mode}")
#             print(f"├─ 驱动: {f.driver}")
#             print(f"└─ 库版本: {h5py.version.hdf5_version}")
#             print()
            
#             # 递归遍历所有组和数据集
#             print("📁 文件结构:")
#             f.visititems(lambda name, obj: print_dataset_info(name, obj))
            
#             # 显示文件的所有组
#             print("📁 顶级组和数据集:")
#             def print_item(name, obj, indent=0):
#                 indent_str = "  " * indent
#                 if isinstance(obj, h5py.Group):
#                     print(f"{indent_str}📁 {name}/")
#                     for key in obj.keys():
#                         print_item(f"{name}/{key}", obj[key], indent + 1)
#                 else:
#                     print(f"{indent_str}📊 {name} (shape: {obj.shape}, dtype: {obj.dtype})")
            
#             for name in f:
#                 print_item(name, f[name])
                
#     except Exception as e:
#         print(f"❌ 无法打开文件: {e}")
