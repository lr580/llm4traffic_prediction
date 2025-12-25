import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.datasets import PEMSDatasetHandler

choice = 'basicts'
match choice:
    case 'basicts':
        for x in [3,4,7,8]:
            handler = PEMSDatasetHandler(x, loadData=True)
            data = handler.dataset.data
            means = data[:, :, 0].mean(axis=0)
            print(f'\nPEMS0{x}')
            print(means)
            ''' 结论： PEMS04, 08 BasicTS 与 ASTGCN 一样，都没有做减去 mean 的预处理
            其中，ASTGCN 的参见 pems0x_preview_raw.py '''
    case 'libcity':
        ''' LibCity 数据下载自 https://github.com/LibCity/Bigscity-LibCity?tab=readme-ov-file ，放在 data/LibCity/数据集文件夹名
        数据源和处理代码见 https://github.com/LibCity/Bigscity-LibCity-Datasets，注释表明，D4/8 取自 ASTGCN，D3/7 取自 STSGCN
        阅读数据，PEMS03 的 .rel 用了 ID；其他都是原本的 0-indexed 无法溯源 ID，这个 (pems03) ID 在 pems 官网能找到 
        https://pems.dot.ca.gov/ 登录后左边 quick links 选 D3，然后点 stations 可以导出全部。
        .geo 没有提供点的坐标信息，是空 list；感觉就是换了个格式的 ASTGCN + STSGCN'''