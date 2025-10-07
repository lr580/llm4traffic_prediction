# 负责跨数据格式转换
from .handler import DatasetHanlder
import pandas as pd
from tqdm import tqdm
import numpy as np
import json

class BasicTS_TSLib_Converter:
    '''BasicTS 格式与 TSLib 格式转换 \n
    BasicTS https://github.com/GestaltCogTeam/BasicTS \n 
    TSLib https://github.com/thuml/Time-Series-Library \n
    see unittest/dataconert_test.py for usage example '''
    def __init__(self, basicTSdataHandler:DatasetHanlder):
        self.handler = basicTSdataHandler
        
    def toTSLib(self, destpath:str): #, yName:str='OT'):
        ''' 按 ETTm1 格式接收；可能需要几分钟，只转换时间序列，图不转换。
        使用多变量模式，多个空间点就看成是多个特征通道'''
        dataset = self.handler.dataset
        raw = dataset.data
        if True: # 优化到只需要数秒
            df = pd.DataFrame(
                data=dataset.data[:, :, 0],
                columns=[f'i_{n}' for n in range(dataset.n)]
            )
            df = df.assign(
                date=[self.handler.timeCalc.getStartTimeStr(t) for t in range(dataset.t)]
            )
            df = df[['date'] + [col for col in df.columns if col != 'date']]
        else: # 朴素暴力，9m10s 生成 PEMS03.csv 51MB
            cols = ['date'] + [f'i_{i}' for i in range(dataset.n)] #+ [yName]
            df = pd.DataFrame(columns=cols)
            for t in tqdm(range(dataset.t)):
                row = [self.handler.timeCalc.getStartTimeStr(t)]
                for n in range(dataset.n):
                    row.append(raw[t,n,0])
                df.loc[t] = row
        
        df.to_csv(destpath, index=False)

class BasicTS_Shrink:
    ''' 删除 BasicTS 数据的一些数据，得到微型数据集，用于快速跑实验 \n
    see unittest/datashrink_test.py for usage example'''
    @staticmethod
    def shrink(basicTSdataHandler:DatasetHanlder, data_destpath:str, json_destpath:str, name:str='', T_reserve_rate:float=0.1):
        dataset = basicTSdataHandler.dataset
        data = dataset.data

        L_new = int(dataset.t * 0.1)
        shape = list(data.shape)
        shape[0] = L_new
        shape = tuple(shape)
        dtype = data.dtype
        
        data_dest = np.memmap(data_destpath, dtype=dtype, mode='w+', shape=shape)
        data_dest[:] = data[:L_new, :, :]
        data_dest.flush()

        jsons = dataset.desc_json.copy()
        jsons['shape'] = list(shape)
        jsons['num_time_steps'] = L_new
        jsons['name'] = name
        with open(json_destpath, 'w') as f:
            f.write(json.dumps(jsons, indent=2))
