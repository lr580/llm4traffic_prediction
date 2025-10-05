# 负责跨数据格式转换
from .handler import DatasetHanlder
import pandas as pd
from tqdm import tqdm
class BasicTS_TSLib_Converter:
    '''BasicTS 格式与 TSLib 格式转换
    BasicTS https://github.com/GestaltCogTeam/BasicTS
    TSLib https://github.com/thuml/Time-Series-Library'''
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
        
# see unittest/dataconert_test.py for usage example