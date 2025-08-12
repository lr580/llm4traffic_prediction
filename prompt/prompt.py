from utils.datasets import DatasetHanlder, SingleInput
import numpy as np
import datetime
from utils.common import date2str


TASK_DESC = '你需要完成交通流量预测任务。交通流量指的是一段时间内经过某个路口探测器的车辆的数目。'
SINGLE_OUTPUT_FORMAT_DESC = '请你按照Python列表格式输出流量，保留4位小数。你只需要输出答案本身，无需输出任何计算过程和解释理由。'
# 在下面的输入里，探测器包含12个连续时间区间的流量数据，每个时间区间的长度为5分钟。

def vec2str(X:np.ndarray):
    strs = []
    for x in X:
        strs.append('%.4f'%x)
    return ', '.join(strs)

class Prompt():
    def __init__(self):
        self.name = 'abstract' # to be implemented by subclass
        
    def generate(self, handler:DatasetHanlder, data:SingleInput):
        return '' # to be implemented by subclass

class PromptPlain(Prompt):
    '''单点，纯原始数据'''
    def __init__(self):
        self.name = 'plain'
        
    def generate(self, handler:DatasetHanlder, data:SingleInput):
        prompt = TASK_DESC
        endTime = handler.timeCalc.getStartTime(data.i)
        startTime = handler.timeCalc.getStartTime(data.i - data.X.size)
        futureTime = handler.timeCalc.getStartTime(data.i + data.X.size)
        week = handler.timeCalc.getWeek(data.i)
        prompt += f'探测器位于{handler.space}。'
        prompt += f'你有已知数据，为12个连续时间区间(每个时间区间的长度为5分钟)的流量。\n'
        prompt += f'该探测器从{date2str(startTime)}到{date2str(endTime)}({week})的流量为:'
        prompt += vec2str(data.X)
        prompt += f'\n现在，请你预测接下来12个区间，即从{date2str(endTime)}到{date2str(futureTime)}的流量。'
        prompt += SINGLE_OUTPUT_FORMAT_DESC
        return prompt