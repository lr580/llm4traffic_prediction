from utils.datasets import DatasetHanlder, SingleInput
from utils.common import date2str, vec2str

class Prompt():
    TASK_DESC = '你需要完成交通流量预测任务。交通流量指的是一段时间内经过某个路口探测器的车辆的数目。'
    INPUT_DESC = '你有已知数据，为12个连续时间区间(每个时间区间的长度为5分钟)的流量。'
    SINGLE_OUTPUT_FORMAT_DESC = '请你按照Python列表格式输出流量，保留4位小数。你只需要输出答案本身，无需输出任何计算过程和解释理由。'
    # 在下面的输入里，探测器包含12个连续时间区间的流量数据，每个时间区间的长度为5分钟。
    def __init__(self):
        self.name = '' # to be implemented by subclass
        
    def generate(self, handler:DatasetHanlder, data:SingleInput) -> str:
        return '' # to be implemented by subclass
    
    def getTimePoints(self, handler:DatasetHanlder, data:SingleInput):
        ''' X ∈ [X_startTime, Y_startTime), Y ∈ [Y_startTime, Y_endTime) \n\n common helper function '''
        Y_startTime = handler.timeCalc.getStartTime(data.i)
        X_startTime = handler.timeCalc.getStartTime(data.i - data.X.size)
        Y_endTime = handler.timeCalc.getStartTime(data.i + data.X.size)
        return X_startTime, Y_startTime, Y_endTime
    
    def saySpace(self, handler:DatasetHanlder):
        return f'探测器位于{handler.space}。'
    
    def sayInputTimeRange(self, handler:DatasetHanlder, data:SingleInput):
        X_startTime, Y_startTime, _ = self.getTimePoints(handler, data)
        week = handler.timeCalc.getWeek(data.i)
        return f'该探测器从{date2str(X_startTime)}到{date2str(Y_startTime)}({week})的流量为:'
    
    def sayOutputTimeRange(self, handler:DatasetHanlder, data:SingleInput):
        _, Y_startTime, Y_endTime = self.getTimePoints(handler, data)
        return f'现在，请你预测接下来12个区间，即从{date2str(Y_startTime)}到{date2str(Y_endTime)}的流量。'

class PromptPlain(Prompt):
    '''单点，纯原始数据'''
    def __init__(self):
        self.name = 'plain'
        
    def generate(self, handler:DatasetHanlder, data:SingleInput) -> str:
        prompt = self.TASK_DESC
        prompt += self.saySpace(handler)
        prompt += self.INPUT_DESC + '\n'
        prompt += self.sayInputTimeRange(handler, data)
        prompt += vec2str(data.X) + '\n'
        prompt += self.sayOutputTimeRange(handler, data)
        prompt += self.SINGLE_OUTPUT_FORMAT_DESC
        return prompt
    
class PromptHA(Prompt):
    '''单点数据，提供历史平均作参考'''
    def __init__(self):
        self.name = 'HA'
        
    def sayInputHA(self, handler:DatasetHanlder, data: SingleInput):
        # ha = handler.stat.getRange(data.i - handler.dataset.T_input, data.i, data.j, handler.dataset.data)
        ha = data.getInputHA(handler)
        return f'这段时间以往的平均流量是{vec2str(ha)}，因此，输入流量距离平均值的偏差是{vec2str(data.X - ha)}。'
        
    def sayOutputHA(self, handler:DatasetHanlder, data: SingleInput):
        # ha = handler.stat.getRange(data.i, data.i + handler.dataset.T_output, data.j, handler.dataset.data)
        ha = data.getOutputHA(handler)
        return f'你要预测的12个区间的在过往的平均流量是{vec2str(ha)}。'
        
    def generate(self, handler:DatasetHanlder, data: SingleInput) -> str:
        prompt = self.TASK_DESC
        prompt += self.saySpace(handler)
        prompt += self.INPUT_DESC + '\n'
        prompt += self.sayInputTimeRange(handler, data)
        prompt += vec2str(data.X) + '\n'
        prompt += self.sayInputHA(handler, data) + '\n'
        prompt += self.sayOutputTimeRange(handler, data)
        prompt += self.sayOutputHA(handler, data) + '\n'
        prompt += self.SINGLE_OUTPUT_FORMAT_DESC
        return prompt