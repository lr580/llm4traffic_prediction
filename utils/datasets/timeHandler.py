import datetime
from utils.common import date2str
class TimeCalc():
    def __init__(self, startTime:datetime.datetime, granularity:datetime.timedelta):
        self.startTime = startTime
        self.granularity = granularity
        
    def getStartTime(self, i:int) -> datetime.datetime:
        '''i is 0-indexed'''
        return self.startTime + i * self.granularity
    def getStartTimeStr(self, i:int):
        return date2str(self.getStartTime(i))
    
    def getWeek(self, i:int):
        return self.getStartTime(i).strftime("%A") # e.g. Monday
        
class PEMSTimeCalc(TimeCalc):
    def __init__(self, x):
        granularity = datetime.timedelta(minutes=5)
        # start time sees any survey (e.g. paper https://arxiv.org/pdf/2101.11174 )
        if x == 3:
            startTime = datetime.datetime(2018, 9, 1, 0, 0, 0)
        elif x == 4:
            startTime = datetime.datetime(2018, 1, 1, 0, 0, 0)
        elif x == 7:
            startTime = datetime.datetime(2017, 5, 1, 0, 0, 0)
            # 数据集实际98天 描述的区间是123天；此对不上的问题，在 LibCity 中亦有记载 https://github.com/LibCity/Bigscity-LibCity-Datasets/blob/master/pemsd7.py
        elif x == 8:
            startTime = datetime.datetime(2016, 7, 1, 0, 0, 0)
        else:
            raise ValueError("Invalid dataset number")
        super().__init__(startTime, granularity)
        
if __name__ == "__main__":
    tc = PEMSTimeCalc(3)
    print(tc.getStartTimeStr(0))
    print(tc.getStartTimeStr(26207))
    tc = PEMSTimeCalc(4)
    print(tc.getStartTimeStr(0))
    print(tc.getStartTimeStr(16991))
    tc = PEMSTimeCalc(7)
    print(tc.getStartTimeStr(0))
    print(tc.getStartTimeStr(28223))
    tc = PEMSTimeCalc(8)
    print(tc.getStartTimeStr(0))
    print(tc.getStartTimeStr(17855))