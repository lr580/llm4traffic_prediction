from .dataset import *
from .graphHandler import *
from .timeHandler import *
from .statHandler import *
from ..metrics import plot_time_series, ndarray2plot
class DatasetHanlder():
    def __init__(self, stat=False, loadData=False): # to be implemented by subclass
        self.space = self.load_space()
        self.name = self.load_name()
        if loadData:
            self.dataset = Dataset()
        self.graph = Graph()
        self.timeCalc = TimeCalc()
        if stat:
            self.stat = HAstat(np.zeros((1,1),np.float32))
            
    def load_space(self): # to be implemented by subclass
        return 'undefined'
    
    def load_name(self): # to be implemented by subclass
        return 'undefined'
    
    def buildSingleInput(self, i:int, j:int, copy:bool):
        X, y = self.dataset.get_data(i, j, copy)
        time = self.timeCalc.getStartTime(i)
        singleInput = SingleInput(X=X, i=i, j=j, time=time)
        return SingleData(input=singleInput, y_true=y)
    
    def buildBatchInput(self, batch_size:int=16, copy=True):
        batch = []
        for _ in range(batch_size):
            i, j = self.dataset.get_random_index()
            batch.append(self.buildSingleInput(i, j, copy))
        return DataList(data=batch)
    
    def plotData(self, l, r, j, figsize=(8, 6)):
        '''plot time interval [l, r) of sensor j '''
        time = self.timeCalc.getStartTime(l)
        granularity = self.timeCalc.granularity
        X = np.squeeze(self.dataset.data[l:r, j, 0])
        plot_time_series(
            [ndarray2plot(X, time, 'flow', 'b', granularity=granularity)],
            title=f'Traffic Flow in time [{l}, {r}) sensor {j} {self.timeCalc.getWeek(l)}', figsize=figsize
        )

class PEMSDatasetHandler(DatasetHanlder):
    def load_space(self):
        # the information from https://www.sciencedirect.com/science/article/pii/S0957417422011654
        if self.x == 3:
            return '中北部区域(North Central Area)'
        elif self.x == 4:
            return '旧金山湾区(San Francisco Bay Area)'
        elif self.x == 7:
            return '洛杉矶区域(Los Angeles Area)'
        elif self.x == 8:
            return '圣贝纳迪诺区(San Bernardino Area)'
        
    def load_name(self):
        return f'PEMS0{self.x}'
    
    def __init__(self, x:int, stat=False, loadData=False):
        self.x = x
        if loadData:
            self.dataset = PEMSDataset(x)
        self.graph = PEMSGraph(x)
        self.timeCalc = PEMSTimeCalc(x)
        self.name = self.load_name()
        self.space = self.load_space()
        if stat:
            self.stat = PEMS_HAstat(self.dataset)