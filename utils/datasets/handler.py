from .dataset import *
from .graphHandler import *
from .timeHandler import *
from .statHandler import *
from .data import *
from ..metrics import plot_time_series, ndarray2plot
import datetime
class DatasetHandler():
    def __init__(self, stat=False, loadData=False): # to be implemented by subclass
        self.loads()
        if loadData:
            self.dataset = Dataset()
        self.graph = Graph()
        self.timeCalc = TimeCalc(datetime.datetime.now(), datetime.timedelta(minutes=5))
        if stat:
            self.stat = HAstat(np.zeros((1,1),np.float32))
            
    def load_space(self): # to be implemented by subclass
        return 'undefined'
    
    def load_name(self): # to be implemented by subclass
        return 'undefined'
    
    def loads(self):
        ''' 载入数据集名字、空间信息 '''
        self.space = self.load_space()
        self.name = self.load_name()
    
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
    
    def rebuildBatchInput(self, batch:DataList):
        '''历史版本可能缺失一些信息，进行补全'''
        for data in batch:
            if data.input.time is None:
                data.input.time = self.timeCalc.getStartTime(data.input.i)
    
    def getInputHA(self, input: SingleInput, nei=-1):
        if nei == -1:
            j = input.j
        else:
            j = nei
        return self.stat.getRange(input.i - self.dataset.T_input, input.i, j, self.dataset.data)
    
    def getOutputHA(self, input: SingleInput):
        return self.stat.getRange(input.i, input.i + self.dataset.T_output, input.j, self.dataset.data)
        
    def plotResultWithHA(self, data: SingleData, granularity=timedelta(minutes=5), show=True, savepath=''):
        # 这个参数，主要是不想把 HA 加到 dataclass 了，没啥必要
        plots = data.dataToPlot(granularity)
        input = data.input
        time = input.time
        assert time is not None
        ha_input = self.getInputHA(input)
        ha_pred = self.getOutputHA(input)
        plots.append(ndarray2plot(ha_input, time - ha_input.size * granularity, 'HA_input', 'orange', granularity=granularity))
        plots.append(ndarray2plot(ha_pred, time, 'HA_pred', 'purple', granularity=granularity))
        plot_time_series(plots, title = f'{input.i} - {input.j} Prediction Result (HA)', show=show, savepath=savepath)
        
    def getPlotData(self, l:int, r:int, j:int, figsize=(8, 6)):
        time = self.timeCalc.getStartTime(l)
        granularity = self.timeCalc.granularity
        X = np.squeeze(self.dataset.data[l:r, j, 0])
        return [ndarray2plot(X, time, 'flow', 'b', granularity=granularity)]
        
    def plotData(self, l:int, r:int, j:int, figsize=(8, 6)):
        '''plot time interval [l, r) of sensor j '''
        plot_time_series(self.getPlotData(l, r, j, figsize), title=f'Traffic Flow in time [{l}, {r}) sensor {j} {self.timeCalc.getWeek(l)}', figsize=figsize)
        
    def plotDataWithHA(self, l:int, r:int, j:int, figsize=(8, 6)):
        '''plot time interval [l, r) of sensor j with HA'''
        plots = self.getPlotData(l, r, j, figsize)
        ha = self.stat.getRange(l, r, j, self.dataset.data)
        plots.append(ndarray2plot(ha, self.timeCalc.getStartTime(l), 'HA', 'orange', granularity=self.timeCalc.granularity))
        plot_time_series(plots, title=f'Traffic Flow in time [{l}, {r}) sensor {j} {self.timeCalc.getWeek(l)}', figsize=figsize)

class PEMSDatasetHandler(DatasetHandler):
    def load_space(self):
        # the information from https://www.sciencedirect.com/science/article/pii/S0957417422011654
        s = '美国加州(California, USA)的'
        if self.x == 3:
            s += '中北部区域(North Central Area)'
        elif self.x == 4:
            s += '旧金山湾区(San Francisco Bay Area)'
        elif self.x == 7:
            s += '洛杉矶区域(Los Angeles Area)'
        elif self.x == 8:
            s += '圣贝纳迪诺区(San Bernardino Area)'
        return s
        # 此外，点数分别是 [358, 307, 883, 170] 在数据集可以验证。
        
    def load_name(self):
        return f'PEMS0{self.x}'
    
    def __init__(self, x:int, stat=False, loadData=False):
        self.x = x
        if loadData:
            # for i in tqdm(range(1), 'loading data'):
            self.dataset = PEMSDataset(x)
        self.graph = PEMSGraph(x)
        self.timeCalc = PEMSTimeCalc(x)
        self.loads()
        if stat:
            self.stat = PEMS_HAstat(self.dataset)
            
class LargeSTDatasetHandler(DatasetHandler):
    def load_space(self):
        ''' LargeST paper https://proceedings.neurips.cc/paper_files/paper/2023/file/ee57cd73a76bd927ffca3dda1dc3b9d4-Paper-Datasets_and_Benchmarks.pdf '''
        s = '美国加州(California, USA)'
        match self.dataset:
            case 'GLA':
                s += '的大洛杉矶区域(GLA, Greater Los Angeles)，包含Los Angeles, Orange, Riverside, San Bernardino, and Ventura'
            case 'GBA':
                s += '的旧金山大湾区(GBA, Greater Bay Area)，包含Alameda, Contra Costa, Marin, Napa, San Benito, San Francisco, San Mateo, Santa Clara, Santa Cruz, Solano, and Sonoma'
            case 'SD':
                s += '的圣地亚哥区域(SD, San Diego)'
        return s
        # 此外，点数分别是 {'CA':8600, 'GBA': 3834, 'GLA': 2352, 'SD': 716} 在数据集可以验证。
        
    def load_name(self):
        return self.dataset
    
    def __init__(self, dataset:int, stat=False, loadData=False):
        self.dataset = dataset
        if loadData:
            self.dataset = BasicTSDataset(dataset)
        self.graph = BasicTSGraph(dataset)
        self.timeCalc = LargeSTTimeCalc()
        if stat:
            self.stat = BasicTSHAstat(self.dataset)
        self.loads()