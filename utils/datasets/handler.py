from .dataset import *
from .graphHandler import *
from .timeHandler import *
from .statHandler import *
class DatasetHanlder():
    def __init__(self, stat=False): # to be implemented by subclass
        self.space = self.load_space()
        self.name = self.load_name()
        self.dataset = Dataset()
        self.graph = Graph()
        self.timeCalc = TimeCalc()
        if stat:
            self.stat = HAstat(np.zeros((1,1),np.float32))
            
    def load_space(self): # to be implemented by subclass
        return 'undefined'
    
    def load_name(self): # to be implemented by subclass
        return 'undefined'

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
    
    def __init__(self, x:int, stat=False):
        self.x = x
        self.dataset = PEMSDataset(x)
        self.graph = PEMSGraph(x)
        self.timeCalc = PEMSTimeCalc(x)
        self.name = self.load_name()
        self.space = self.load_space()
        if stat:
            self.stat = PEMS_HAstat(self.dataset)