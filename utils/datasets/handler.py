from .dataset import *
from .graphHandler import *
from .timeHandler import *
from .statHandler import *
class DatasetHanlder():
    def __init__(self, stat=False): # to be implemented by subclass
        self.dataset = Dataset()
        self.graph = Graph()
        self.timeCalc = TimeCalc()
        if stat:
            self.stat = HAstat(np.zeros((1,1),np.float32))

class PEMSDatasetHandler(DatasetHanlder):
    def __init__(self, x:int, stat=False):
        self.dataset = PEMSDataset(x)
        self.graph = PEMSGraph(x)
        self.timeCalc = PEMSTimeCalc(x)
        if stat:
            self.stat = PEMS_HAstat(self.dataset)