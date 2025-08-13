import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from prompt import LLMmodel, DeepseekQuery, PromptHANeighbor, PromptHA, PromptPlain, PromptNeighbor
from utils.datasets import PEMSDatasetHandler, DataList
x = 3
tinyset = '32-1'
datalist = DataList.load(f'data/tiny/PEMS0{x}/{tinyset}.json')
prompt = PromptHANeighbor()
handler = PEMSDatasetHandler(x, loadData=True, stat=True)
model = LLMmodel(DeepseekQuery(model='deepseek-reasoner'), prompt, handler, tinyset+'r')
model.tiny_test(datalist)
    
# datalist = DataList.load(f'data/tiny/PEMS04/{tinyset}.json')
# datalist = DataList.load('data/tiny/PEMS03/16-3.json')