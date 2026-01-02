import sys, os, tqdm
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from prompt import LLMmodel, DeepseekQuery, PromptHANeighbor, PromptHA, PromptPlain, PromptNeighbor
from utils.datasets import PEMSDatasetHandler, DataList
for x in [3,4,7,8]:
    tinyset = '32-1'
    datalist = DataList.load(f'data/tiny/PEMS0{x}/{tinyset}.json')
    prompts = [PromptPlain(), PromptNeighbor(), PromptHA(), PromptHANeighbor()]
    for prompt in tqdm.tqdm(prompts, 'Models'):
        handler = PEMSDatasetHandler(x, loadData=True, stat=True)
        model = LLMmodel(DeepseekQuery(), prompt, handler, tinyset)
        model.tiny_test(datalist)
    
# datalist = DataList.load(f'data/tiny/PEMS04/{tinyset}.json')
# datalist = DataList.load('data/tiny/PEMS03/16-3.json')