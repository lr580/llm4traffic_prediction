import sys, os, tqdm
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from prompt import LLMmodel, DeepseekQuery, PromptHANeighbor, PromptHA, PromptPlain, PromptNeighbor
from utils.datasets import PEMSDatasetHandler, DataList
x = 3
tinyset = '1-1'
datalist = DataList.load(f'data/tiny/PEMS0{x}/{tinyset}.json')
prompts = [PromptPlain(), PromptNeighbor(), PromptHA(), PromptHANeighbor()]
handler = PEMSDatasetHandler(x, loadData=True, stat=True)
for prompt in tqdm.tqdm(prompts, 'Models'):
    model = LLMmodel(DeepseekQuery(), prompt, handler, tinyset)
    model.tiny_test(datalist, 3)