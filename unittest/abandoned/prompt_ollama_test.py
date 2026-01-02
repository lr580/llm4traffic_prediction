import sys, os, tqdm
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from prompt import LLMmodel, OllamaQuery, PromptHANeighbor, PromptHA, PromptPlain, PromptNeighbor
from utils.datasets import PEMSDatasetHandler, DataList, getDataResults
# modelname = 'llama3.2:latest'
# model_shortname = "llama32"
# modelname = 'deepseek-coder-v2:16b'
# model_shortname = "deepseekCoderV2-16b"
# modelname = 'deepseek-v2:16b'
# model_shortname = "deepseekv2-16b"
# modelname = 'qwen2.5:14b'
# model_shortname = 'qwen2.5-14b'
# modelname = 'qwen2.5:7b'
# model_shortname = 'qwen2.5-7b'
# modelname = 'qwen2.5:3b'
# model_shortname = 'qwen2.5-3b'
# modelname = 'qwen2.5:1.5b'
# model_shortname = 'qwen2.5-1.5b'
modelname = 'qwen2.5:0.5b'
model_shortname = 'qwen2.5-0.5b'
x = 3
tinyset = '32-1'
handler = PEMSDatasetHandler(x, loadData=True, stat=True)
prompts = [PromptPlain(), PromptNeighbor(), PromptHA(), PromptHANeighbor()]
resultList = []
resultNames = ['Plain', 'Neighbor', 'HA', 'HA_Nei']
for prompt in tqdm.tqdm(prompts, 'Models'):
    datalist = DataList.load(f'data/tiny/PEMS0{x}/{tinyset}.json')
    model = LLMmodel(OllamaQuery(model=modelname), prompt, handler, tinyset + model_shortname)
    model.tiny_test(datalist)
    resultList.append(datalist)
print(getDataResults(resultList, resultNames))