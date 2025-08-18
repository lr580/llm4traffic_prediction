import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from prompt import LLMmodel, DeepseekQuery, PromptPlain
from utils.datasets import PEMSDatasetHandler
for x in [3, 4, 7, 8]:
    handler = PEMSDatasetHandler(x, loadData=True)
    model = LLMmodel(DeepseekQuery('results/tmp'), PromptPlain(), handler)
    print(f'PEMS0{x}', model.cost())