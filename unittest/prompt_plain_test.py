import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from prompt import LLMmodel, DeepseekQuery, PromptPlain
from utils.datasets import PEMSDatasetHandler, DataList

model = LLMmodel(DeepseekQuery(), PromptPlain(), PEMSDatasetHandler(3), '16-2')
datalist = DataList.load('data/tiny/PEMS03/16-2.json')
model.tiny_test(datalist)