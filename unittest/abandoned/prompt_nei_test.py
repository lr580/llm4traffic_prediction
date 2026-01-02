import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from prompt import LLMmodel, DeepseekQuery, PromptNeighbor
from utils.datasets import PEMSDatasetHandler, DataList

model = LLMmodel(DeepseekQuery(model='deepseek-reasoner'), PromptNeighbor(), PEMSDatasetHandler(3, loadData=True), '1-r')
datalist = DataList.load('data/tiny/PEMS03/1-1.json')
# model = LLMmodel(DeepseekQuery(), PromptNeighbor(), PEMSDatasetHandler(3, loadData=True), '1-6')
# datalist = DataList.load('data/tiny/PEMS03/1-1.json')
# model = LLMmodel(DeepseekQuery(), PromptNeighbor(), PEMSDatasetHandler(3, loadData=True), '32-2')
# datalist = DataList.load('data/tiny/PEMS03/32-1.json')
model.tiny_test(datalist, 3)