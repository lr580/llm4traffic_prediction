import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from prompt import LLMmodel, DeepseekQuery, PromptHANeighbor
from utils.datasets import PEMSDatasetHandler, DataList
# model = LLMmodel(DeepseekQuery(), PromptHANeighbor(), PEMSDatasetHandler(3, loadData=True, stat=True), '1-2')
# datalist = DataList.load('data/tiny/PEMS03/1-1.json')
# model.tiny_test(datalist, 3)
# model = LLMmodel(DeepseekQuery(), PromptHANeighbor(), PEMSDatasetHandler(3, loadData=True, stat=True), '32')
# datalist = DataList.load('data/tiny/PEMS03/32-1.json')
model = LLMmodel(DeepseekQuery(), PromptHANeighbor(), PEMSDatasetHandler(3, loadData=True, stat=True), '64')
datalist = DataList.load('data/tiny/PEMS03/64-1.json')
model.tiny_test(datalist)