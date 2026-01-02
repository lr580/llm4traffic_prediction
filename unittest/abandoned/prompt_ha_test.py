import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from prompt import LLMmodel, DeepseekQuery, PromptHA
from utils.datasets import PEMSDatasetHandler, DataList

# model = LLMmodel(DeepseekQuery(), PromptHA(), PEMSDatasetHandler(3, loadData=True, stat=True), '1-1')
# datalist = DataList.load('data/tiny/PEMS03/1-1.json')
# model.tiny_test(datalist, 3)

# model = LLMmodel(DeepseekQuery(), PromptHA(), PEMSDatasetHandler(3, loadData=True, stat=True), '16-2')
# datalist = DataList.load('data/tiny/PEMS03/16-2.json')
# model.tiny_test(datalist)

# model = LLMmodel(DeepseekQuery(), PromptHA(), PEMSDatasetHandler(3, loadData=True, stat=True), '32-1')
# datalist = DataList.load('data/tiny/PEMS03/32-1.json')
# model.tiny_test(datalist)

model = LLMmodel(DeepseekQuery(), PromptHA(), PEMSDatasetHandler(3, loadData=True, stat=True), '64-1')
datalist = DataList.load('data/tiny/PEMS03/64-1.json')
model.tiny_test(datalist)