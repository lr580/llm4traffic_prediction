import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from prompt import LLMmodel, DeepseekQuery, PromptPlain, PromptTimeLLM, PromptTimeLLMWithHANei, PromptHANeighbor
from utils.datasets import PEMSDatasetHandler, DataList

# model = LLMmodel(DeepseekQuery(), PromptPlain(), PEMSDatasetHandler(3, loadData=True), '32-tl') # 对比实验
# model = LLMmodel(DeepseekQuery(), PromptTimeLLM(), PEMSDatasetHandler(3, loadData=True), '32-tl') 
# model = LLMmodel(DeepseekQuery(), PromptTimeLLMWithHANei(), PEMSDatasetHandler(3, loadData=True, stat=True), '32-tl') 
model = LLMmodel(DeepseekQuery(), PromptHANeighbor(), PEMSDatasetHandler(3, loadData=True, stat=True), '32-tl') 
datalist = DataList.load('data/tiny/PEMS03/32-5.json')
# model.tiny_test(datalist, verbose=3)
model.tiny_test(datalist)

# Plain Average: EvaluateResult(mae=28.645971298217773, mape=0.2037845104932785, rmse=44.037010192871094) 
# TimeLLM Average: EvaluateResult(mae=25.299345016479492, mape=0.19872832298278809, rmse=36.0983772277832)
# TimeLLM+HANei Average: EvaluateResult(mae=21.867490768432617, mape=0.17460091412067413, rmse=32.664939880371094)
# HANei Average: EvaluateResult(mae=19.084497451782227, mape=0.15802296996116638, rmse=28.001447677612305) 