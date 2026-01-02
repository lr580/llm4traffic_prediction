# 标准基于提示词的小批量预测，可以修改配置来选择不同的模型、提示词、数据集等
# 更多不同设置参见 unittest/abandoned/prompt_*.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from prompt import LLMmodel, OpenAIQuery, PromptHANeighbor
from utils.datasets import PEMSDatasetHandler, DataList

prompt = PromptHANeighbor()
handler = PEMSDatasetHandler(3, loadData=True, stat=True)
datalist = DataList.load('data/tiny/PEMS03/32-5.json')
model_scheme = 'gpt-5.2'
match model_scheme:
    case 'gpt-5.2':
        query = OpenAIQuery(name='renice')
        runid = '32-5-gpt5-2'
    case _: # deepseek
        query = OpenAIQuery(name='deepseek')
        runid = '32-5a'
model = LLMmodel(query, prompt, handler, runid)
model.tiny_test(datalist)
