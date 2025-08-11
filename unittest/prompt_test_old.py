import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from prompt import LLMmodel, DeepseekQuery, PromptPlain
from utils.datasets import PEMSDatasetHandler

model = LLMmodel(DeepseekQuery(), PromptPlain(), PEMSDatasetHandler(3), 'tiny1')
model.tiny_test(16)