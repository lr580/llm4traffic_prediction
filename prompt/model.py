from .query import CacheQuery
from .prompt import Prompt
from utils.datasets import DatasetHanlder, DataList
from utils.common import now2str
from utils.metrics import evaluate_average
import numpy as np
import os
from tqdm import tqdm
class LLMmodel():
    def __init__(self, query:CacheQuery, prompt:Prompt, handler:DatasetHanlder, runID=''):
        self.queryHandler = query
        self.prompt = prompt
        self.handler = handler
        if not runID:
            # runID = randomStr(6) # old scheme
            runID = now2str()
        self.runID = runID
        self.path = f'results/{prompt.name}/{handler.name}/{self.runID}'
        self.queryHandler.update_path(self.path)
        
    def output2vec(self, output:str):
        array = eval(output)
        return np.array(array)
    
    def tiny_test(self, datalist:DataList, verbose=2):
        for idx, data in tqdm(enumerate(datalist), total=len(datalist)):
            y_true = data.y_true
            i, j = data.input.i, data.input.j
            prompt = self.prompt.generate(self.handler, data.input)
            qid = f'{i}_{j}'
            output = self.queryHandler.query(qid, userMessage=prompt)
            y_pred = self.output2vec(output)
            data.y_pred = y_pred
            data.evaluate()
            if verbose>=2:
                print('index:', idx)
                if verbose>=3:
                    print('Prompt:', prompt)
                print('Answer:', y_true)
                print('Pred:', y_pred)
                print('Result:', data.result)
        datalist.evaluate()
        if verbose>=1:
            print('Average:', datalist.totalResult)
        datalist.save(os.path.join(self.path, 'results_tiny_test.json'))