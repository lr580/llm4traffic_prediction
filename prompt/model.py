from .query import *
from .prompt import *
from utils.datasets.handler import *
from utils.common import now2str, randomStr
from utils.metrics.evaluation import evaluate_average
import numpy as np
class LLMmodel():
    def __init__(self, query:CacheQuery, prompt:Prompt, data:DatasetHanlder, runID=''):
        self.queryHandler = query
        self.prompt = prompt
        self.data = data
        if not runID:
            # runID = randomStr(6) # old scheme
            runID = now2str()
        self.runID = runID
        self.path = f'results/{prompt.name}/{data.dataset.name}/{self.runID}'
        self.queryHandler.update_path(self.path)
        
    def output2vec(self, output:str):
        array = eval(output)
        return np.array(array)
    
    def tiny_test(self, batch_size=16, verbose=2):
        Xs, ys, indices = self.data.dataset.get_random_batch(batch_size)
        y_preds, y_trues = [], []
        for b in range(batch_size):
            X = Xs[b]
            y_true = ys[b]
            i, j = indices[b]
            prompt = self.prompt.generate(self.data, X, i, j)
            qid = f'{i}_{j}'
            output = self.queryHandler.query(qid, userMessage=prompt)
            y_pred = self.output2vec(output)
            if verbose>=2:
                print(prompt)
                print('Answer: ', y_true)
                print('Pred:', y_pred)
                evaluate_average(y_pred, y_true)
            y_preds.append(y_pred)
            y_trues.append(y_true)
        y_preds = np.stack(y_preds, axis=0)
        y_trues = np.stack(y_trues, axis=0)
        if verbose>=1:
            print('Average:')
            evaluate_average(y_preds, y_trues)
        
        