import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
import torch
from utils.metrics.evaluation import evaluate_average
y_pred = torch.tensor([[[1.,2,3],[4,5,6],[7,8,9]],[[1.,2,3],[4,5,6],[7,8,9]]])
y_true = torch.tensor([[[1.,2,3],[4,5,6],[7,8,9]],[[1.,2,3],[4,5,6],[7,8,9+18]]])
a,b,c = evaluate_average(y_pred, y_true)
print(a,b,c)
a,b,c = evaluate_average(y_pred, y_true, False)
print(a,b,c)