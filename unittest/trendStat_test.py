import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.datasets import TrendCalc

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print(TrendCalc().lags(x, 5))
print(TrendCalc().trendStat(x))