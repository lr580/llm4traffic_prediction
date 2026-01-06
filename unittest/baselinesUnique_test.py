# 给 baselineResults.csv 去重
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.baselines import Results
baselines = Results.get_baseline_results()
baselines.deduplicate()
baselines.to_csv('utils/baselines/baselineResults2.csv')