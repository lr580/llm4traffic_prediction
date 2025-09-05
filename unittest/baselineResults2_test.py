import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.baselines import Results
BASELINE_RESULTS = Results.from_csv('utils/baselines/baselineResults.csv')
print(BASELINE_RESULTS.flit(['PEMS04'], inner = False).sort())
print(BASELINE_RESULTS.rank())