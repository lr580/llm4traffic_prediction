import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.baselines import Results
BASELINE_RESULTS = Results.from_csv('utils/baselines/baselineResults.csv')
# print(BASELINE_RESULTS.flit(['PEMS03'], inner = False).sort())
# print(BASELINE_RESULTS.rank())
df = BASELINE_RESULTS.flit(['PEMS07'], inner = False).sort('mae')
print(df[['model', 'mae', 'mape', 'rmse']].to_string(index=False))