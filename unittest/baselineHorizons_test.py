import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.baselines import Results
BASELINE_RESULTS = Results.from_csv('utils/baselines/baselineResults.csv')
BASELINE_RESULTS.flit(['PEMS04'], horizons=[3,6,12])
print(BASELINE_RESULTS.horizon_view().sort_values(by=(3, 'mae')))