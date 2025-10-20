import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.baselines import Results
BASELINE_RESULTS = Results.get_baseline_results()
df = BASELINE_RESULTS.flit(['PEMS04'], inner = False).sort('mape')
print(df[['model', 'mae', 'mape', 'rmse']].to_string(index=False))