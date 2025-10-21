import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.baselines import Results, CitationHandler
from utils.common import pdDataFrame2str
results = Results.get_baseline_results()
results.flit([f'PEMS0{x}' for x in '3478'], horizons=[-1], tags='survey')
results.printSOTA()
df = results.dataset_view().sort_values(by=('PEMS03', 'mae'), ascending=False)
print(df)