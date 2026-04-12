import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.baselines import Results
results = Results.from_csv('utils/baselines/gwnet_rag.csv')
datasets = ['SD', 'GBA', 'GLA', 'CA'] + [f'PEMS0{x}' for x in '3478']
results.flit(datasets)
df = results.dataset_view(datasets=datasets).sort_values(by=('SD', 'mae'), ascending=False)
print(df)
