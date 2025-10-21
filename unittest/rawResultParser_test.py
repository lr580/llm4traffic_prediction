import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.baselines import ParserBasicTS, Results
# downloaded in STD-MAE https://github.com/Jimmy-7664/STD-MAE
paths = {
    'PEMS03':r'D:\_lr580_desktop\research\STD-MAE\checkpoints\STDMAE_300\PEMS03\training_log_20231123120118.log',
    'PEMS04':r'D:\_lr580_desktop\research\STD-MAE\checkpoints\STDMAE_300\PEMS04\training_log_20231214135934.log',
    'PEMS07':r'D:\_lr580_desktop\research\STD-MAE\checkpoints\STDMAE_300\PEMS07\training_log_20240416070716.log',
    'PEMS08':r'D:\_lr580_desktop\research\STD-MAE\checkpoints\STDMAE_300\PEMS08\training_log_20231018202051.log',
}
results = Results()
for dataset in paths:
    path = paths[dataset]
    res = ParserBasicTS.parse_horizon(path, dataset, 'STD-MAE')
    results = Results.merge(results, res)

results.to_csv('results.csv')