import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.baselines import ParserBasicTS, Results, ParserD2STGNN, ParserLargeST, ParserPatchSTG, RawLargeST, RawPatchSTG, RawRAGL
cases = 'PatchSTG_RAGL'
if cases == 'STD-MAE':
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
elif cases == 'LargeST':
    results = []
    for dataset, raw in RawLargeST.TABLE2.items():
        results.append( ParserLargeST.parse(raw, dataset) )
    results = Results.merge(*results)
    results.to_csv('results_largest.csv')
elif cases == 'PatchSTG_RAGL':
    results = []
    for dataset, raw in RawPatchSTG.TABLE3.items():
        results.append( ParserPatchSTG.parse(raw, dataset, tags='survey,2019') )
    # results = Results.merge(*results)
    # results.to_csv('results_patchtst.csv')

    # results = []
    for dataset, raw in RawRAGL.TABLE3.items():
        results.append( ParserPatchSTG.parse(raw, dataset, tags='survey,2019') )
    results = Results.merge(*results)
    results.to_csv('results_largest.csv')
elif cases == 'D2STGNN': # 早期史山，已经完成使命懒得改了，规范性不如其他的类
    dataset_names = ['PEMS08']
    spj = 'STSGCN 15.45 24.39 10.22% 16.93 26.53 10.84% 19.50 30.43 12.27%'
    for raw_horizon in (spj, ):
        dataset_name = dataset_names[0]
        csv_output = ParserD2STGNN.parse_D2STGNN_horizon(raw_horizon, dataset_name)
        output_filename = f'd:\\_lr580_desktop\\research\\llm_tfp\\utils\\baselines\\horizons_D2STGNN_spj_{dataset_name}.csv'
        with open(output_filename, 'w', newline='', encoding='utf-8') as f:
            f.write(csv_output)
        print(f"结果已保存到 {dataset_name}_horizon_results.csv")