import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

from utils.baselines import parseLine_horizons, parseLine_datasets, Results
paper = 'RAST'
match paper:
    case 'RAST':
        # A Retrieval Augmented Spatio-Temporal Framework for Traffic Prediction
        # Table 1 (PEMS03/04/07/08)
        line_PEMS = '15.20 25.81 16.12 18.39 29.93 12.43 19.52 32.73 8.23 14.16 23.33 9.27'
        result_PEMS = parseLine_datasets(line_PEMS, paper)

        # Table 2 (LargeST, 最终版 PDF；数值顺序 MAE RMSE MAPE)
        line_SD = '15.84 26.41 10.15 18.55 31.56 12.13 22.18 39.43 15.38 18.39 31.96 12.19'
        result_SD = parseLine_horizons(line_SD, paper, 'SD', tags='survey,2019')

        line_GBA = '17.71 29.29 13.72 20.86 34.26 16.63 24.97 41.33 21.25 20.64 34.47 16.63'
        result_GBA = parseLine_horizons(line_GBA, paper, 'GBA', tags='survey,2019')
        result = Results.merge(result_PEMS, result_SD, result_GBA)
        result.to_csv('RAST.csv')
