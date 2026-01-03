import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

from utils.baselines import parseLine_horizons, parseLine_datasets, Results
paper = 'RAST'
match paper:
    case 'RAST':
        # A Retrieval Augmented Spatio-Temporal Framework for Traffic Prediction
        # Table 1
        line_PEMS = '15.36 25.81 16.47 18.39 29.93 12.43 19.52 32.73 8.23 14.20 23.49 9.29'
        result_PEMS = parseLine_datasets(line_PEMS, paper)

        # Table 3
        line_SD = '16.23 26.75 10.23 19.12 32.05 12.45 23.20 40.72 16.06 19.00 32.64 12.53'
        result_SD = parseLine_horizons(line_SD, paper, 'SD', tags='survey,2019')

        line_GBA = '18.57 30.12 14.60 22.12 35.54 18.06 26.31 43.14 21.75 21.75 35.82 17.90'
        result_GBA = parseLine_horizons(line_GBA, paper, 'GBA', tags='survey,2019')
        result = Results.merge(result_PEMS, result_SD, result_GBA)
        result.to_csv('RAST.csv')
