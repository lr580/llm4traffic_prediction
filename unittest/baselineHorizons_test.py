import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.baselines import Results, CitationHandler
from utils.common import pdDataFrame2str, latex_add_decorator
BASELINE_RESULTS = Results.from_csv('utils/baselines/baselineResults.csv')
BASELINE_RESULTS.flit(['PEMS04'], horizons=[3,6,12])
# CitationHandler.parse_rawinfo(r'D:\_lr580_desktop\trafficFlowPredictionSurvey\具体论文笔记.md', r'utils\baselines\citations.json') # INIT
CitationHandler.render(BASELINE_RESULTS, defaultCite='T-138')
df = BASELINE_RESULTS.horizon_view().sort_values(by=(3, 'mae'), ascending=False)
print(df)
print(pdDataFrame2str(df, suffix=r' \\ \cline{2-11}')) #, cell_decorator=latex_add_decorator))
print(df.shape) 
# BASELINE_RESULTS.flit(['PEMS04'], horizons=[3,6,12,-1])
# print(BASELINE_RESULTS.horizon_view([3,6,12,-1]).sort_values(by=(3, 'mae'), ascending=False).dropna())

# some trival EDA
import pandas as pd
growth_df = pd.DataFrame(index=df.index)
for matrix in ['mae', 'rmse', 'mape']:
    for nxt, prv in zip([12], [3]):
    # for nxt, prv in zip([6, 12, 12], [3, 3, 6]):
        base = df[(prv, matrix)]
        new = df[(nxt, matrix)]
        growth = (new - base) / base * 100
        col_name = f"{prv}→{nxt} {matrix}"
        growth_df[col_name] = growth.round(2)
growth_df.sort_values(by='3→12 mae', inplace=True)
print(growth_df)