import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.baselines import Results, CitationHandler
from utils.common import pdDataFrame2str, latex_add_decorator
BASELINE_RESULTS = Results.from_csv('utils/baselines/baselineResults.csv')
BASELINE_RESULTS.flit(['PEMS08'], horizons=[3,6,12])
# CitationHandler.parse_rawinfo(r'D:\_lr580_desktop\trafficFlowPredictionSurvey\具体论文笔记.md', r'utils\baselines\citations.json') # INIT
CitationHandler.render(BASELINE_RESULTS, defaultCite='T-138')
df = BASELINE_RESULTS.horizon_view().sort_values(by=(3, 'mae'), ascending=False)
print(pdDataFrame2str(df, suffix=r' \\ \cline{2-11}')) #, cell_decorator=latex_add_decorator))
print(df.shape) 
# BASELINE_RESULTS.flit(['PEMS04'], horizons=[3,6,12,-1])
# print(BASELINE_RESULTS.horizon_view([3,6,12,-1]).sort_values(by=(3, 'mae'), ascending=False).dropna())