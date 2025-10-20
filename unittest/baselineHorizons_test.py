import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.baselines import Results, CitationHandler
from utils.common import pdDataFrame2str
BASELINE_RESULTS = Results.from_csv('utils/baselines/baselineResults.csv')
BASELINE_RESULTS.flit(['PEMS04'], horizons=[3,6,12])
CitationHandler.render(BASELINE_RESULTS)
df = BASELINE_RESULTS.horizon_view().sort_values(by=(3, 'mae'), ascending=False)
print(pdDataFrame2str(df))
print(df)
# BASELINE_RESULTS.flit(['PEMS04'], horizons=[3,6,12,-1])
# print(BASELINE_RESULTS.horizon_view([3,6,12,-1]).sort_values(by=(3, 'mae'), ascending=False).dropna())
# CitationHandler.parse_rawinfo(r'D:\_lr580_desktop\trafficFlowPredictionSurvey\具体论文笔记.md') # INIT