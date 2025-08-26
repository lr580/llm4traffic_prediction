import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.baselines import Results, Result, EvaluateResult
BASELINE_RESULTS = Results.from_csv('utils/baselines/baselineResults.csv')
# print(BASELINE_RESULTS)
BASELINE_RESULTS.printSOTA()
basicTSresults = BASELINE_RESULTS.flit(tags='BasicTS')
print(basicTSresults.to_dataframe())
# print(basicTSresults.flit(['PEMS03']).to_dataframe())
print(basicTSresults.flit(['PEMS03']).sort())

# test merge
result = Result('iTrans+STGCN', 'PEMS03', EvaluateResult(14.75, 14.67, 25.34), tags='self')
results = Results.merge(basicTSresults.flit(['PEMS03']), Results([result]))
print(results.sort())
# results.to_csv('test.csv')
results.sort().to_csv('test.csv', index=False)