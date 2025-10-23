import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.baselines import Results, CitationHandler
from utils.common import pdDataFrame2str, round_add_decorator, bold_column_min_in_latex_table, wrap_cells_with_added
results = Results.get_baseline_results()
cases = 'LargeST'
if cases == 'PEMS0x':
    results.flit([f'PEMS0{x}' for x in '3478'], horizons=[-1], tags='survey')
    extCite = {'ARIMA': 'O-64', 'VAR': 'O-65', 'SVR': 'O-66'}
    CitationHandler.render(results, externalCite=extCite)
    results.printSOTA()
    df = results.dataset_view().sort_values(by=('PEMS03', 'mae'), ascending=False)
    print(df)
    print(pdDataFrame2str(df, prefix = ''))
    # print(pdDataFrame2str(df, prefix = '', cell_decorator=round_add_decorator))
elif cases == 'LargeST':
    results.flit(tags='2019', horizons=[-1])
    results.flit(models=['DSTAGNN', 'STGODE', 'STWave', 'AGCRN', 'DGCRN', 'D2STGNN', 'GWNet', 'ASTGCN', 'DCRNN', 'STGCN'])
    CitationHandler.render(results)
    df = results.dataset_view(['SD', 'GBA', 'GLA', 'CA']).sort_values(by=('SD', 'mae'), ascending=False)
    print(df)
    # print(pdDataFrame2str(df, prefix = '', cell_decorator=round_add_decorator))
    tex = pdDataFrame2str(df, prefix = '')
    bold_tex = bold_column_min_in_latex_table(tex)
    bold_tex = bold_tex.replace('nan', '-')
    # tex = wrap_cells_with_added(bold_tex)
    print(bold_tex)