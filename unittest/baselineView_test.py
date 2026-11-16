import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.baselines import Results, CitationHandler, Paper
from utils.common import pdDataFrame2str, round_add_decorator, bold_column_min_in_latex_table, wrap_cells_with_added
results = Results.get_baseline_results()
cases = 'paperMeta'
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
    results.flit(tags='2019', horizons=[-1], inLen=12, outLen=12)
    # results.flit(models=['DSTAGNN', 'STGODE', 'STWave', 'AGCRN', 'DGCRN', 'D2STGNN', 'GWNet', 'ASTGCN', 'DCRNN', 'STGCN'])
    CitationHandler.render(results)
    df = results.dataset_view(['SD', 'GBA', 'GLA', 'CA']).sort_values(by=('SD', 'mae'), ascending=False)
    print(df)
    # print(pdDataFrame2str(df, prefix = '', cell_decorator=round_add_decorator))
    tex = pdDataFrame2str(df, prefix = '')
    bold_tex = bold_column_min_in_latex_table(tex)
    bold_tex = bold_tex.replace('nan', '-')
    # tex = wrap_cells_with_added(bold_tex)
    # print(bold_tex)
elif cases == 'paperMeta':
    # 论文元数据演示：按 CCF-A/年份过滤后，用 Paper 渲染显示名（inner=False 深拷贝，不串改原集合）
    paper_meta = {r.model: CitationHandler.paper(r.model) for r in results}
    res = Results.get_baseline_results().flit(ccf=['A'], year_range=(2025, 2026), paper_meta=paper_meta, inner=False)
    res.flit(tags='2019', horizons=[-1])
    CitationHandler.render(res, formatter=Paper.display)
    df = res.dataset_view(['SD', 'GBA', 'GLA', 'CA'])
    print(pdDataFrame2str(df, prefix=''))
    print(bold_column_min_in_latex_table(pdDataFrame2str(df, prefix='')))
elif cases == 'v3':
    results.flit(tags='2019', horizons=[-1], inLen=12, outLen=12)
    results.flit(models=['HL', 'LSTM', 'ASTGCN', 'DSTAGNN', 'DCRNN', 'STGCN', 'STGODE', 'RAST', 'BigST', 'STTN', 'STWave', 'AGCRN', 'DGCRN'])
    my_result = Results.from_csv("utils/baselines/v3.csv")
    results = Results.merge(results, my_result)
    df = results.dataset_view(['SD', 'GBA', 'GLA', 'CA']).sort_values(by=('SD', 'mae'), ascending=False)
    print(df)
