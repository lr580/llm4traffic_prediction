'''论文元数据功能的回归验证（脚本式，不引入 pytest）。

运行：python unittest/paperMeta_test.py
依赖 papers.json 快照与代码内的 CCF_A_CATALOG 常量（离线可跑）。
'''
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
import json
from utils.baselines import Results, CitationHandler, Paper
from utils.baselines.ccfCatalog import CCF_A_VENUES
from utils.common import pdDataFrame2str, bold_column_min_in_latex_table

P = 'utils/baselines/'
papers = json.load(open(P + 'papers.json', encoding='utf-8'))
cites = json.load(open(P + 'citations.json', encoding='utf-8'))


def check(cond, msg):
    assert cond, 'FAIL: ' + msg
    print('  OK  ' + msg)


print('[1] 目录口径与数据契约')
check('IJCAI' in CCF_A_VENUES, 'IJCAI 按项目口径为 A')
check(all(set(v) <= {'venue', 'year', 'ccf'} for k, v in papers.items() if not k.startswith('_')),
      'papers.json 字段为三字段子集（空值省略）')
check(all('unknown' not in v.values() and all(x != '' and x is not None for x in v.values())
          for v in papers.values() if isinstance(v, dict)),
      'papers.json 不写 unknown / 空串 / null，空值字段不存')
check(cites['STID'] == 'T-376' and 'T-376' in papers, 'STID 引用键已修正为 T-376')
check(all(k in papers for k in cites.values()),
      f'citations.json 的 {len(cites)} 个引用键全部有元数据')

print('[2] 黄金条目（Zotero 同步结果）')
gold = {
    'T-28':    {'venue': 'IJCAI', 'year': 2018, 'ccf': 'A'},
    'T-40':    {'venue': 'ICLR',  'year': 2018, 'ccf': 'A'},
    'T-379':   {'venue': 'KDD',   'year': 2025, 'ccf': 'A'},
    'T-414':   {'venue': 'AAAI',  'year': 2026, 'ccf': 'A'},
    'T-456':   {'venue': 'IJCAI', 'year': 2026, 'ccf': 'A'},
    'T-LLM-1': {'venue': 'ICLR',  'year': 2024, 'ccf': 'A'},
    'T-467':   {'year': 2026},  # arXiv 不在 A 类目录，venue/ccf 不写入
    'O-33':    {'venue': 'NeurIPS', 'year': 2014, 'ccf': 'A'},
}
for k, want in gold.items():
    check(papers.get(k) == want, f'{k} -> {want}')

print('[3] 查表与 formatter')
check(CitationHandler.meta(CitationHandler.cite_of_model('STGCN')) == {'venue': 'IJCAI', 'year': 2018, 'ccf': 'A'},
      'STGCN 经 cite_of_model -> meta 查到 IJCAI 2018')
check(CitationHandler.meta('T-467') == {'venue': '', 'year': 2026, 'ccf': ''},
      '快照省略的字段由 meta 补空串')
built = CitationHandler.zotero.build_papers([
    {'data': {'itemType': 'preprint', 'citationKey': 'T-空', 'repository': 'arXiv.org'}},
    {'data': {'itemType': 'conferencePaper', 'citationKey': 'T-满',
              'conferenceName': 'KDD', 'date': '2025'}},
])
check(built['T-空'] == {} and built['T-满'] == {'venue': 'KDD', 'year': 2025, 'ccf': 'A'},
      'build_papers 构建快照时只写非空字段')
check(CitationHandler.meta('T-不存在的键') == {'venue': '', 'year': '', 'ccf': ''},
      '缺键返回空值三元组')
saved = CitationHandler.paper_data
CitationHandler.paper_data = {'K': {'venue': 'KDD'},
                              'N': {'venue': None, 'year': None, 'ccf': None}}
check(CitationHandler.meta('K') == {'venue': 'KDD', 'year': '', 'ccf': ''},
      '不存字段自然就是未知，缺的字段补空串')
check(CitationHandler.meta('N') == {'venue': '', 'year': '', 'ccf': ''},
      'JSON 写 null 同样归一为空串')
CitationHandler.paper_data = saved
st = CitationHandler.paper('STGCN')
check((st.model, st.cite, st.venue, st.year, st.ccf) == ('STGCN', 'T-28', 'IJCAI', 2018, 'A'),
      'paper 一次查齐引用键与元数据，year 为 int')
check(CitationHandler.paper('HA').unknown()
      and CitationHandler.paper('不存在的模型').year is None,
      '无元数据 / 缺键时字段留空且 unknown() 成立')
check(CitationHandler.paper('X', {'X': ''}).display() == 'X'
      and Paper('HA').display() == 'HA',
      'venue/year 全缺时显示名原样返回')
check(Paper('X', cite='', venue='KDD').display() == 'X (KDD)'
      and Paper('X', cite='', year=2026).display() == 'X (2026)',
      'venue / year 有其一即标注（如 year-only 的非 A 论文）')
check(CitationHandler.paper('ARIMA', {'ARIMA': 'O-64'}).year is not None,
      'externalCite 临时补键可查到元数据')
check(CitationHandler.paper('STGCN').display(withCite=False) == 'STGCN (IJCAI 2018)'
      and CitationHandler.paper('HA').display() == 'HA',
      'display 只对已知 venue/year 加标注')
check(CitationHandler.paper('STGCN').display() == 'STGCN (IJCAI 2018) \\cite{T-28}',
      'display 默认同时加标注和引用')

print('[4] flit 元数据筛选与深拷贝')
all_res = Results.get_baseline_results()
paper_meta = {r.model: CitationHandler.paper(r.model) for r in all_res}
check(len(all_res) > 0, f'结果集非空（baselineResults.csv 当前 {len(all_res)} 行，不硬编码行数）')
plain = all_res.flit(inner=False)
check(0 < len(plain) <= len(all_res), '不带元数据条件时按 flit 默认实验条件筛')
res_a = all_res.flit(ccf=['A'], paper_meta=paper_meta, inner=False)
check(len(res_a) > 0 and all(CitationHandler.paper(r.model).ccf == 'A' for r in res_a),
      f'ccf=[A] 筛出 {len(res_a)} 行且全部为 A')
res_y = all_res.flit(year_range=(2025, 2026), paper_meta=paper_meta, inner=False)
check(all(2025 <= CitationHandler.paper(r.model).year <= 2026 for r in res_y),
      f'year_range=(2025,2026) 筛出 {len(res_y)} 行且年份全在区间内')
res_v = all_res.flit(venues=['KDD'], paper_meta=paper_meta, inner=False)
check(all(CitationHandler.paper(r.model).venue == 'KDD' for r in res_v),
      f'venues=[KDD] 筛出 {len(res_v)} 行且 venue 全匹配')
check(not any(r.model == 'HA' for r in res_a), '默认排除 13 个无引用键模型')
res_u = all_res.flit(ccf=['A'], include_unknown=True, paper_meta=paper_meta, inner=False)
check(any(r.model == 'HA' for r in res_u), 'include_unknown 保留 HA 等三字段皆空的对照基线')
base = Results.get_baseline_results()
sub = base.flit(ccf=['A'], paper_meta=paper_meta, inner=False)
CitationHandler.render(sub, formatter=Paper.display)
check(all('\\cite{' not in r.model for r in base), 'flit(inner=False) 深拷贝，render 不串改原 Results')

print('[5] Zotero 子笔记录用事实兜底')
from utils.baselines.ccfCatalog import match_venue

def note_item(text, parent='P'):
    return {'data': {'itemType': 'note', 'parentItem': parent, 'note': f'<p>{text}</p>'}}

note_samples = (   # 覆盖 _notes() 支持的各种真实写法（docstring 只举 2 例，完整以本清单为准）
    ('Comment: Accept by ICML 2026', 'ICML 2026'),
    ('Accepted to NeurIPS 2026 (main conference)', 'NeurIPS 2026 (main conference)'),
    ('Accepted by NeurIPS 2025 (Spotlight)', 'NeurIPS 2025 (Spotlight)'),
    ('15 pages, 9 figures, accepted by ICDE 2026', 'ICDE 2026'),
    ("to be published in the 42nd IEEE International Conference on Data Engineering (ICDE '26)",
     "the 42nd IEEE International Conference on Data Engineering (ICDE '26)"),
    ('To appear at ICLR 2018. 12 pages.', 'ICLR 2018'),
    ('Published as a conference paper at ICLR 2017', 'ICLR 2017'),
)
for text, expected in note_samples:
    got = CitationHandler.zotero._notes([note_item(text)]).get('P')
    check(got == expected, f'note 提取 {text[:32]!r} -> {expected!r}')
check(not CitationHandler.zotero._notes([note_item('11 pages, 3 figures')])
      and not CitationHandler.zotero._notes([note_item('Under review in IEEE TITS')]),
      '页数 / under review 等无录用语义的笔记不提取')
workshop = CitationHandler.zotero._notes(
    [note_item('Accepted at the ICML 2019 Workshop "Learning and Reasoning"')]).get('P')
check(workshop and match_venue(workshop) is None, 'workshop 可提取但被 match_venue 反例否决')

items = [
    {'data': {'key': 'AAAA', 'itemType': 'conferencePaper', 'archiveLocation': 'T-460',
              'date': '2025-07-13',
              'proceedingsTitle': 'Proceedings of the 42nd International Conference on Machine Learning'}},
    {'data': {'key': 'BBBB', 'itemType': 'preprint', 'archiveLocation': 'T-59',
              'date': '2021-03-29', 'repository': 'arXiv'}},
    {'data': {'key': 'CCCC', 'itemType': 'preprint', 'archiveLocation': 'T-455',
              'date': '2026-05-19', 'repository': 'arXiv'}},
    note_item(note_samples[0][0], 'CCCC'),
]
built = CitationHandler.zotero.build_papers(items)
check(built['T-460'] == {'venue': 'ICML', 'year': 2025, 'ccf': 'A'}, '正式 venue 字段命中时优先')
check(built['T-59'] == {'year': 2021}, 'arXiv 且无录用笔记时不写 venue/ccf')
check(built['T-455'] == {'venue': 'ICML', 'year': 2026, 'ccf': 'A'}, '子笔记 Accept by 兜底 venue/ccf')

print('[6] 视图输出链路')
res = Results.get_baseline_results().flit(ccf=['A'], year_range=(2025, 2026), paper_meta=paper_meta, inner=False)
res.flit(tags='2019', horizons=[-1])
CitationHandler.render(res, formatter=Paper.display)
df = res.dataset_view(['SD', 'GBA', 'GLA', 'CA'])
tex = pdDataFrame2str(df, prefix='')
check(any('(' in str(i) for i in df.index), '显示名落在 df.index，不新增文本列')
check('\\cite{' in tex, '引用注入进入 LaTeX 输出')
check(len(bold_column_min_in_latex_table(tex)) > 0, 'bold_column_min_in_latex_table 仍可处理')

print('全部通过')
