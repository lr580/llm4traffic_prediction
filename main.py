'''项目入口：查看基准模型结果并导出 LaTeX 对比表（从仓库根目录运行）。

    python main.py                                  # 默认 LargeST，导出 LargeST_view.xlsx
    python main.py --format df                      # 覆盖场景配置，打印 DataFrame
    python main.py --format latex                   # 输出 LaTeX 表格行
    python main.py --format excel --output table.xlsx
    python main.py --case PEMS0x --name-style plain  # 不渲染模型名
    python main.py --ccf-a --year 2025 2026        # 只保留 CCF-A 且 2025~2026 年发表的模型
    python main.py --name-style paper --format latex  # 模型名渲染为 Model (VENUE YEAR) + 引用键
'''
from __future__ import annotations

import argparse
from pathlib import Path

from utils.baselines import Results, CitationHandler, Paper
from utils.common import pdDataFrame2excel

ROOT = Path(__file__).resolve().parent
BASELINES_DIR = ROOT / 'utils' / 'baselines'

CASES = {
    'LargeST': dict(
        filter=dict(tags='2019', horizons=[-1], inLen=12, outLen=12),
        datasets=['SD', 'GBA', 'GLA', 'CA'],
        sort_by=('SD', 'MAE'),
        bold_min=True,
        name_style='paper',
        format='excel',
    ),
    'PEMS0x': dict(
        filter=dict(datasets=[f'PEMS0{x}' for x in '3478'], horizons=[-1], tags='survey'),
        datasets=[f'PEMS0{x}' for x in '3478'],
        sort_by=('PEMS03', 'MAE'),
        bold_min=False,
        name_style='cite',
        externalCite={'ARIMA': 'O-64', 'VAR': 'O-65', 'SVR': 'O-66'},
        printSOTA=True,
    ),
    'v3': dict( # v3 是我个人其他模型的实验数据，不公开
        filter=dict(tags='2019', horizons=[-1], inLen=12, outLen=12,
                    models=['HL', 'LSTM', 'ASTGCN', 'DSTAGNN', 'DCRNN', 'STGCN',
                            'STGODE', 'RAST', 'BigST', 'STTN', 'STWave', 'AGCRN', 'DGCRN']),
        merge_csv=BASELINES_DIR / 'v3.csv',
        datasets=['SD', 'GBA', 'GLA', 'CA'],
        sort_by=('SD', 'MAE'),
        bold_min=True,
        name_style='plain',
    ),
}


def run_baseline_view(case: str, fmt: str = None, output: str = '-',
                      ccf_a: bool = False, year_range: tuple = None, name_style: str = None) -> str:
    cfg = CASES[case]
    fmt = fmt or cfg.get('format', 'df')  # 场景 dict 可配默认输出格式，CLI --format 仍可覆盖
    results = Results.get_baseline_results()
    extCite = cfg.get('externalCite', {})
    # {model: Paper}；externalCite 让临时补键模型（如 PEMS0x 的 ARIMA）也能查到元数据
    paper_meta = {r.model: CitationHandler.paper(r.model, extCite) for r in results}

    filter_cfg = dict(cfg['filter'])
    if ccf_a:
        filter_cfg['ccf'] = ['A']
    if year_range is not None:
        filter_cfg['year_range'] = year_range
    if ccf_a or year_range is not None:
        filter_cfg['paper_meta'] = paper_meta
    results.flit(**filter_cfg)
    if cfg.get('merge_csv'):
        results = Results.merge(results, Results.from_csv(str(cfg['merge_csv'])))
    style = name_style or cfg.get('name_style', 'plain')
    papers = [paper_meta.get(r.model, Paper(r.model)) for r in results]  # 渲染前快照，与 results 顺序对应
    if style == 'cite':                     # Model \cite{T-xxx}（render 默认 formatter）
        CitationHandler.render(results, externalCite=extCite)
    elif style == 'paper':                  # Model (VENUE YEAR) \cite{T-xxx}；venue/year 缺失自动降级
        CitationHandler.render(results, formatter=Paper.display, externalCite=extCite)
    if cfg.get('printSOTA'):
        results.printSOTA()
    if fmt == 'latex':
        df = None
        text = results.to_latex_table(datasets=cfg['datasets'], sort_by=cfg['sort_by'],
                                      bold_min=cfg.get('bold_min', True))
    else:
        df = results.dataset_view(cfg['datasets']).sort_values(by=cfg['sort_by'], ascending=False)
        text = str(df)
    if fmt == 'excel':
        # 渲染后 model 即显示名；CCF-A 的显示名集合，供表格首列加粗标注（无元数据者自然保持常规字体）
        ccf_a = {r.model for r, p in zip(results, papers) if p.ccf == 'A'}
        dest = Path(f'{case}_view.xlsx' if output == '-' else output).with_suffix('.xlsx')
        pdDataFrame2excel(df, dest, bold_index=ccf_a)
        print(f'已写入 {dest}')
        return text
    if output == '-':
        print(text)
    else:
        Path(output).write_text(text, encoding='utf-8')
        print(f'已写入 {output}')
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description='查看基准模型结果并导出 LaTeX 对比表')
    parser.add_argument('--case', choices=CASES, default='LargeST', help='结果场景，默认 LargeST')
    parser.add_argument('--format', choices=['df', 'latex', 'excel'], default=None,
                        help='输出内容，默认用场景配置（未配置则 df，打印 DataFrame）；'
                             'latex 输出 LaTeX 表格行；excel 导出样式化表格（双表头合并、列宽自适应、'
                             '每列最优加粗/次优下划线斜体），默认写 {场景}_view.xlsx，可用 --output 指定路径')
    parser.add_argument('--output', default='-', help='输出文件；- 表示标准输出')
    parser.add_argument('--ccf-a', action='store_true',
                        help='只保留发表于 CCF-A 的论文对应的模型（无引用键模型默认剔除）')
    parser.add_argument('--year', nargs=2, type=int, metavar=('START', 'END'), default=None,
                        help='只保留论文发表年份在区间内的模型，如 --year 2025 2026')
    parser.add_argument('--name-style', dest='name_style', choices=['plain', 'cite', 'paper'], default=None,
                        help=r'模型名渲染：plain 原样 / cite 注入 \cite / paper 加 (VENUE YEAR) 标注；'
                             '默认按场景配置')
    args = parser.parse_args(argv)
    run_baseline_view(args.case, args.format, args.output,
                      ccf_a=args.ccf_a, year_range=tuple(args.year) if args.year else None,
                      name_style=args.name_style)


if __name__ == '__main__':
    main()
