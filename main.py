'''项目入口：查看基准模型结果并导出 LaTeX 对比表（从仓库根目录运行）。

    python main.py                                  # 默认 LargeST，打印 DataFrame
    python main.py --format latex                   # 输出 LaTeX 表格行
    python main.py --format latex --output table.tex
    python main.py --output table.xlsx             # 写出完整 Excel 表格
    python main.py --case PEMS0x --no-cite
'''
from __future__ import annotations

import argparse
from pathlib import Path

from utils.baselines import Results, CitationHandler

ROOT = Path(__file__).resolve().parent
BASELINES_DIR = ROOT / 'utils' / 'baselines'

CASES = {
    'LargeST': dict(
        filter=dict(tags='2019', horizons=[-1], inLen=12, outLen=12),
        datasets=['SD', 'GBA', 'GLA', 'CA'],
        sort_by=('SD', 'mae'),
        bold_min=True,
        cite=True,
    ),
    'PEMS0x': dict(
        filter=dict(datasets=[f'PEMS0{x}' for x in '3478'], horizons=[-1], tags='survey'),
        datasets=[f'PEMS0{x}' for x in '3478'],
        sort_by=('PEMS03', 'mae'),
        bold_min=False,
        cite=True,
        externalCite={'ARIMA': 'O-64', 'VAR': 'O-65', 'SVR': 'O-66'},
        printSOTA=True,
    ),
    'v3': dict(
        filter=dict(tags='2019', horizons=[-1], inLen=12, outLen=12,
                    models=['HL', 'LSTM', 'ASTGCN', 'DSTAGNN', 'DCRNN', 'STGCN',
                            'STGODE', 'RAST', 'BigST', 'STTN', 'STWave', 'AGCRN', 'DGCRN']),
        merge_csv=BASELINES_DIR / 'v3.csv',
        datasets=['SD', 'GBA', 'GLA', 'CA'],
        sort_by=('SD', 'mae'),
        bold_min=True,
        cite=False,
    ),
}


def run_baseline_view(case: str, fmt: str = 'df', output: str = '-', cite: bool = None) -> str:
    cfg = CASES[case]
    results = Results.get_baseline_results()
    results.flit(**cfg['filter'])
    if cfg.get('merge_csv'):
        results = Results.merge(results, Results.from_csv(str(cfg['merge_csv'])))
    use_cite = cfg.get('cite', False) if cite is None else cite
    if use_cite:
        CitationHandler.render(results, externalCite=cfg.get('externalCite', {}))
    if cfg.get('printSOTA'):
        results.printSOTA()

    if fmt == 'latex':
        df = None
        text = results.to_latex_table(datasets=cfg['datasets'], sort_by=cfg['sort_by'],
                                      bold_min=cfg.get('bold_min', True))
    else:
        df = results.dataset_view(cfg['datasets']).sort_values(by=cfg['sort_by'], ascending=False)
        text = str(df)
    if output == '-':
        print(text)
    elif fmt == 'latex':
        Path(output).write_text(text, encoding='utf-8')
        print(f'已写入 {output}')
    else:
        dest = Path(output).with_suffix('.xlsx')
        excel_df = df.copy()
        # 两级列索引展平成单层表头，避免 to_excel 多写一行空表头且数据集名只出现一次
        excel_df.columns = [f'{a} {b}' for a, b in excel_df.columns]
        excel_df.to_excel(dest, engine='openpyxl')
        print(f'已写入 {dest}')
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description='查看基准模型结果并导出 LaTeX 对比表')
    parser.add_argument('--case', choices=CASES, default='LargeST', help='结果场景，默认 LargeST')
    parser.add_argument('--format', choices=['df', 'latex'], default='df',
                        help='输出内容，默认 df（打印 DataFrame）；latex 输出 LaTeX 表格行')
    parser.add_argument('--cite', dest='cite', action='store_true', default=None,
                        help='给模型名注入 LaTeX 引用')
    parser.add_argument('--no-cite', dest='cite', action='store_false', default=None,
                        help='不注入引用；不指定时按 case 默认配置')
    parser.add_argument('--output', default='-', help='输出文件；- 表示标准输出')
    args = parser.parse_args(argv)
    run_baseline_view(args.case, args.format, args.output, args.cite)


if __name__ == '__main__':
    main()
