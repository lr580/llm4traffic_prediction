# Excel 输出相关通用函数（样式化 DataFrame 导出）
from math import isnan
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, Side
from openpyxl.utils import get_column_letter
from pandas import DataFrame, isna

THIN = Side(style='thin')
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
HEADER_ALIGN = Alignment(horizontal='center', vertical='center')
BEST_FONT = Font(bold=True)
SECOND_FONT = Font(italic=True, underline='single')


def _add_border(cell, left=None, right=None, top=None, bottom=None):
    '''在保留现有边的前提下加指定边（openpyxl 的 Border 需整体赋值）。'''
    b = cell.border
    cell.border = Border(left=left or b.left, right=right or b.right,
                         top=top or b.top, bottom=bottom or b.bottom)


def _disp_width(s: str) -> int:
    '''显示宽度：ASCII 字符记 1，其余（中文等宽字符）记 2。'''
    return sum(1 if ord(ch) < 128 else 2 for ch in str(s))


def _to_num(v):
    '''单元格数值化：可转 float 且非 NaN 则返回 float，否则返回 None。'''
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if isnan(f) else f


def pdDataFrame2excel(df: DataFrame, path, index_header='model', best='min', second=True,
                      number_format='0.000', width_pad=2, sheet_name='baselines',
                      bold_index: set = None) -> Path:
    '''DataFrame 导出为样式化 Excel 表格，返回写入路径。

    - 双表头：MultiIndex 列渲染为两行——首行数据集名横向合并、次行 metric 名，
      左上角 index_header 纵向合并两行；单层列自动退化为单行表头。
    - 最优/次优：best='min'/'max'/None 指定方向，second 控制是否标注次优，
      并列时同值同待遇（最优加粗、次优下划线+斜体），非数值（如 '-'）跳过。
    - bold_index：索引值集合，命中的模型名（第一列）加粗（如 CCF-A 模型），其余保持常规字体。
    - 边框：模型列与各数据集分组之间加竖向分隔线（thin），整个表格外圈加边框（thin）；
      表头加粗居中 + 细边框。
    - 列宽自适应：按表头与单元格显示宽度（ASCII 记 1、中文记 2）取最大值加 padding。
    - 冻结前两行与首列，便于下拉浏览时表头与模型名可见。
    数值单元格写 float 并套用 number_format（如 '0.000' 对应 3 位小数），缺失写空单元格。
    '''
    if df.columns.nlevels > 2:
        raise ValueError(f'仅支持 1~2 层列索引，收到 {df.columns.nlevels} 层')
    multi = df.columns.nlevels == 2
    cols = list(df.columns)
    first_data_row = 3 if multi else 2
    decimals = len(number_format.partition('.')[2])

    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name

    # ---- 表头（先写值后合并，合并区域的非左上角单元格是只读 MergedCell）----
    ws.cell(row=1, column=1, value=index_header)
    widths = {1: _disp_width(index_header)}
    merges = []
    if multi:
        merges.append((1, 1, 2, 1))  # 左上角纵向合并两行
        seg = 0
        for end in range(len(cols) + 1):
            if end == len(cols) or cols[end][0] != cols[seg][0]:
                if end - seg == 1:  # 单 metric 的组：组名纵向占两行
                    merges.append((1, seg + 2, 2, seg + 2))
                else:
                    merges.append((1, seg + 2, 1, end + 1))
                ws.cell(row=1, column=seg + 2, value=cols[seg][0])
                for j in range(seg, end):
                    ws.cell(row=2, column=j + 2, value=cols[j][1])
                    widths[j + 2] = _disp_width(cols[j][1])
                seg = end
    else:
        for j, col in enumerate(cols):
            ws.cell(row=1, column=j + 2, value=str(col))
            widths[j + 2] = _disp_width(str(col))
    for r1, c1, r2, c2 in merges:
        ws.merge_cells(start_row=r1, start_column=c1, end_row=r2, end_column=c2)

    # ---- 数据 ----
    for offset, (idx, row) in enumerate(df.iterrows()):
        r = first_data_row + offset
        name_cell = ws.cell(row=r, column=1, value=str(idx))
        if bold_index and idx in bold_index:
            name_cell.font = Font(bold=True)
        widths[1] = max(widths[1], _disp_width(idx))
        for c, col in enumerate(cols, start=2):
            v = _to_num(row[col])
            if v is None:
                ws.cell(row=r, column=c, value='' if isna(row[col]) else str(row[col]))
            else:
                cell = ws.cell(row=r, column=c, value=v)
                cell.number_format = number_format
                widths[c] = max(widths.get(c, 0), _disp_width(f'{v:.{decimals}f}'))

    # ---- 最优 / 次优标注 ----
    if best is not None:
        for c, col in enumerate(cols, start=2):
            nums = [n for n in (_to_num(v) for v in df[col]) if n is not None]
            if not nums:
                continue
            uniq = sorted(set(nums), reverse=(best == 'max'))
            best_v, second_v = uniq[0], (uniq[1] if len(uniq) > 1 else None)
            for offset, v in enumerate(df[col]):
                n = _to_num(v)
                if n == best_v:
                    ws.cell(row=first_data_row + offset, column=c).font = BEST_FONT
                elif second and second_v is not None and n == second_v:
                    ws.cell(row=first_data_row + offset, column=c).font = SECOND_FONT

    # ---- 表头样式 / 边框 / 冻结 / 列宽 ----
    for r in range(1, first_data_row):
        for c in range(1, len(cols) + 2):
            cell = ws.cell(row=r, column=c)
            cell.font = Font(bold=True)
            cell.alignment = HEADER_ALIGN
            cell.border = BORDER

    last_col = len(cols) + 1
    last_row = first_data_row + len(df) - 1
    if multi:
        # 竖向分隔线：各组首列左侧，贯穿表头到表尾；首组左侧即 model 列与第一个数据集的分隔
        seg = 0
        for end in range(len(cols) + 1):
            if end == len(cols) or cols[end][0] != cols[seg][0]:
                for r in range(1, last_row + 1):
                    _add_border(ws.cell(row=r, column=seg + 2), left=THIN)
                seg = end
    # 整个表格外圈边框
    for c in range(1, last_col + 1):
        _add_border(ws.cell(row=1, column=c), top=THIN)
        _add_border(ws.cell(row=last_row, column=c), bottom=THIN)
    for r in range(1, last_row + 1):
        _add_border(ws.cell(row=r, column=1), left=THIN)
        _add_border(ws.cell(row=r, column=last_col), right=THIN)

    ws.freeze_panes = f'B{first_data_row}'
    for c, w in widths.items():
        ws.column_dimensions[get_column_letter(c)].width = max(w + width_pad, 6)

    dest = Path(path)
    wb.save(dest)
    return dest
