import re
from .log import round_decorator
from typing import List, Optional, Dict

def latex_add_decorator(s: str):
    return f"\\added{{{s}}}"

# 懒得用装饰器/模式了，就这样吧
def round_add_decorator(s: str):
    rounded = round_decorator(s)
    return latex_add_decorator(rounded)

import re
from typing import List

def wrap_cells_with_added(table_text: str) -> str:
    """
    将 LaTeX 表格的每个单元格（以 & 分隔）包裹为 \added{...}。
    保留每行原有的缩进、以及行尾的 '\\' 与可选的 '\hline'。
    """
    lines: List[str] = table_text.splitlines()
    wrapped_lines: List[str] = []

    tail_pat = re.compile(r'(\\\\(?:\s*\\hline)?)\s*$')  # 捕获 \\ 和可选 \hline

    for line in lines:
        if line.strip() == "":
            wrapped_lines.append(line)
            continue

        # 保留行首缩进
        leading_ws_match = re.match(r'\s*', line)
        leading_ws = leading_ws_match.group(0) if leading_ws_match else ""
        content = line[len(leading_ws):].rstrip()

        # 提取行尾标记（\\ 及可选 \hline）
        tail = ""
        m = tail_pat.search(content)
        if m:
            tail = m.group(1)
            main = content[:m.start()].rstrip()
        else:
            main = content

        if main == "":
            # 空主内容，直接还原
            wrapped_lines.append(leading_ws + (tail if tail else ""))
            continue

        # 以 & 分列（忽略两侧空格），然后用标准 ' & ' 连接
        cells = re.split(r'\s*&\s*', main)

        # 包裹每个单元格；若已是 \added{...} 也继续包裹，反函数会去重
        wrapped_cells = [f"\\added{{{c}}}" for c in cells]

        new_main = " & ".join(wrapped_cells)
        new_line = leading_ws + new_main + (" " + tail if tail else "")
        wrapped_lines.append(new_line)

    return "\n".join(wrapped_lines)

def unwrap_added(text: str) -> str:
    """
    去除文本中所有 \added{...} 包裹（递归、正确处理花括号嵌套）。
    """
    i = 0
    n = len(text)
    out_chars: List[str] = []

    added_prefix = "\\added{"
    plen = len(added_prefix)

    while i < n:
        # 命中 \added{
        if text.startswith(added_prefix, i):
            i += plen  # 跳过 \added{
            depth = 1
            inner_start = i

            # 寻找与 \added{ 匹配的右花括号
            while i < n and depth > 0:
                ch = text[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                i += 1

            # depth==0 正常闭合；截取内部内容（不含最外层的花括号）
            inner = text[inner_start:i-1] if depth == 0 else text[inner_start:]
            # 递归去除内部可能嵌套的 \added{...}
            out_chars.append(unwrap_added(inner))
        else:
            out_chars.append(text[i])
            i += 1

    return "".join(out_chars)

_NUM_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
_NAN_RE = re.compile(r'\bnan\b', re.IGNORECASE)

def _extract_first_number_text(cell: str) -> Optional[str]:
    m = _NUM_RE.search(cell)
    return m.group(0) if m else None

def _parse_cell_value(cell: str) -> Optional[float]:
    if _NAN_RE.search(cell):
        return None
    num_text = _extract_first_number_text(cell)
    if num_text is None:
        return None
    try:
        return float(num_text)
    except ValueError:
        return None

def _bold_first_number(cell: str) -> str:
    return _NUM_RE.sub(lambda m: r'\textbf{' + m.group(0) + '}', cell, count=1)

def bold_column_max_in_latex_table(
    table_text: str,
    numeric_start_col: int = 1  # 从第2列开始视为数值列
) -> str:
    lines = [ln for ln in (table_text or "").splitlines() if ln.strip()]
    
    # 预解析成单元格
    rows: List[List[str]] = []
    for ln in lines:
        # 保留最后单元格中的 '\\ \hline' 等尾部内容
        cells = [c.strip() for c in ln.split('&')]
        rows.append(cells)

    # 统计每列最大值
    col_max: Dict[int, float] = {}
    for cells in rows:
        for idx in range(numeric_start_col, len(cells)):
            val = _parse_cell_value(cells[idx])
            if val is None:
                continue
            if idx not in col_max or val > col_max[idx]:
                col_max[idx] = val

    # 第二遍，加粗等于该列最大值的单元格（忽略 None）
    out_lines: List[str] = []
    for cells in rows:
        new_cells = cells[:]
        for idx in range(numeric_start_col, len(cells)):
            val = _parse_cell_value(cells[idx])
            if val is None:
                continue
            max_val = col_max.get(idx, None)
            if max_val is None:
                continue
            # 浮点比较：这里数值来自原始字符串解析，直接比较足够；如需更稳健可加容差
            if val == max_val:
                new_cells[idx] = _bold_first_number(new_cells[idx])
        out_lines.append(' & '.join(new_cells))
    return '\n'.join(out_lines)

def bold_column_min_in_latex_table(
    table_text: str,
    numeric_start_col: int = 1  # 从第2列开始视为数值列
) -> str:
    ''' 例子见 unittest/baselineView_test.py 的 LargeST case '''
    # 依赖上个实现中的: _parse_cell_value, _bold_first_number
    lines = [ln for ln in (table_text or "").splitlines() if ln.strip()]
    rows = [[c.strip() for c in ln.split('&')] for ln in lines]

    col_min = {}
    for cells in rows:
        for idx in range(numeric_start_col, len(cells)):
            val = _parse_cell_value(cells[idx])
            if val is None:
                continue
            if idx not in col_min or val < col_min[idx]:
                col_min[idx] = val

    out_lines = []
    for cells in rows:
        new_cells = cells[:]
        for idx in range(numeric_start_col, len(cells)):
            val = _parse_cell_value(cells[idx])
            if val is None:
                continue
            min_val = col_min.get(idx, None)
            if min_val is None:
                continue
            if val == min_val:
                new_cells[idx] = _bold_first_number(new_cells[idx])
        out_lines.append(' & '.join(new_cells))
    return '\n'.join(out_lines)