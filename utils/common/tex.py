import re
from .log import round_decorator

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