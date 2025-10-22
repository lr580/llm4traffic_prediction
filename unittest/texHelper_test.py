import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.common import wrap_cells_with_added, unwrap_added
s = r"""
\cite{T-371} & CA & California, USA & $2017/01/01\sim2021/12/31$ & 525888 & 5 minues & 8600 & Graph & Vehicle \\ \hline
    \cite{T-371} & GLA & California, USA & $2017/01/01\sim2021/12/31$ & 525888 & 5 minues & 3834 & Graph & Vehicle \\ \hline
    \cite{T-371} & GBA & California, USA & $2017/01/01\sim2021/12/31$ & 525888 & 5 minues & 2352 & Graph & Vehicle \\ \hline
    \cite{T-371} & SD & California, USA & $2017/01/01\sim2021/12/31$ & 525888 & 5 minues & 716 & Graph & Vehicle \\ \hline
""".strip("\n")

wrapped = wrap_cells_with_added(s)
restored = unwrap_added(wrapped)
print(wrapped)
print(restored)
assert s == restored