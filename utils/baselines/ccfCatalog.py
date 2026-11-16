'''CCF 目录与 venue 解析：venue 匹配与 CCF 等级判定只认目录条目；arXiv 预印本的
兜底不归本文件，由调用方 citations.py 在候选链之后按 repository 补 venue='arXiv'。

模块组成：
    ccfA.json / ccfB.json           目录数据（同目录，离线可用，不读 PDF）
    CCF_A_CATALOG / CCF_B_CATALOG   缩写 -> 全称，对应项目口径的 A / B 类
    _CATALOG                        匹配用合并目录（键冲突 A 优先）
    CCF_LEVEL                       缩写 -> 'A'/'B'，等级判定
    match_venue()                   只认目录条目，arXiv 不在目录里，传进去返回 None

数据口径（对照 CCF 第七版/2026）：
    A 表：
        - CCF 第七版 A 类会议/期刊
        - 与第七版的差异：IJCAI 第七版降为 B，项目按 A 计
    B 表：
        - 第七版 B 类会议/期刊
        - 同一缩写对应多个全称的（TCC/DCC/AAMAS/RE 同时挂会议和期刊），只存一个，
          其余多对一放 VENUE_ALIASES
        - 不含与 A 撞缩写的 ASE/CSCW/FSE/WWW（A 会议 vs B 期刊，保 A 口径优先）

用法：
    python -m utils.baselines.ccfCatalog   # 统计 Zotero 实际写法命中情况
'''
from __future__ import annotations
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_catalog(filename) -> dict:
    '''读取同目录的目录 JSON：{缩写: 全称}。

    例：_load_catalog('ccfA.json')['ICML'] -> 'InternationalConferenceonMachineLearning'
    '''
    return json.loads((HERE / filename).read_text(encoding='utf-8'))


CCF_A_CATALOG = _load_catalog('ccfA.json')
CCF_B_CATALOG = _load_catalog('ccfB.json')

# CCF_LEVEL：缩写 -> 等级（'A'/'B'），A/B 两表合成一张，citations.py 的 ccf 字段查这里
assert not (set(CCF_A_CATALOG) & set(CCF_B_CATALOG)), '缩写同时出现在 A/B 目录，等级歧义，需人工消歧'
CCF_LEVEL = {**{abbr: 'A' for abbr in CCF_A_CATALOG}, **{abbr: 'B' for abbr in CCF_B_CATALOG}}

# 匹配用合并目录：B 在前、A 在后覆盖，键冲突时 A 优先（B 表已排除撞缩写条目，此序只作保险）。
_CATALOG = {**CCF_B_CATALOG, **CCF_A_CATALOG}

# ---------- Zotero 数据源 ----------
# 本地 API 地址与 venue 字段优先级：citations.py 的 sync_from_zotero() 与
# 下方 scan_zotero_venues() 共用这一份，避免两处各写一份。
ZOTERO_API = 'http://127.0.0.1:23119/api/users/0/items'

# 候选 venue 字段，按优先级取第一个非空作为论文 venue。
# 已知问题：第一个非空字段可能顶掉可匹配的低优先级字段（漏判），例如
# T-S-9 的 series "SIGMOD/PODS '25" 可匹配 SIGMOD，却被 proceedingsTitle
# 'Companion of the 2025 ...' 顶掉；T-ZS-LLM-6 / T-391 同类。
VENUE_FIELDS = ['proceedingsTitle', 'conferenceName', 'publicationTitle', 'repository', 'series']

# 人工别名：全称之外的匹配候选，第二、三步共用。A 类沿用原有；B 类补充
# 同缩写双载体的另一全称（TCC/DCC/AAMAS/RE）与 Zotero 常见异写。
VENUE_ALIASES = {
    'CVPR':   ['IEEE/CVF Conference on Computer Vision and Pattern Recognition',
               'IEEE Conference on Computer Vision and Pattern Recognition'],
    'IJCAI':  ['Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August 10-16, 2019'],
    'KDD':    ['SIGKDD'],
    # A 表第七版补录（2026-11，原项目裁掉的 4 条）：S&P 缩写含 '&' 无法词命中、
    # USENIX ATC 第七版升 A 且更名（ACM SIGOPS ATC），两个全称都放别名兜住。
    'S&P':    ['IEEE S&P'],
    'USENIXATC': ['USENIX ATC', 'ACM SIGOPS Annual Technical Conference'],
    # 注意：这条全称必须带空格写——normalize_venue 会剥掉 'Proceedings of the ' 前缀，
    # 使其与查询串 'Proceedings of the IEEE' 同样归一为 'ieee' 后精确对齐；融合写法剥不掉前缀。
    'Proc. IEEE': ['Proc IEEE'],
    'NeurIPS':['Advances in Neural Information Processing Systems', 'Neural Information Processing Systems'],
    'WWW':    ['ACM Web Conference', 'World Wide Web Conference'],
    'TCC':    ['IEEE Transactions on Cloud Computing'],
    'DCC':    ['Designs, Codes and Cryptography'],
    'AAMAS':  ['Autonomous Agents and Multi-Agent Systems'],
    'RE':     ['IEEE International Conference on Requirements Engineering'],
    'ECML-PKDD': ['Machine Learning and Knowledge Discovery in Databases',
                  'European Conference on Machine Learning and Knowledge Discovery in Databases'],
    'EMNLP':  ['Empirical Methods in Natural Language Processing'],
    'NAACL':  ['Annual Meeting of the North American Chapter of the Association for Computational Linguistics'],
    'UAI':    ['Uncertainty in Artificial Intelligence'],
    'COLT':   ['Annual Conference on Computational Learning Theory'],
    'DASFAA': ['Database Systems for Advanced Applications'],          # Zotero 常缺 'International Conference on' 前缀
    'TKDD':   ['ACM Trans. Knowl. Discov. Data'],                      # ACM 刊名缩写式
    'JWS':    ['Web Semantics: Science, Services and Agents on the World Wide Web'],
    'PLOS Computational Biology': ['PLoS Comput Biol'],
    # 全称被其他 venue 名内嵌的条目：简称改放别名（词命中停用，第三步相似度兜住）。
    'SIGGRAPH': ['ACM SIGGRAPH', 'SIGGRAPH'],
    'GROUP':  ['GROUP'],
    'Eurographics': ['Eurographics'],
    'Performance': ['Performance'],
}


# ---------- 公共预处理 ----------
def normalize_venue(raw) -> str:
    '''venue 原始串 -> 归一化串：去前缀 / 届次 / 年份 / V.1 / 标点，小写。

    例：normalize_venue('Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1')
        -> 'acm sigkdd conference on knowledge discovery and data mining'
    '''
    s = str(raw or '').strip().lower()
    s = s.replace('&', ' and ')
    s = re.sub(r'^proceedings\.?\s+(of\s+)?(the\s+)?', '', s)
    s = re.sub(r'^(?:19|20)\d{2}\b', ' ', s)                       # 行首年份
    s = re.sub(r'\b(?:19|20)\d{2}\b', ' ', s)                       # 年份
    s = re.sub(r'\b\d+(?:st|nd|rd|th)\b', ' ', s)                   # 24th / 31st
    s = re.sub(r'\b(?:twenty|thirty)[-\s]?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)\b', ' ', s)
    s = re.sub(r'\bv\.?\s*\d+\b', ' ', s)                           # V.1 / V.2
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


# ---------- 第一步（反例）----------
'''确定不是目录条目的写法：归一化后命中任一反例子串直接 None（后两步不再尝试）。

ICDM 原是反例（防吸附 ICDE），第七版归 B 后转正例、交第三步 top1 相似度区分；
后三条是实库误判修复（详见 paperMeta_test [5b] 与扫描输出）：'ACM Conferences'
0.867 相似误判 WWW，ICRITO 词命中 'infocom'、ICC 0.93 相似误判 INFOCOM。
待定：T-ZS-LLM-6 的 ACL-IJCNLP 联合写法命中 ACL；T-159 PACMMOD 非 SIGMOD 主会（正确 None）。
'''
VENUE_NEGATIVES = [
    'Workshop',   # workshop 一律不算目录正例（如 'ICML 2019 Workshop ...' 否则词命中 ICML）
    'ACM Conferences',
    'ICRITO',
    'IEEE International Conference on Communications',
]

_NEGATIVES = [normalize_venue(n).replace(' ', '') for n in VENUE_NEGATIVES]   # 反例紧凑串


# ---------- 第二步（词命中）----------
# 通用简称匹配时跳过的通用词，避免 'Web' / 'Data' 之类误命中。
# 后 5 个是 B 类并入后新增：这些词出现在其他 venue 的全称里（如 HPDC / SC 全称含
# 'Performance'，BIBM 全称含 'Bioinformatics'，I3D 全称含 'SIGGRAPH'），词命中会
# 把人家的全称抢走；停用后靠第三步相似度（全称自匹配恒 1.0）正确归位。
SHORT_STOPWORDS = {'the', 'and', 'for', 'web', 'data', 'ieee', 'acm', 'sig',
                   'conference', 'journal', 'transactions', 'international', 'advances',
                   'performance', 'group', 'bioinformatics', 'eurographics', 'siggraph'}


def _build_keyword_to_venues(catalog: dict, abbrs=None) -> dict:
    '''第二步（词命中）用：{关键词: 缩写}。
    关键词 = 目录缩写与 abbrs 额外简称归一化后的词（如 'sigkdd'），过滤掉
    太短 / 含非字母数字 / 停用词的词，避免 'the'、'acm' 之类通用词在词边界
    匹配时误命中。
    例：KDD -> {'kdd': 'KDD', 'sigkdd': 'KDD'}。
    '''
    keyword_to_venues = {}
    for short in catalog:
        tokens = {normalize_venue(short)}
        for extra in (abbrs or {}).get(short, []):
            tokens.add(normalize_venue(extra))
        for token in tokens:
            if len(token) >= 3 and token.isalnum() and token not in SHORT_STOPWORDS:
                existing = keyword_to_venues.setdefault(token, short)
                assert existing == short, f'关键词 {token!r} 被多个缩写占用：{existing} / {short}'
    return keyword_to_venues


# 预计算：目录 = 合并后的 _CATALOG，导入时算一次、全进程复用（模块即单例）。
_KEYWORD_TO_VENUES = _build_keyword_to_venues(_CATALOG, VENUE_ALIASES)
_TOKENS = sorted(_KEYWORD_TO_VENUES, key=len, reverse=True)   # 长 token 更特异，优先尝试（如 'sigkdd' 先于 'kdd'）


# ---------- 第三步（相似度）----------
MATCH_THRESHOLD = 0.85   # top1 最低相似度（A 类实测正例 0.89~1.00，负例 <=0.77；B 类扩充后需复核）


def _build_venue_names(catalog: dict, aliases=None) -> dict:
    '''第三步（相似度）用：{缩写: [该缩写的全部名称写法, ...]}。
    每个写法 = 全称或 aliases 别名（别名里也可能有简称 / 缩写期刊名）经
    normalize_venue() 归一化（小写、去 proceedings 前缀 / 届次 / 年份 / 标点）
    后，再删掉全部空格。两边都去空格，是为了对齐 Zotero / PDF 抓来的丢空格
    全称；等价写法去重。
    例：KDD -> ['acmsigkddconferenceonknowledgediscoveryanddatamining', 'sigkdd']。
    '''
    venue_names = {}
    for short, full in catalog.items():
        compact = []                       # 当前缩写的名称写法（去重）
        for text in [full] + list((aliases or {}).get(short, [])):
            c = normalize_venue(text).replace(' ', '')
            if c and c not in compact:
                compact.append(c)
        # 全称与别名归一化后全为空时，兜底用缩写本身，保证列表非空
        venue_names[short] = compact or [normalize_venue(short).replace(' ', '')]
    return venue_names


_VENUE_NAMES = _build_venue_names(_CATALOG, VENUE_ALIASES)

# 反例豁免集：目录官方名（含别名）本身含反例词的写法（NOSSDAV / CSFW 官方名含
# 'Workshop'）。compact 串与之完全一致时跳过反例否决，避免自家全称被自家反例拦下；
# 非 official 写法（如 'ICML 2019 Workshop ...'）不受豁免，照常否决。
_NEGATIVE_EXEMPT = {c for cands in _VENUE_NAMES.values() for c in cands if any(n in c for n in _NEGATIVES)}


# ---------- 匹配入口 ----------
# 输入统一为 {缩写: 全称} 的 kv 目录，与等级 / 来源完全解耦。
_CACHE = {}   # 查询缓存：归一化串 -> 结果；Zotero 里写法重复率极高，重复串直接查表


def match_venue(raw):
    '''raw -> 匹配到的目录缩写（A / B 合并查）；未命中（含空值 / None）返回 None。

    三步匹配，都先过 normalize_venue()（PDF 全称丢空格也能对齐）：
    第一步（反例）：命中 VENUE_NEGATIVES 子串直接 None；
    第二步（词命中）：串里完整包含某缩写 / 别名简称（词边界），直接返回该缩写；
    第三步（相似度）：与各缩写的名称写法（_VENUE_NAMES）比 SequenceMatcher
    相似度，最高分 top1 >= MATCH_THRESHOLD 才返回，否则 None。
    等级不在本函数职责内，拿到缩写后查 CCF_LEVEL 即得 'A' / 'B'。

    raw 正常是 Zotero item 里取出的 venue 字段（proceedingsTitle / conferenceName / ...），
    内部用 str(raw or '') 兜底，其它类型不保证语义。更多写法用例见 paperMeta_test.py [5b]。
    例：
        match_venue('Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1')
            -> 'KDD'
        match_venue('arXiv.org') -> None（arXiv 预印本的 venue 兜底在 citations.py，优先级低于子笔录用事实）
    '''
    if not str(raw or '').strip():
        return None
    norm = normalize_venue(raw)
    if norm in _CACHE:
        return _CACHE[norm]
    result = None
    compact = norm.replace(' ', '')
    # 第一步（反例）：命中反例子串直接 None——确定不是目录条目的写法，无论后两步
    # 多相似都不再尝试（防 'ACM Conferences' 吸附到 WWW、ICC 吸附到 INFOCOM 之类）；
    # 与官方名完全一致的串豁免（NOSSDAV / CSFW 官方名自带 'Workshop'）。
    if compact in _NEGATIVE_EXEMPT or not any(n in compact for n in _NEGATIVES):
        # 第二步（词命中）：归一化串里出现完整的缩写 / 别名简称（词边界）。
        for token in _TOKENS:
            if re.search(r'\b' + re.escape(token) + r'\b', norm):
                result = _KEYWORD_TO_VENUES[token]
                break
        # 第三步（相似度）：与各缩写的名称写法比字符相似度，取最高分且过阈值。
        # SequenceMatcher.ratio() = 2 × 匹配字符数 / 两串总长（0~1）：先找最长公共
        # 连续块，再在两侧递归找，只比字面重合、不看语序，故对丢空格 / 拼写差异宽容。
        if result is None:
            scores = ((max(SequenceMatcher(None, compact, t).ratio() for t in candidates), short)
                      for short, candidates in _VENUE_NAMES.items())
            top1, short = max(scores)
            if top1 >= MATCH_THRESHOLD:
                result = short
    _CACHE[norm] = result
    return result


# ---------- 可选：Zotero 扫描 ----------
def scan_zotero_venues():
    '''可选：扫描 Zotero 实际写法，统计命中数与未命中样例（不自动写入别名）。'''
    import requests
    from collections import Counter
    url = f'{ZOTERO_API}?limit=100&start='
    rows = []
    for start in range(0, 3000, 100):
        try:
            batch = requests.get(url + str(start), timeout=3).json()
        except Exception as e:
            print(f'[ccfCatalog] Zotero 不可用，跳过扫描：{e}')
            return {}, []
        if not batch:
            break
        rows.extend(batch)
    hits, unresolved = Counter(), Counter()
    for item in rows:
        data = item.get('data', {})
        raw = next((data.get(f) for f in VENUE_FIELDS if data.get(f)), '')
        short = match_venue(raw)
        if short:
            hits[f'{short}({CCF_LEVEL[short]})'] += 1   # 命中键带等级，便于核对 A/B 分布
        elif str(raw).strip():
            unresolved[str(raw)] += 1
    return dict(hits), [v for v, _ in unresolved.most_common(20)]


if __name__ == '__main__':
    hits, samples = scan_zotero_venues()
    print(f'[ccfCatalog] Zotero venue 命中数：{hits}')
    print(f'[ccfCatalog] 未解析 venue 样例：{samples}')
