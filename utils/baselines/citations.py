''' 管理每个模型对应的引用名，以及论文元数据（venue / year / CCF）。

citations.json : 模型名 -> 引用键（保持原样，是 model 到论文的唯一映射）
papers.json    : 引用键 -> 元数据快照（空值字段省略），由 sync_from_zotero() 写出
CCF_LEVEL      : 缩写 -> 'A'/'B'（ccfCatalog.py，目录数据在同目录 ccfA/ccfB.json），venue 匹配与等级判定
ZoteroHandler  : 本文件内的 Zotero API 拉取与条目解析，由 CitationHandler 组合持有
Paper          : results.py 的论文条目 dataclass；paper() 查表构建，render(Paper.display) 按其显示名渲染

运行时默认只读快照，sync_from_zotero 需显式调用；不查论文元数据时输出与旧链路完全一致。
'''
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Callable

from .results import Paper, Results
from .ccfCatalog import CCF_LEVEL, VENUE_FIELDS, ZOTERO_API, match_venue, normalize_venue

HERE = Path(__file__).resolve().parent
PAPERS_PATH = HERE / 'papers.json'
# 写快照时空值字段直接省略（不写即未知），不写 'unknown' 哨兵；
# 读取时缺键 / 缺字段 / null 一律补空字符串，下游 formatter / 筛选行为不变。
META_FIELDS = ('venue', 'year', 'ccf')


class ZoteroHandler:
    '''Zotero 侧逻辑：分页拉取 + item 解析；由 CitationHandler 组合持有。

    只产出 {引用键: {venue, year, ccf}} 与统计，不读写 citations.json / papers.json，
    也不持有 CitationHandler 的状态；快照落盘与缓存刷新仍由 CitationHandler 负责。
    '''

    def __init__(self, api: str = ZOTERO_API):
        self.api = api

    def fetch_items(self, timeout: float = 3, limit: int = 100, max_pages: int = 50):
        '''分页拉全量 Zotero item；网络异常 / 非 200 / 非列表响应统一抛 RuntimeError。

        requests 延迟导入：离线只读快照的主链路（main.py 等）不因此强依赖该第三方包。
        '''
        import requests
        items = []
        for page in range(max_pages):
            try:
                resp = requests.get(self.api, params={'limit': limit, 'start': page * limit},
                                    timeout=timeout)
                resp.raise_for_status()
                batch = resp.json()
            except Exception as e:
                raise RuntimeError(f'Zotero 请求失败：{e}') from e
            if not isinstance(batch, list):
                raise RuntimeError(f'Zotero 响应不是列表（疑似 API 报错）：{str(batch)[:120]}')
            if not batch:
                break
            items.extend(batch)
        return items

    @staticmethod
    def _cite_key(data: dict):
        '''主键：archiveLocation 优先，否则 citationKey；两者不一致时返回冲突标记。'''
        vals = [str(v).strip() for v in (data.get('archiveLocation'), data.get('citationKey'))
                if v and re.match(r'^(T|O)-', str(v))]
        return (vals[0] if vals else None), (len(vals) > 1 and vals[0] != vals[1])

    @staticmethod
    def _year(date):
        '''Zotero date 形如 2018/7 / 2025-07-20 / 7/2022 / 六月 24, 2024；
        取串中第一个 19xx/20xx 年份，取不到则 ''（空字符串即未知）。'''
        m = re.search(r'\b((?:19|20)\d{2})\b', str(date or ''))
        return int(m.group(1)) if m else ''

    @staticmethod
    def _notes(items: list) -> dict:
        '''子笔记里的录用事实：{父条目 key: 从笔记推断出的 venue 串}，没有笔记的条目不出现在结果里。

        Zotero 里"已录用、待正式发表"常写成论文下的子笔记（如 'Comment: Accept by ICML 2026'）。
        note 与父条目的先后顺序不保证，所以单独扫一遍 item 列表，供主循环查表兜底。

        每条笔记的处理：剥 HTML 标签 → 按 'accepted by/to/at'、'to be published in' 等句式
        取第一句 → 去掉 'IJCAI2022' 这类连写年份，得到候选串。本函数只管取出候选串，
        是否采用、属于哪个目录由主循环交给 match_venue() 判定（正式字段优先）。

        例：'Comment: Accept by ICML 2026' -> 'ICML 2026'；
            'to be published in the 42nd IEEE ... (ICDE 26)' -> 全称保留，交给 match_venue 判 ICDE。
        页数、under review、仓库链接等没有录用语义的笔记直接跳过；workshop 类能提取，
        但会被 match_venue 的反例否决。完整写法清单见 unittest/paperMeta_test.py 的 [5]。
        '''
        notes = {}
        for item in items:
            data = item.get('data', {})
            if data.get('itemType') != 'note' or not data.get('parentItem'):
                continue
            text = re.sub(r'<[^>]+>', ' ', str(data.get('note') or ''))
            text = re.sub(r'\s+', ' ', text).strip()
            for pat in (r'accept(?:ed)?\s+(?:by|to|at|in)\s+(.+)',
                        r'(?:to\s+)?(?:published|appear)(?:\s+as\s+(?:a\s+)?(?:conference|journal)\s+paper)?'
                        r'\s+(?:in|at)\s+(.+)'):
                m = re.search(pat, text, re.I)
                if m:
                    venue = re.split(r'\.\s|[;]', m.group(1))[0]      # 只取第一句
                    venue = re.sub(r'(?<=[A-Za-z])(?:19|20)\d{2}\b', '', venue)   # IJCAI2022 -> IJCAI
                    notes[data['parentItem']] = venue.strip(' .,:')
                    break
        return notes

    def build_papers(self, items: list, verbose: bool = False):
        '''Zotero item 列表 -> {引用键: 非空元数据字段}；只有 verbose=True 才打印统计
        （item 数、引用键数、unresolved 种类、conflicts / badDates / duplicateKeys）。

        约定：空值字段不写入（不写即未知），读取端 meta() 补空串。

        处理流程：
          0. 预扫子笔记：_notes() 收集 {父条目 key: 录用串}，供第 2 步兜底
          1. 过滤与取键
             - 跳过 itemType 为 attachment / note 的条目
             - 引用键取 archiveLocation，其次 citationKey，只认 'T-' / 'O-' 前缀
             - 两字段都有效但不一致时仍取 archiveLocation，只记 conflicts
          2. venue / ccf
             2.1 候选链：VENUE_FIELDS 中第一个非空字段 -> 子笔记录用串，依次交给 match_venue()
                 - 命中目录写 venue 缩写；未命中的非空候选串计入 unresolved
                 - 预印本与正式论文一视同仁，字段能命中就按等级判
             2.2 arXiv 兜底：候选链全未命中且 repository 是 arXiv -> venue='arXiv'，ccf 留空
                 - 子笔录用事实优先于 arXiv 兜底
             - ccf 按 CCF_LEVEL 写 'A' / 'B'；都判不出则 ccf 留空
          3. year：date 中第一个 19xx/20xx 年份；date 非空但取不到记 badDates
          4. 写表：同键后写覆盖并记 duplicateKeys

        已知局限：高优先级字段可能顶掉能匹配的低优先级字段，见 ccfCatalog.py 的 VENUE_FIELDS 注释。

        例（note 的 parentItem 是父条目的 item key，不是引用键；完整用例见 paperMeta_test.py 的 [5]）：
            正式字段命中：
                {'data': {'key': 'a1', 'itemType': 'conferencePaper', 'archiveLocation': 'T-460',
                          'date': '2025-07-13',
                          'proceedingsTitle': 'Proceedings of the 42nd International '
                                              'Conference on Machine Learning'}}
                -> {'T-460': {'venue': 'ICML', 'year': 2025, 'ccf': 'A'}}
            仅子笔记录用事实：
                {'data': {'key': 'b2', 'itemType': 'preprint', 'archiveLocation': 'T-455',
                          'date': '2026-05-19', 'repository': 'arXiv'}}
                {'data': {'itemType': 'note', 'parentItem': 'b2',
                          'note': 'Comment: Accept by ICML 2026'}}
                -> {'T-455': {'venue': 'ICML', 'year': 2026, 'ccf': 'A'}}
            arXiv 兜底（候选链全未命中）：
                {'data': {'key': 'c3', 'itemType': 'preprint', 'archiveLocation': 'T-467',
                          'repository': 'arXiv'}}
                -> {'T-467': {'venue': 'arXiv'}}
        '''
        papers, unresolved = {}, set()
        conflicts, bad_dates, duplicates = 0, 0, 0
        notes = self._notes(items)   # 子笔记录用事实，venue 未命中时兜底
        for item in items:
            # (1) 过滤无关条目，解析引用键
            data = item.get('data', {})
            item_type = data.get('itemType', '')
            if item_type in ('attachment', 'note'):
                continue
            key, conflict = self._cite_key(data)
            if not key:
                continue
            if conflict:
                conflicts += 1

            # (2.1) venue 候选链：正式字段原文优先，子笔记录用事实（'Comment: Accept by ICML 2026'）兜底
            candidates = (
                next((data.get(f) for f in VENUE_FIELDS if data.get(f)), ''),   # 正式字段原文
                notes.get(data.get('key', ''), ''),                            # 子笔记录用事实
            )
            candidates = [cand for cand in candidates if cand.strip()]         # 空候选直接剔除
            short = None
            for cand in candidates:
                short = match_venue(cand)
                if short:
                    break
                unresolved.add(cand)   # 未命中串计入统计
            # (2.2) arXiv 兜底：候选链（含子笔记事实）全未命中且仓库为 arXiv -> venue='arXiv'，
            # ccf 留空。必须在候选链之后：'Comment: Accept by ICML 2026' 的事实要优先于预印本身份。
            venue = short or ''
            if not venue and 'arxiv' in normalize_venue(data.get('repository')):
                venue = 'arXiv'
            ccf = CCF_LEVEL.get(short, '')

            # (3) 年份：解析 date，取不到记 badDates
            year = self._year(data.get('date'))
            if not year and str(data.get('date') or '').strip():
                bad_dates += 1

            # (4) 写表：同键后写覆盖，空值字段省略
            if key in papers:
                duplicates += 1
            papers[key] = {k: v for k, v in (('venue', venue), ('year', year), ('ccf', ccf))
                           if v != '' and v is not None}

        if verbose:
            print(f'[citations] {len(items)} 个 Zotero item -> {len(papers)} 条引用键，'
                  f'unresolved venue {len(unresolved)} 种，conflicts {conflicts}，'
                  f'badDates {bad_dates}，duplicateKeys {duplicates}')
        return papers


class CitationHandler:
    zotero = ZoteroHandler()          # Zotero 拉取/解析委托给本文件内的 handler

    @staticmethod
    def parse_rawinfo(srcpath, destpath='citations.json'):
        ''' 根据给定的格式提取模型，这里以 .md 格式，3级标题 引用名+空格+模型名为例；返回 dict[模型名] = 引用名；并且把结果存起来；
         参见 unittest/baselineHorizons_test.py 中被注释的 INIT 调用（全仓库唯一调用点）
         当前未使用：仅用于一次性初始化 citations.json；运行时只读快照，保留备用，非废弃接口 '''
        result = {}
        try:
            with open(srcpath, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    # 检查是否为3级标题
                    if line.startswith('### '):
                        content = line[3:].strip()
                        # 分割引用名和模型名（只分割第一个空格）
                        if ' ' in content:
                            citation, model_name = content.split(' ', 1)
                            result[model_name.strip()] = citation.strip()
        except FileNotFoundError:
            print(f"文件未找到：{srcpath}")
        except Exception as e:
            print(f"解析文件时出错：{e}")
        with open(destpath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    citation_data = None
    paper_data = None

    @staticmethod
    def init_cites():
        if CitationHandler.citation_data is not None:
            return
        try:
            current_file = Path(__file__).resolve()
            current_dir = current_file.parent
            path = current_dir / 'citations.json'
            with open(path, 'r', encoding='utf-8') as f:
                CitationHandler.citation_data = json.load(f)
        except FileNotFoundError:
            print(f"文件未找到：{path}")
            CitationHandler.citation_data = {}

    @staticmethod
    def init_papers():
        '''懒加载 papers.json 快照。'''
        if CitationHandler.paper_data is not None:
            return
        if not PAPERS_PATH.exists():
            print(f'[citations] 找不到 {PAPERS_PATH.name}，请先运行 sync_from_zotero()')
            CitationHandler.paper_data = {}
            return
        with open(PAPERS_PATH, 'r', encoding='utf-8') as f:
            CitationHandler.paper_data = json.load(f)

    @staticmethod
    def render(results: Results, formatter: Callable[[Paper], str] = Paper.display_cite,
               defaultCite: str = '', externalCite: dict = None):
        ''' 就地修改一个 Results 对象，把每个模型名替换成 formatter(Paper) 的显示名。

        每个模型经 paper(model, externalCite) 查成 Paper 再交给 formatter；
        默认 Paper.display_cite 为 cite 风格，paper 风格传 Paper.display。
        例：res = Results.get_baseline_results()
            render(res) 后 Result.model: 'STSGCN' -> 'STSGCN \\cite{T-87}'，返回 res 本身
            render(res, formatter=Paper.display) 后 'STGCN' -> 'STGCN (IJCAI 2018) \\cite{T-28}'，HA 原样
            render(res, defaultCite='T-138') -> 无引用键的 'HA' 也标成 'HA \\cite{T-138}'
        '''
        for result in results: # result是引用，就地修改
            info = CitationHandler.paper(result.model, externalCite)
            if not info.cite and defaultCite:
                info = Paper(model=info.model, cite=defaultCite)
            result.model = formatter(info)
        return results

    # ---------- 论文元数据：查表 ----------

    @staticmethod
    def meta(citeKey: str):
        '''引用键 -> {venue, year, ccf}；缺键 / 缺字段 / null 一律补空字符串。

        例：meta('T-28')     -> {'venue': 'IJCAI', 'year': 2018, 'ccf': 'A'}
            meta('T-不存在') -> {'venue': '', 'year': '', 'ccf': ''}
        '''
        CitationHandler.init_papers()
        raw = CitationHandler.paper_data.get(citeKey) or {}
        return {k: raw.get(k) or '' for k in META_FIELDS}

    @staticmethod
    def cite_of_model(model: str, externalCite: dict | None = None):
        '''模型名 -> 引用键；externalCite 让运行时临时补键的模型也能查元数据。

        例：cite_of_model('STGCN')                    -> 'T-28'
            cite_of_model('HA')                       -> ''
            cite_of_model('ARIMA', {'ARIMA': 'O-64'}) -> 'O-64'
        '''
        CitationHandler.init_cites()
        return (CitationHandler.citation_data | (externalCite or {})).get(model, '')

    @staticmethod
    def paper(model: str, externalCite: dict | None = None) -> Paper:
        '''模型名 -> Paper：引用键与元数据一次查齐；缺键 / 缺字段 / null 一律留空。

        例：paper('STGCN') -> Paper('STGCN', 'T-28', 'IJCAI', 2018, 'A')
            paper('HA')    -> Paper('HA')（无引用键、无元数据，unknown() 成立）
            paper('ARIMA', {'ARIMA': 'O-64'}) -> cite='O-64'，元数据按该键查
        '''
        CitationHandler.init_cites()
        CitationHandler.init_papers()
        cite = (CitationHandler.citation_data | (externalCite or {})).get(model, '')
        raw = CitationHandler.paper_data.get(cite) or {}
        year = raw.get('year')
        return Paper(model=model, cite=cite,
                     venue=raw.get('venue') or '',
                     year=year if isinstance(year, int) else None,
                     ccf=raw.get('ccf') or '')

    # ---------- 论文元数据：从 Zotero 构建快照（委托 ZoteroHandler） ----------

    @staticmethod
    def sync_from_zotero(out_path=None, timeout: float = 3, limit: int = 100,
                         max_pages: int = 50, verbose: bool = True):
        '''分页拉 Zotero 本地 API 生成 papers.json；Zotero 不可用或返回空结果时
        保留现有快照并返回 None，绝不覆盖成空表。

        拉取与条目解析交给 CitationHandler.zotero，统计由 build_papers(verbose=...) 就地打印；
        本方法只做快照落盘、缓存刷新和 missingKeys 统计（依赖 citations.json）。
        快照只写非空字段，空值省略即未知；成功返回构建出的 {引用键: 元数据}。
        '''
        zotero = CitationHandler.zotero
        try:
            items = zotero.fetch_items(timeout=timeout, limit=limit, max_pages=max_pages)
            if not items:
                raise RuntimeError('Zotero 返回 0 条 item（未登录 / API 地址有误 / 库为空）')
            built = zotero.build_papers(items, verbose=verbose)
        except RuntimeError as e:
            if verbose:
                print(f'[citations] Zotero 不可用，保留现有快照：{e}')
            return None

        CitationHandler.init_cites()
        missing_keys = len(set(CitationHandler.citation_data.values()) - set(built))
        payload = {k: built[k] for k in sorted(built)}
        out = Path(out_path or PAPERS_PATH)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        CitationHandler.paper_data = payload
        if verbose:
            print(f'[citations] 写出 {out}；missingKeys {missing_keys}')
        return built
