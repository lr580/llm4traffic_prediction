''' 管理每个模型对应的引用名，以及论文元数据（venue / year / CCF）。

citations.json : 模型名 -> 引用键（保持原样，是 model 到论文的唯一映射）
papers.json    : 引用键 -> 元数据快照（空值字段省略），由 sync_from_zotero() 写出
CCF_A_CATALOG  : 缩写 -> 全称（ccfCatalog.py 代码常量），用于判断 CCF-A 与 venue 匹配
ZoteroHandler  : 本文件内的 Zotero API 拉取与条目解析，由 CitationHandler 组合持有

运行时默认只读快照，sync_from_zotero 需显式调用；不查论文元数据时输出与旧链路完全一致。
'''
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Callable

import requests

from .results import Results
from .ccfCatalog import CCF_A_VENUES, VENUE_FIELDS, ZOTERO_API, match_venue

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
        '''分页拉全量 Zotero item；API 异常统一抛 RuntimeError。'''
        items = []
        for page in range(max_pages):
            try:
                batch = requests.get(self.api, params={'limit': limit, 'start': page * limit},
                                     timeout=timeout).json()
            except Exception as e:
                raise RuntimeError(str(e)) from e
            if not isinstance(batch, list) or not batch:
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

    def build_papers(self, items: list, verbose: bool = False):
        '''Zotero item 列表 -> {引用键: 非空元数据字段}；统计由 verbose 就地打印，不返回。

        空值字段不写入（不写即未知），不写 'unknown' 哨兵；读取端 meta() 补空串。
        verbose=True 时打印 Zotero item 数、引用键数、未解析 venue 种类数以及
        conflicts / badDates / duplicateKeys；missingKeys 依赖 citations.json，
        由 CitationHandler.sync_from_zotero() 自行统计并打印。

        处理规则（对应循环内的 (1)~(4)）：
        (1) 过滤无关条目、解析引用键：itemType 为 attachment / note 直接跳过；
            引用键只认 archiveLocation / citationKey 里以 'T-' 或 'O-' 开头的值，
            archiveLocation 优先；两者都有效但不一致时仍取前者，只记 conflicts；
            两字段都无有效值（空值或前缀不符）就跳过。被跳过的条目不进 papers，
            也不触发后面的任何计数，但 verbose 的 Zotero item 数仍是原始 items 长度。
            本步没有独立计数，过滤量只能由 len(items) 与 papers 数间接估算。
            2026-11 实测当前库：1669 条 item 里过滤 1026 条
            （attachment 966 / note 45 / 无有效键 15），保留 643 条且全部写入 papers；
            数字随 Zotero 库变化，仅示量级。
        (2) venue：按 VENUE_FIELDS 顺序取第一个非空字段，全空为 ''，
            match_venue('') 返回 None 即 venue 留空；未命中的非空原始串计入 unresolved。
            命中 A 类目录才写 ccf='A'；预印本与其它 itemType 一视同仁，
            只要 venue 字段命中 A 类目录（如填了 'KDD'）同样写 ccf='A'。
            已知漏判（高优先级字段顶掉可匹配的低优先级字段）见 ccfCatalog.py 的
            VENUE_FIELDS 注释。
        (3) 年份：从 date 取第一个 19xx/20xx 年份，取不到且 date 非空记 badDates。
        (4) 写表：同键后写覆盖并记 duplicateKeys；空值字段省略。

        例（items 为 Zotero 本地 API 返回的元素，只列 data 中用到的字段）：
            items = [
                {'data': {'itemType': 'conferencePaper', 'archiveLocation': 'T-460',
                          'date': '2025-07-13',
                          'proceedingsTitle': 'Proceedings of the 42nd International '
                                              'Conference on Machine Learning'}},
                {'data': {'itemType': 'preprint', 'archiveLocation': 'T-59',
                          'date': '2021-03-29', 'repository': 'arXiv'}},
            ]
            -> {'T-460': {'venue': 'ICML', 'year': 2025, 'ccf': 'A'},
                'T-59':  {'year': 2021}}
            arXiv 未命中 A 类目录，venue/ccf 为空故不写入；attachment / note 与
            无有效引用键的条目会被跳过，不计入结果。
        '''
        papers, unresolved = {}, set()
        conflicts, bad_dates, duplicates = 0, 0, 0
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

            # (2) venue 原始串 -> CCF-A 缩写 -> venue / ccf
            raw = next((data.get(f) for f in VENUE_FIELDS if data.get(f)), '')
            short = match_venue(raw)
            if short is None and str(raw).strip():
                unresolved.add(str(raw))
            venue = short or ''
            ccf = 'A' if short in CCF_A_VENUES else ''

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
    def init():
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
    def latexFormatter(model: str, cite: str):
        return f"{model} \\cite{{{cite}}}"

    @staticmethod
    def normalFormatter(model: str, cite: str):
        return f"{cite} ({model})"

    @staticmethod
    def render(results: Results, formatter: Callable[[str, str], str] = latexFormatter, defaultCite: str = '', externalCite: dict = {}):
        ''' 就地修改一个 Results 对象，把每个模型名替换成模型名+引用名，替换格式为 formatter

        例：res = Results.get_baseline_results()
            render(res) 后 Result.model: 'STSGCN' -> 'STSGCN \\cite{T-87}'（默认 latexFormatter），返回 res 本身
            render(res, formatter=normalFormatter) 后 Result.model: 'STSGCN' -> 'T-87 (STSGCN)'
            render(res, defaultCite='T-138') -> 无引用键的 'HA' 也标成 'HA \\cite{T-138}'
        '''
        CitationHandler.init()
        map = CitationHandler.citation_data | externalCite
        assert map is not None
        for result in results: # result是引用，就地修改
            model = result.model
            if model in map:
                cite = map[model]
            elif defaultCite:
                cite = defaultCite
            else:
                continue
            result.model = formatter(model, cite)
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
        CitationHandler.init()
        return (CitationHandler.citation_data | (externalCite or {})).get(model, '')

    @staticmethod
    def paperFormatter(model: str, cite: str, meta: dict):
        '''元数据标注：Model (KDD 2025)；venue/year 缺失时原样返回。

        空字符串 / None / 缺字段都算缺失，不加标注。
        例：paperFormatter('STGCN', 'T-28', meta('T-28')) -> 'STGCN (IJCAI 2018)'
            paperFormatter('HA', '', meta(''))             -> 'HA'
            paperFormatter('X', '', {})                    -> 'X'（不存字段即未知）
        '''
        venue, year = meta.get('venue'), meta.get('year')
        if not venue or not year:
            return model
        return f'{model} ({venue} {year})'

    @staticmethod
    def citeFormatter(model: str, cite: str):
        '''引用注入：Model \\cite{T-xxx}。

        例：citeFormatter('STGCN', 'T-28') -> 'STGCN \\cite{T-28}'
            citeFormatter('HA', '')        -> 'HA'（无引用键原样返回）
        '''
        return f'{model} \\cite{{{cite}}}' if cite else model

    @staticmethod
    def paperCiteFormatter(model: str, cite: str, meta: dict):
        '''显示名 = 元数据标注 + 引用注入；两项各自独立判断，缺失时跳过。

        例：paperCiteFormatter('STGCN', 'T-28', meta('T-28'))
                -> 'STGCN (IJCAI 2018) \\cite{T-28}'
            paperCiteFormatter('HA', '', meta('')) -> 'HA'
        '''
        model = CitationHandler.paperFormatter(model, cite, meta)
        return CitationHandler.citeFormatter(model, cite)

    # ---------- 论文元数据：从 Zotero 构建快照（委托 ZoteroHandler） ----------

    @staticmethod
    def sync_from_zotero(out_path=None, timeout: float = 3, limit: int = 100,
                         max_pages: int = 50, verbose: bool = True):
        '''分页拉 Zotero 本地 API 生成 papers.json；Zotero 不可用时保留现有快照并返回 None。

        拉取与条目解析交给 CitationHandler.zotero，统计由 build_papers(verbose=...) 就地打印；
        本方法只做快照落盘、缓存刷新和 missingKeys 统计（依赖 citations.json）。
        快照只写非空字段，空值省略即未知；成功返回构建出的 {引用键: 元数据}。
        '''
        zotero = CitationHandler.zotero
        try:
            items = zotero.fetch_items(timeout=timeout, limit=limit, max_pages=max_pages)
            built = zotero.build_papers(items, verbose=verbose)
        except RuntimeError as e:
            if verbose:
                print(f'[citations] Zotero 不可用，保留现有快照：{e}')
            return None

        CitationHandler.init()
        missing_keys = len(set(CitationHandler.citation_data.values()) - set(built))
        payload = {k: built[k] for k in sorted(built)}
        out = Path(out_path or PAPERS_PATH)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        CitationHandler.paper_data = payload
        if verbose:
            print(f'[citations] 写出 {out}；missingKeys {missing_keys}')
        return built
