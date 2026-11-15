'''CCF 目录与 venue 解析（纯代码常量，无数据文件依赖）。

CCF_A_CATALOG 是唯一的目录数据：缩写 -> 全称，判断是否 CCF-A 查 key 即可。
运行时目录就是 CCF_A_CATALOG；不读取数据文件，也不读取 PDF。

用法：
    python -m utils.baselines.ccfCatalog  # 统计 Zotero 实际写法命中情况
'''
from __future__ import annotations
import re
from difflib import SequenceMatcher


# ---------- 目录数据 ----------
# CCF 第七版（2026）A 类目录：缩写 -> 全称。目录不会频繁变动，换版时整体替换本表。
CCF_A_CATALOG = {
    'AAAI': 'AAAIConferenceonArtificialIntelligence',
    'ACL': 'AnnualMeetingoftheAssociationforComputational Linguistics',
    'ACMMM': 'ACMInternationalConferenceonMultimedia',
    'AI': 'ArtificialIntelligence',
    'ASE': 'InternationalConferenceonAutomatedSoftwareEngineering',
    'ASPLOS': 'InternationalConferenceonArchitecturalSupportfor ProgrammingLanguagesandOperatingSystems',
    'Bioinformatics': 'Bioinformatics',
    'CAV': 'InternationalConferenceonComputerAidedVerification',
    'CCS': 'ACMConferenceonComputerandCommunications Security',
    'CHI': 'ACMConferenceonHumanFactorsinComputing Systems',
    'CRYPTO': 'InternationalCryptologyConference',
    'CSCW': 'ACMConferenceOnComputer-SupportedCooperative WorkAndSocialComputing',
    'CVPR': 'IEEE/CVFComputerVisionandPatternRecognition Conference',
    'DAC': 'DesignAutomationConference',
    'EUROCRYPT': 'InternationalConferenceontheTheoryandApplications of CryptographicTechniques',
    'EuroSys': 'EuropeanConferenceonComputerSystems',
    'FAST': 'USENIXConferenceonFileandStorageTechnologies',
    'FM': 'InternationalSymposiumonFormalMethods',
    'FOCS': 'IEEEAnnualSymposiumonFoundationsofComputer Science',
    'FSE': 'ACMInternationalConferenceontheFoundationsof SoftwareEngineering',
    'HPCA': 'IEEEInternationalSymposiumonHighPerformance ComputerArchitecture',
    'HPDC': 'TheInternationalACMSymposiumonHigh-Performance ParallelandDistributedComputing',
    'IANDC': 'InformationandComputation',
    'ICCV': 'InternationalConferenceonComputerVision',
    'ICDE': 'IEEEInternationalConferenceonDataEngineering',
    'ICLR': 'InternationalConferenceonLearningRepresentations',
    'ICML': 'InternationalConferenceonMachineLearning',
    'ICSE': 'InternationalConferenceonSoftwareEngineering',
    'IEEEVIS': 'IEEEVisualizationConference',
    'IJCAI': 'InternationalJointConferenceonArtificialIntelligence',
    'IJCV': 'InternationalJournalofComputerVision',
    'IJHCS': 'InternationalJournalofHuman-ComputerStudies',
    'INFOCOM': 'IEEEInternationalConferenceonComputerCommunications',
    'ISCA': 'InternationalSymposiumonComputerArchitecture',
    'ISSTA': 'InternationalSymposiumonSoftwareTestingandAnalysis',
    'JACM': 'JournaloftheACM',
    'JMLR': 'JournalofMachineLearningResearch',
    'JSAC': 'IEEEJournalonSelectedAreasinCommunications',
    'KDD': 'ACMSIGKDDConferenceonKnowledgeDiscoveryand DataMining',
    'LICS': 'ACM/IEEESymposiumonLogicinComputerScience',
    'MICRO': 'IEEE/ACMInternationalSymposiumonMicroarchitecture',
    'MobiCom': 'ACMInternationalConferenceonMobileComputingand Networking',
    'NDSS': 'NetworkandDistributedSystemSecuritySymposium',
    'NSDI': 'SymposiumonNetworkSystemDesignandImplementation',
    'NeurIPS': 'ConferenceonNeuralInformationProcessingSystems',
    'OOPSLA': 'ConferenceonObject-OrientedProgrammingSystems, Languages,andApplications',
    'OSDI': 'USENIXSymposiumonOperatingSystemsDesignand Implementation',
    'PLDI': 'ACMSIGPLANConferenceonProgrammingLanguage DesignandImplementation',
    'POPL': 'ACMSIGPLAN-SIGACTSymposiumonPrinciplesof ProgrammingLanguages',
    'PPoPP': 'ACMSIGPLANSymposiumonPrinciples&Practiceof ParallelProgramming',
    'RTSS': 'IEEEReal-TimeSystemsSymposium',
    'SC': 'InternationalConferenceforHighPerformance Computing,Networking,Storage,andAnalysis',
    'SCIS': 'ScienceChinaInformationSciences',
    'SICOMP': 'SIAMJournalonComputing',
    'SIGCOMM': 'ACMInternationalConferenceonApplications,Technologies, Architectures,andProtocolsforComputerCommunication',
    'SIGGRAPH': 'ACMSpecialInterestGrouponComputerGraphics',
    'SIGIR': 'InternationalACMSIGIRConferenceonResearchand DevelopmentinInformationRetrieval',
    'SIGMOD': 'ACMSIGMODConference',
    'SODA': 'ACM-SIAMSymposiumonDiscreteAlgorithms',
    'SOSP': 'ACMSymposiumonOperatingSystemsPrinciples',
    'STOC': 'ACMSymposiumontheTheoryofComputing',
    'TACO': 'ACMTransactionsonArchitectureandCodeOptimization',
    'TC': 'IEEETransactionsonComputers',
    'TCAD': 'IEEETransactionsonComputer-AidedDesignofIntegrated CircuitsandSystems',
    'TDSC': 'IEEETransactionsonDependableandSecureComputing',
    'TIFS': 'IEEETransactionsonInformationForensicsandSecurity',
    'TIP': 'IEEETransactionsonImageProcessing',
    'TIT': 'IEEETransactionsonInformationTheory',
    'TKDE': 'IEEETransactionsonKnowledgeandDataEngineering',
    'TMC': 'IEEETransactionsonMobileComputing',
    'TMM': 'IEEETransactionsonMultimedia',
    'TOCHI': 'ACMTransactionsonComputer-HumanInteraction',
    'TOCS': 'ACMTransactionsonComputerSystems',
    'TODS': 'ACMTransactionsonDatabaseSystems',
    'TOG': 'ACMTransactionsonGraphics',
    'TOIS': 'ACMTransactionsonInformationSystems',
    'TON': 'IEEETransactionsonNetworking',
    'TOPLAS': 'ACMTransactionsonProgrammingLanguagesand Systems',
    'TOS': 'ACMTransactionsonStorage',
    'TOSEM': 'ACMTransactionsonSoftwareEngineeringand Methodology',
    'TPAMI': 'IEEETransactionsonPatternAnalysisandMachine Intelligence',
    'TPDS': 'IEEETransactionsonParallelandDistributedSystems',
    'TSC': 'IEEETransactionsonServicesComputing',
    'TSE': 'IEEETransactionsonSoftwareEngineering',
    'TVCG': 'IEEETransactionsonVisualizationandComputer Graphics',
    'UIST': 'ACMSymposiumonUserInterfaceSoftwareand Technology',
    'USENIXSecurity': 'USENIXSecuritySymposium',
    'UbiComp': 'ACMinternationaljointconferenceonPervasiveand UbiquitousComputing',
    'VLDB': 'InternationalConferenceonVeryLargeDataBases',
    'VLDBJ': 'TheVLDBJournal',
    'VR': 'IEEEConferenceonVirtualRealityand3DUserInterfaces',
    'WWW': 'TheWebConference',
}

CCF_A_VENUES = frozenset(CCF_A_CATALOG)

# ---------- Zotero 数据源 ----------
# 本地 API 地址与 venue 字段优先级：citations.py 的 sync_from_zotero() 与
# 下方 scan_zotero_venues() 共用这一份，避免两处各写一份。
ZOTERO_API = 'http://127.0.0.1:23119/api/users/0/items'

# 候选 venue 字段，按优先级取第一个非空作为论文 venue。
# 已知问题：第一个非空字段可能顶掉可匹配的低优先级字段（漏判），例如
# T-S-9 的 series "SIGMOD/PODS '25" 可匹配 SIGMOD，却被 proceedingsTitle
# 'Companion of the 2025 ...' 顶掉；T-ZS-LLM-6 / T-391 同类。
VENUE_FIELDS = ['proceedingsTitle', 'conferenceName', 'publicationTitle', 'repository', 'series']

# 人工别名：全称之外的匹配候选，第二、三步共用。
VENUE_ALIASES = {
    'CVPR':   ['IEEE/CVF Conference on Computer Vision and Pattern Recognition',
               'IEEE Conference on Computer Vision and Pattern Recognition'],
    'IJCAI':  ['Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August 10-16, 2019'],
    'KDD':    ['SIGKDD'],
    'NeurIPS':['Advances in Neural Information Processing Systems', 'Neural Information Processing Systems'],
    'WWW':    ['ACM Web Conference', 'World Wide Web Conference'],
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
'''与某 CCF-A 全称高度相似、但确定不是该 venue 的写法。
归一化后命中任一反例子串的 raw 直接判 None（后两步都不再尝试）。
背景：'IEEE International Conference on Data Mining' (ICDM, CCF-B) 与
ICDE 全称仅差 'Mining/Engineering'，第三步相似度过阈值曾误判为 ICDE。'''
VENUE_NEGATIVES = [
    'IEEE International Conference on Data Mining',   # ICDM（含 ... Workshops 变体）
]

'''已知问题（2026-11 全库 632 篇实测，3 篇残余错误，均未进入 baselineResults.csv）：
- T-ZS32  '...Reliability, Infocom Technologies and Optimization (ICRITO)' 误判 INFOCOM
- T-96    'ICC 2020 - IEEE International Conference on Communications (ICC)' 误判 INFOCOM
  两者同 ICDM 一个道理，修法：往本表补对应反例。
- T-ZS-LLM-6  ACL-IJCNLP 2021 联合会议写法漏判（是否算 ACL-A 待定）。
- match_venue('ACM Conferences') 误判 WWW（'acmconferences' vs 'acmwebconference'
  相似度 0.867 过阈值）。当前无条目触发，但 ACM DL 抓取的 conferencePaper 常只填
  series='ACM Conferences'，一旦出现即误判。修法：往本表补 'ACM Conferences' 反例。
另：T-159 'Proc. ACM Manag. Data'（PACMMOD 卷）经确认非 SIGMOD 主会，正确返回 None。
'''

_NEGATIVES = [normalize_venue(n).replace(' ', '') for n in VENUE_NEGATIVES]   # 反例紧凑串


# ---------- 第二步（词命中）----------
# 通用简称匹配时跳过的通用词，避免 'Web' / 'Data' 之类误命中
SHORT_STOPWORDS = {'the', 'and', 'for', 'web', 'data', 'ieee', 'acm', 'sig',
                   'conference', 'journal', 'transactions', 'international', 'advances'}


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


# 预计算：目录固定为 CCF_A_CATALOG，导入时算一次、全进程复用（模块即单例）。
_KEYWORD_TO_VENUES = _build_keyword_to_venues(CCF_A_CATALOG, VENUE_ALIASES)
_TOKENS = sorted(_KEYWORD_TO_VENUES, key=len, reverse=True)   # 长 token 更特异，优先尝试（如 'sigkdd' 先于 'kdd'）


# ---------- 第三步（相似度）----------
MATCH_THRESHOLD = 0.85   # top1 最低相似度（实测正例 0.89~1.00，负例 <=0.77）


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


_VENUE_NAMES = _build_venue_names(CCF_A_CATALOG, VENUE_ALIASES)


# ---------- 匹配入口 ----------
# 输入统一为 {缩写: 全称} 的 kv 目录，与等级 / 来源完全解耦。
_CACHE = {}   # 查询缓存：归一化串 -> 结果；Zotero 里写法重复率极高，重复串直接查表


def match_venue(raw):
    '''raw -> 匹配到的目录缩写；未命中（含空值 / None）返回 None。

    三步匹配，都先过 normalize_venue()（PDF 全称丢空格也能对齐）：
    第一步（反例）：命中 VENUE_NEGATIVES 子串直接 None；
    第二步（词命中）：串里完整包含某缩写 / 别名简称（词边界），直接返回该缩写；
    第三步（相似度）：与各缩写的名称写法（_VENUE_NAMES）比 SequenceMatcher
    相似度，最高分 top1 >= MATCH_THRESHOLD 才返回，否则 None。

    raw 正常是 Zotero item 里取出的 venue 字段（proceedingsTitle / conferenceName / ...），
    内部用 str(raw or '') 兜底，其它类型不保证语义。写法很不统一，例如：
        会议全称   'Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1'
        带届次地点 'Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August 10-16, 2019'
        期刊名     'IEEE Transactions on Knowledge and Data Engineering'
        简称       'SIGKDD'
        仓库名     'arXiv.org'
        缺失       '' / None
    前四例分别解析为 KDD / IJCAI / TKDE / KDD；'arXiv.org' 不在 A 类目录
    返回 None，空值与无法解析的写法同样返回 None。
    '''
    if not str(raw or '').strip():
        return None
    norm = normalize_venue(raw)
    if norm in _CACHE:
        return _CACHE[norm]
    result = None
    compact = norm.replace(' ', '')
    # 第一步（反例）：命中反例子串直接 None——确定不是 CCF-A 的写法，无论后两步
    # 多相似都不再尝试（防 'Data Mining' 吸附到 ICDE 之类）。
    if not any(n in compact for n in _NEGATIVES):
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
            hits[short] += 1
        elif str(raw).strip():
            unresolved[str(raw)] += 1
    return dict(hits), [v for v, _ in unresolved.most_common(20)]


if __name__ == '__main__':
    hits, samples = scan_zotero_venues()
    print(f'[ccfCatalog] Zotero venue 命中数：{hits}')
    print(f'[ccfCatalog] 未解析 venue 样例：{samples}')
