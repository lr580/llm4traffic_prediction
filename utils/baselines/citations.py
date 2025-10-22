''' 管理每个模型对应的引用名，方便载入导论文。引用名列表可以根据自己的 .bib 来修改 '''
import json
from pathlib import Path
from .results import Results
from typing import Callable
class CitationHandler:
    @staticmethod
    def parse_rawinfo(srcpath, destpath='citations.json'):
        ''' 根据给定的格式提取模型，这里以 .md 格式，3级标题 引用名+空格+模型名为例；返回 dict[模型名] = 引用名；并且把结果存起来；
         参见 unittest/baselineHorizons_test.py '''
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
    def latexFormatter(model: str, cite: str):
        return f"{model} \\cite{{{cite}}}"
    
    @staticmethod
    def normalFormatter(model: str, cite: str):
        return f"{cite} ({model})"

    @staticmethod
    def render(results: Results, formatter: Callable[[str, str], str] = latexFormatter, defaultCite:str='', externalCite: dict = {}):
        ''' 就地修改一个 Results 对象，把每个模型名替换成模型名+引用名，替换格式为 formatter'''
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
    