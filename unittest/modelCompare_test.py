import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from pathlib import Path
from utils.datasets import DataList, getDataResults
basePath = Path('results/HA_neighbor/PEMS03')
sufPath = Path('results_tiny_test.json')
results = [
    DataList.load(basePath / '32-5a' / sufPath),
    DataList.load(basePath / '32-5-gpt5-2' / sufPath),
]
names = ['deepseek', 'gpt5.2 (T)']
print(getDataResults(results, names))