import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from utils.datasets import DataList, getDataResults
list1 = DataList.load('results/plain/PEMS03/64/results_tiny_test.json')
list2 = DataList.load('results/HA/PEMS03/64-1/results_tiny_test.json')
list3 = DataList.load('results/HA_neighbor/PEMS03/64/results_tiny_test.json')
lists = [list1, list2, list3]
names = ['Plain', 'HA', 'HA_Nei']
print('PEMS03-64:')
print(getDataResults(lists, names))

# list1 = DataList.load(f'results/plain/PEMS03/32/results_tiny_test.json')
# list2 = DataList.load(f'results/neighbor/PEMS03/32-2/results_tiny_test.json')
# list3 = DataList.load(f'results/HA/PEMS03/32-1/results_tiny_test.json')
# list4 = DataList.load(f'results/HA_neighbor/PEMS03/32/results_tiny_test.json')
for x in [3,4,7,8]:
    list1 = DataList.load(f'results/plain/PEMS0{x}/32-1/results_tiny_test.json')
    list2 = DataList.load(f'results/neighbor/PEMS0{x}/32-1/results_tiny_test.json')
    list3 = DataList.load(f'results/HA/PEMS0{x}/32-1/results_tiny_test.json')
    list4 = DataList.load(f'results/HA_neighbor/PEMS0{x}/32-1/results_tiny_test.json')
    lists = [list1, list2, list3, list4]
    names = ['Plain', 'Neighbor', 'HA', 'HA_Nei']
    if os.path.exists(f'results/HA_neighbor/PEMS0{x}/32-1r/results_tiny_test.json'):
        list5 = DataList.load(f'results/HA_neighbor/PEMS0{x}/32-1r/results_tiny_test.json')
        lists.append(list5)
        names.append('HA_Nei(T)')
    print(f'\nPEMS0{x}-32:')
    print(getDataResults(lists, names))

list1 = DataList.load('results/plain/PEMS03/16-3/results_tiny_test.json')
list2 = DataList.load('results/neighbor/PEMS03/16-3/results_tiny_test.json')
list3 = DataList.load('results/HA/PEMS03/16-3/results_tiny_test.json')
list4 = DataList.load('results/HA_neighbor/PEMS03/16-3/results_tiny_test.json')
lists = [list1, list2, list3, list4]
names = ['Plain', 'Neighbor', 'HA', 'HA_Nei']
print('\nPEMS03-16-3:')
print(getDataResults(lists, names))

# list1 = DataList.load('results/plain/PEMS04/32-1/results_tiny_test.json')
# list2 = DataList.load('results/neighbor/PEMS04/32-1/results_tiny_test.json')
# list3 = DataList.load('results/HA/PEMS04/32-1/results_tiny_test.json')
# list4 = DataList.load('results/HA_neighbor/PEMS04/32-1/results_tiny_test.json')
# lists = [list1, list2, list3, list4]
# names = ['Plain', 'Neighbor', 'HA', 'HA_Nei']
# print('\nPEMS04-32:')
# print(getDataResults(lists, names))