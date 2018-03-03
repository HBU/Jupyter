# NumPy 是一种用于进行科学计算的常用工具包。pandas Series 可用作大多数 NumPy 函数的参数：
import pandas as pd
import numpy as np

city_names = pd.Series(['Beijing', 'Tianjin', 'Baoding'])
population = pd.Series([852469, 1015785, 485199])
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print('------------np.log(population))---------------')
print(np.log(population))
print('------------population.apply(lambda val: val > 1000000)---------------')
#对于更复杂的单列转换，您可以使用 Series.apply。像 Python 映射函数一样，Series.apply 将以参数形式接受 lambda 函数，而该函数会应用于每个值。
str = population.apply(lambda val: val > 1000000)
print(str)
print('--------向现有 DataFrame 添加了两个 Series-------------------')
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
print(cities)
print('---------------------------')
# 通过添加一个新的布尔值列（当且仅当以下两项均为 True 时为 True）修改 cities 表格：
# 城市以圣人命名。
# 城市面积大于 50 平方英里。
cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['City name'].apply(lambda name: name.startswith('San'))
print(cities)
print('---------------------------')
print(city_names.index)
print(cities.index)
print('---------------------------')
print(cities.reindex([2, 0, 1]))
print(cities.reindex(np.random.permutation(cities.index)))
print(cities.reindex([0, 4, 5, 2]))
# reindex输入数组包含原始DataFrame索引值中没有的值，reindex会为此类“丢失的”索引添加新行，并在所有对应列中填充NaN值