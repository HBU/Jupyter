#https://colab.research.google.com/notebooks/mlcc/intro_to_pandas.ipynb?hl=zh-cn#scrollTo=oa5wfZT7VHJl
import pandas as pd
import matplotlib.pyplot as plt
version = pd.__version__
print(version)
print('---------------------------')
# pandas 中的主要数据结构被实现为以下两类：
# DataFrame，您可以将它想象成一个关系型数据表格，其中包含多个行和已命名的列。
# Series，它是单一列。DataFrame 中包含一个或多个 Series，每个 Series 均有一个名称。

# 创建 Series 的一种方法是构建 Series 对象:
DemoPd = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
print(DemoPd.head())
print('---------------------------')
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
DemoPd = pd.DataFrame({ 'CityName': city_names, 'Population': population })
print(DemoPd.head())
print('---------------------------')
#但是在大多数情况下，您需要将整个文件加载到 DataFrame 中。 下面的示例加载了一个包含加利福尼亚州住房数据的文件。
#california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv", sep=",")
california_housing_dataframe = pd.read_csv("california_housing_train.csv", sep=",")
print(california_housing_dataframe.describe())#DataFrame.describe 来显示关于 DataFrame 的有趣统计信息
print('---------------------------')
print(california_housing_dataframe.head())#DataFrame.head，它显示 DataFrame 的前几个记录
print('---------------------------')
print(california_housing_dataframe.hist('housing_median_age'))#DataFrame.hist，您可以快速了解一个列中值的分布
plt.show()