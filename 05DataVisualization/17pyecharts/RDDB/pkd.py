import requests
import re
import time
from lxml import html
from selenium import webdriver
from pyecharts import Map
import pandas as pd
import re

r = requests.get(url='http://www.sohu.com/a/223885358_118392')    # 最基本的GET请求
# time.sleep(10)
html = r.text
content = re.findall(u'<p>([\u2E80-\u9FFF]{2,3})[u"市"u"省"u"特"u"壮"u"回"u"维"u"自"u"人"].{0,6}\uff08(\d*)\u540d\uff09 </p>.*?<p>(.*?)</p>', html, re.S)
import pandas as pd
# 取地区，姓名，性别，族
data = [(_[0],_[2]) for _ in content]
print(data[0])

def get_locality_name_sex_race(param):
    locality = param[0]
    # '、'分割姓名
    for el in param[1].split(u'、'):
        temp = re.split(u'\uff08|\uff0c|\uff09', el)
        length = len(temp)
        if length == 1:
            temp += [u'男', u'汉族']
        elif length == 3:
            if len(temp[1]) == 1:
                temp[2] = u'汉族'
            else:
                temp[2] = temp[1]
                temp[1] = u'男'
        elif length == 4:
            temp.pop()
            if u'族' not in temp[2]:
                temp[2] = u'汉族'

        df.loc[df.shape[0]] = [locality]+temp  #df.shape[0]，df.shape[1]分别获取行数、列数,loc——通过行标签索引行数据

df = pd.DataFrame(columns=('locality', 'name', 'sex', 'race'))#生成空的pandas表
# mystr = map(get_locality_name_sex_race, data)
# df.add(str(map(get_locality_name_sex_race, data)))
# time.sleep(20)
# print(list(mystr))
# print('=====================')
# df.add(str(mystr))
# # df.add(str(mystr))
list(map(get_locality_name_sex_race, data))
print(df)
print('=====================')
# map() 会根据提供的函数对指定序列做映射。
# 第一个参数 function 以参数序列中的每一个元素调用 function 函数，
# 返回包含每次 function 函数返回值的新列表。
# df.add(str(test))
# print(df)
def get_attr_sex_v(race_groupby_sex):
    attr_sex = race_groupby_sex.count().index.values.tolist()
    attr_v = race_groupby_sex.count().values[0:2,0].tolist()
    return attr_sex,attr_v

from pyecharts import Bar
race_groupby_sex = df[df.race==u'汉族'].groupby(df['sex'])
attr_sex,attr_v = get_attr_sex_v(race_groupby_sex)
bar = Bar("", "", width=600, height=400)
bar.add("汉族", attr_sex, attr_v, is_more_utils=True)
race_groupby_sex = df[df.race!=u'汉族'].groupby(df['sex'])
minorities_sex,minorities_v = get_attr_sex_v(race_groupby_sex)
bar.add("少数民族", minorities_sex, minorities_v, is_more_utils=True)
bar.render()
bar