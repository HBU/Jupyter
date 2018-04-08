import requests
import re
import time
from lxml import html
from selenium import webdriver
from pyecharts import Map, Bar, Pie

r = requests.get(url='http://www.sohu.com/a/223885358_118392')    # 最基本的GET请求
time.sleep(60)
html = r.text
content = re.findall(u'<p>([\u2E80-\u9FFF]{2,3})[u"市"u"省"u"特"u"壮"u"回"u"维"u"自"u"人"].{0,6}\uff08(\d*)\u540d\uff09 </p>.*?<p>(.*?)</p>', html, re.S)
data = [_[0:2] for _ in content]
print(data)
map = Map("人大代表分布", title_pos="center",width=800, height=500)
attr, value =map.cast(data)
attr[-1] = u"南海诸岛"
map.add("", attr, value, maptype='china',
        is_label_show=True,label_pos="inside",label_text_color="#000",
        is_visualmap=True, visual_text_color='#000', visual_range=[12, 172],visual_range_text=['低','高'],
       visual_pos = [500,500])
# map.show_config()
map.render()
map