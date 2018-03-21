import sys
from PyQt5 import QtWidgets
from UI import Ui_MainWindow
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class myform(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(myform, self).__init__()
        self.setupUi(self)
        #设置一个4*4的表格数据模型
        self.model = QStandardItemModel(4, 4)
        #设置横坐标每项的属性名
        self.model.setHorizontalHeaderLabels(['id', '分组码', '东经', '北纬'])
        #配置数据，注意！！！需要使用QStandardItem格式的文本
        self.model.setItem(0, 0, QStandardItem('186'))
        self.model.setItem(0, 1, QStandardItem('0000'))
        self.model.setItem(0, 2, QStandardItem('108.1721'))
        self.model.setItem(0, 3, QStandardItem('20.1231'))
        self.model.setItem(1, 0, QStandardItem('189'))
        self.model.setItem(1, 1, QStandardItem('1001'))
        self.model.setItem(1, 2, QStandardItem('108.1721'))
        self.model.setItem(1, 3, QStandardItem('20.1231'))
        self.model.setItem(2, 0, QStandardItem('175'))
        self.model.setItem(2, 1, QStandardItem('1002'))
        self.model.setItem(2, 2, QStandardItem('108.1721'))
        self.model.setItem(2, 3, QStandardItem('20.1231'))
        self.model.setItem(3, 0, QStandardItem('152'))
        self.model.setItem(3, 1, QStandardItem('1003'))
        self.model.setItem(3, 2, QStandardItem('108.1721'))
        self.model.setItem(3, 3, QStandardItem('20.1231'))
        self.tableView.setModel(self.model)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = myform()
    ex.show()
    sys.exit(app.exec_())

# 作者：Symbian米汤
# 链接：https://www.jianshu.com/p/7812da75db13
# 來源：简书
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。