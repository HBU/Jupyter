import pymssql
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QDialog

from main import Ui_Dialog_Main


class MainDialog(QDialog,Ui_Dialog_Main):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)
        # 设置一个4*4的表格数据模型
        self.model = QStandardItemModel(0, 0)
        # 设置横坐标每项的属性名
        self.model.setHorizontalHeaderLabels(['UseID', 'PassWord'])
        # 配置数据，注意！！！需要使用QStandardItem格式的文本
        Data = self.DatabaseQuery()
        print(str(Data))
        i = 0
        for r in Data:
            self.model.setItem(i, 0, QStandardItem(str(r[0])))
            self.model.setItem(i, 1, QStandardItem(str(r[1])))
            i += 1
        self.tableView.setModel(self.model)
    def btnInsert(self):
        pass
    def btnDelete(self):
        pass
    def btnUpdate(self):
        pass
    def btnQuery(self):
        pass
    def btnClose(self):
        self.close()

    def DatabaseQuery(self): # 在数据库中查询
        print("start DatabaseQuery:")
        conn = pymssql.connect(host='.', user='sa', password='sql', database='Test', charset="GBK")
        print("Database connected!")
        sqlstr = 'select * from usertable'
        try:
            cur = conn.cursor()
            cur.execute(sqlstr)
            result = cur.fetchall()
        except:
            print("Error: unable to fetch data")
        finally:
            print("End of SQL")
        cur.close()
        conn.close()
        return result