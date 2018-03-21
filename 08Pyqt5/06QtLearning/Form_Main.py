import pymssql
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QDialog, QHeaderView, QMessageBox
from gevent import event

from Form_Insert import insertDialog
from main import  Ui_Dialog


class MainDialog(QDialog,Ui_Dialog):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)
        result = self.DatabaseQuery()
        self.modelLoad(result)
        # 下面代码让表格100%填满窗口
        self.tableView.horizontalHeader().setStretchLastSection(True)
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def modelLoad(self,result):
        self.model = QStandardItemModel(self)   # 设置横坐标每项的属性名
        self.model.setHorizontalHeaderLabels(['UseID', 'PassWord']) # 配置数据，注意！！！需要使用QStandardItem格式的文本
        i = 0
        # print('tableShow:' + str(result))
        for r in result:
            self.model.setItem(i, 0, QStandardItem(str(r[0])))
            self.model.setItem(i, 1, QStandardItem(str(r[1])))
            i += 1
        self.tableView.setModel(self.model)

    def btnInsert(self):
        # 插入数据
        self.FormInsert = insertDialog()
        self.FormInsert.show()
        # 显示数据
        result = self.DatabaseQuery()
        self.modelLoad(result)
        pass

    def btnDelete(self):
        pass

    def btnUpdate(self):
        pass

    def btnQuery(self):
        conn = pymssql.connect(host='.', user='sa', password='sql', database='Test', charset="GBK")
        uid = self.lineEdit.text()
        sqlstr = 'select * from usertable where userid = \''+ str(uid) +'\''
        # print(sqlstr)
        try:
            cur = conn.cursor()
            cur.execute(sqlstr)
            result = cur.fetchall()
        except:
            print("Error: unable to fetch data")
        self.modelLoad(result)
        cur.close()
        conn.close()

    def btnClose(self):
        self.close()

    def DatabaseQuery(self): # 在数据库中查询全部数据
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