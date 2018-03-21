import sys
from PyQt5 import QtWidgets

from PyQt5.QtSql import QSqlDatabase, QSqlQuery
from PyQt5.QtWidgets import QApplication, QMessageBox

from main import Ui_MainWindow


class MyWindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MyWindow,self).__init__()
        self.setupUi(self)

    def btnShow(self):
        print ('btnShow')
        db = QSqlDatabase.addDatabase('QODBC')
        # TODO: Add support for trusted connections.
        # ("Driver={SQLServer};Server=Your_Server_Name;Database=Your_Database_Name;Trusted_Connection=yes;")
        db.setDatabaseName('DRIVER={SQL Server};SERVER=%s;DATABASE=%s;UID=%s;PWD=%s;' % ('.', 'Test', 'sa', 'sql'))
        # 判断是否打开
        if not db.open():
            QMessageBox.critical(None, ("Cannot open database"),
                                 ("Unable to establish a database connection. \n"
                                  "This example needs SQLite support. Please read "
                                  "the Qt SQL driver documentation for information "
                                  "how to build it.\n\n"
                                  "Click Cancel to exit."),
                                 QMessageBox.Cancel)
            return False
        print('btnShow2')
        # 声明数据库查询对象
        query = QSqlQuery()
        # 创建表
        query.exec_("create table student(id int primary key, name vchar, sex vchar, age int, deparment vchar)")
        # 添加记录
        query.exec_("insert into student values(1,'张三1','男',20,'计算机')")
        query.exec_("insert into student values(2,'李四1','男',19,'经管')")
        query.exec_("insert into student values(3,'王五1','男',22,'机械')")

        # 关闭数据库
        db.close()

        return True


    def queryDB(self):
        db = QSqlDatabase.addDatabase('QODBC')
        # TODO: Add support for trusted connections.
        # ("Driver={SQLServer};Server=Your_Server_Name;Database=Your_Database_Name;Trusted_Connection=yes;")
        db.setDatabaseName('DRIVER={SQL Server};SERVER=%s;DATABASE=%s;UID=%s;PWD=%s;' % ('.', 'Test', 'sa', 'sql'))

        if not db.open():
            print('db.open failed')
            QMessageBox.critical(None, ("无法打开数据库"),
                                 ("无法建立到数据库的连接,这个例子需要SQLite 支持，请检查数据库配置。\n\n"
                                  "点击取消按钮退出应用。"),
                                 QMessageBox.Cancel)
            return False

        query = QSqlQuery()
        # query.exec_("create table people(id int primary key, name varchar(20), address varchar(30))")
        query.exec_("select * from people")
        print(str(query.result()))
        db.close()
        return True

    def createDB(self):
        db = QSqlDatabase.addDatabase('QODBC')
        # TODO: Add support for trusted connections.
        # ("Driver={SQLServer};Server=Your_Server_Name;Database=Your_Database_Name;Trusted_Connection=yes;")
        db.setDatabaseName('DRIVER={SQL Server};SERVER=%s;DATABASE=%s;UID=%s;PWD=%s;' % ('.', 'Test', 'sa', 'sql'))

        if not db.open():
            print('db.open failed')
            QMessageBox.critical(None, ("无法打开数据库"),
                                 ("无法建立到数据库的连接,这个例子需要SQLite 支持，请检查数据库配置。\n\n"
                                  "点击取消按钮退出应用。"),
                                 QMessageBox.Cancel)
            return False

        print("db.open success ")
        query = QSqlQuery()
        query.exec_("create table people(id int primary key, name varchar(20), address varchar(30))")
        query.exec_("insert into people values(1, 'zhangsan1', 'BeiJing')")
        query.exec_("insert into people values(2, 'lisi1', 'TianJing')")
        query.exec_("insert into people values(3, 'wangwu1', 'HenNan')")
        query.exec_("insert into people values(4, 'lisi2', 'HeBei')")
        query.exec_("insert into people values(5, 'wangwu2', 'shanghai')")
        db.close()
        return True

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())