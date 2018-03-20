import sys
from PyQt5.QtSql import QSqlDatabase, QSqlQuery
from PyQt5.QtWidgets import QApplication, QMessageBox


def createDB():
    db = QSqlDatabase.addDatabase('QODBC')
    #TODO: Add support for trusted connections.
    #("Driver={SQLServer};Server=Your_Server_Name;Database=Your_Database_Name;Trusted_Connection=yes;")
    db.setDatabaseName('DRIVER={SQL Server};SERVER=%s;DATABASE=%s;UID=%s;PWD=%s;'% ('.','Test','sa', 'sql'))

    if not db.open():
        print('db.open failed')
        QMessageBox.critical(None, ("无法打开数据库"),
                             ("无法建立到数据库的连接,这个例子需要SQLite 支持，请检查数据库配置。\n\n"
                              "点击取消按钮退出应用。"),
                             QMessageBox.Cancel)
        return False

    print("db.open success ")
    query = QSqlQuery()
    query.exec_("create table people1(id int primary key, name varchar(20), address varchar(30))")
    query.exec_("insert into people1 values(1, 'zhangsan1', 'BeiJing')")
    query.exec_("insert into people1 values(2, 'lisi1', 'TianJing')")
    query.exec_("insert into people1 values(3, 'wangwu1', 'HenNan')")
    query.exec_("insert into people1 values(4, 'lisi2', 'HeBei')")
    query.exec_("insert into people1 values(5, 'wangwu2', 'shanghai')")
    db.close()
    return True

if __name__ == '__main__':
    app = QApplication(sys.argv)
    createDB()
    sys.exit(app.exec_())