from PyQt5.QtCore import Qt
from PyQt5.QtSql import QSqlQuery, QSqlDatabase, QSqlTableModel
from PyQt5.QtWidgets import QDialog, QMessageBox, QHeaderView
from qtpy import QtWidgets

from formMain import Ui_Main


class mainForm(QDialog,Ui_Main):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)
        # 连接SQLserver数据库
        self.db = QSqlDatabase.addDatabase('QODBC')
        self.db.setDatabaseName('DRIVER={SQL Server};SERVER=%s;DATABASE=%s;UID=%s;PWD=%s;' % ('.', 'Test', 'sa', 'sql'))
        self.model = QSqlTableModel()
        self.model.setTable('student')  # 载入student表
        self.model.setHeaderData(0, Qt.Horizontal, "学号")
        self.model.setHeaderData(1, Qt.Horizontal, "姓名")
        self.tableView.horizontalHeader().setStretchLastSection(True)# 表格宽度充满父控件
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.query = QSqlQuery()
        self.btnAll()

    def btnCreate(self):
        if not self.db.open():
            print('db.open failed')
            QMessageBox.critical(None, ("无法打开数据库"),("无法建立到数据库的连接,请检查数据库配置。\n\n"
                                  "点击取消按钮退出应用。"), QMessageBox.Cancel)
            return False

        self.query.exec_("DROP TABLE  IF EXISTS student")
        self.query.exec_("create table student(sno varchar(10), sname varchar(10))")
        self.query.exec_("insert into student values('101', 'Tom')")
        self.query.exec_("insert into student values('102', 'Jerry')")
        self.query.exec_("insert into student values('103', 'Mike')")
        self.query.exec_("insert into student values('104', 'Jane')")
        self.query.exec_("insert into student values('105', 'David')")
        self.db.close()
        return True

    def btnAll(self):
        self.model.select()
        self.tableView.setModel(self.model)

    def btnInsert(self):
        self.model.insertRows(self.model.rowCount(), 1)

    def btnDelete(self):
        self.model.removeRow(self.tableView.currentIndex().row())

    def btnUpdate(self):
        QMessageBox.information(None, ("提示"), ("单击左侧需要修改的单元格。\n\n""在单元格中修改即可。"), QMessageBox.Close)

    def btnQuery(self):
        studentid = self.lineEdit.text()
        self.query.exec_("select * from student where sno = '%s'" %(studentid))
        self.model.setQuery(self.query)
        self.tableView.setModel(self.model)

    def btnClose(self):
        self.close()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    myForm = mainForm()
    myForm.show()
    sys.exit(app.exec_())