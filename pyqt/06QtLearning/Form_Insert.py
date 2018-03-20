from PyQt5.QtWidgets import QDialog

from insert import Ui_Dialog


class insertDialog(QDialog,Ui_Dialog):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)

    def insertDlg(self):
        # 把新数据 insert 到数据库：打开数据库，insert语句插入，关闭数据库
        self.close()
        pass