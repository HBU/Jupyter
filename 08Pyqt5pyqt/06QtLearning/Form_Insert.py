from PyQt5.QtWidgets import QDialog
import pymssql
from insert import Ui_Dialog


class insertDialog(QDialog,Ui_Dialog):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)

    def insertDlg(self):
        pass