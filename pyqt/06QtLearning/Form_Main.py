from PyQt5.QtWidgets import QDialog

from main import Ui_Dialog_Main


class MainDialog(QDialog,Ui_Dialog_Main):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)
    def btnClose(self):
        self.close()