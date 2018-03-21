from PyQt5.QtWidgets import QDialog
from PyQt5.uic.properties import QtWidgets
from AA import Ui_Dialog2

class myDlg2(QDialog,Ui_Dialog2):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)
    def CClick(self):
        date=self.calendarWidget.selectedDate()
        self.label.setText(str(date.toPyDate()))

