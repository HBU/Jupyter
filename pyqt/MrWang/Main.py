from PyQt5.QtWidgets import QDialog
from PyQt5.uic.properties import QtWidgets

from CDlg import myDlg2
from a1 import *

class myDlg(QDialog,Ui_Dialog):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)
    def btn1_clicked(self):
        self.md=myDlg2()
        self.md.show()
        pass
    def btn2_clicked(self):
        self.label1.setText("hahaha")

if __name__=="__main__":
    import sys
    app=QtWidgets.QApplication(sys.argv)
    ui=myDlg()
    ui.show()
    sys.exit(app.exec_())