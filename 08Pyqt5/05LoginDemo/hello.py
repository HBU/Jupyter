from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import sys

class hello_mainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(hello_mainWindow,self).__init__()
        self.initUI()
    def initUI(self):
        self.setWindowTitle("PushButton")
        self.setGeometry(400,400,300,260)

        self.closeButton = QPushButton(self)
        self.closeButton.setText("Close")          #text
        self.closeButton.setIcon(QIcon("close.png")) #icon
        self.closeButton.setShortcut('Ctrl+D')  #shortcut key
        self.closeButton.clicked.connect(self.close)
        self.closeButton.setToolTip("Close the widget") #Tool tip
        self.closeButton.move(100,100)
    # def __init__(self):
    #     super(hello_mainWindow,self).__init__()
    #     self.setupUi(self)
    #     self.retranslateUi(self)
    #
    # def setupUi(self, mainWindow):
    #     mainWindow.setObjectName("DataWindow")
    #     mainWindow.setWindowModality(QtCore.Qt.WindowModal)
    #     mainWindow.resize(624, 511)
    #     self.retranslateUi(mainWindow)
    #     QtCore.QMetaObject.connectSlotsByName(mainWindow)
    #
    # def retranslateUi(self, mainWindow):
    #     _translate = QtCore.QCoreApplication.translate
    #     mainWindow.setWindowTitle(_translate("DataWindow", "hello world"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = hello_mainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())

