from PyQt5 import QtCore, QtGui, QtWidgets

class hello_mainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(hello_mainWindow,self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)

    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.setWindowModality(QtCore.Qt.WindowModal)
        mainWindow.resize(624, 511)
        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "hello world"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = hello_mainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())

