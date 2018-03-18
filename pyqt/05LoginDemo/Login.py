from PyQt5.QtGui import QIcon
from hello import *
from PyQt5.QtWidgets import *
import pymssql

class Ui_MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(Ui_MainWindow,self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(386, 127)
        MainWindow.setWindowIcon(QIcon('logo.png'))
        MainWindow.setStyleSheet("background-image:url(Background.jpg)")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.lineEdit = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit.setGeometry(QtCore.QRect(250, 20, 100, 20))
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(250, 50, 100, 20))
        self.lineEdit_2.setText("")
        self.lineEdit_2.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label = QtWidgets.QLabel(self.centralWidget)
        self.label.setGeometry(QtCore.QRect(200, 24, 24, 12))
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralWidget)
        self.label_2.setGeometry(QtCore.QRect(200, 54, 24, 12))
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton.setGeometry(QtCore.QRect(190, 90, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_2.setGeometry(QtCore.QRect(290, 90, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        MainWindow.setCentralWidget(self.centralWidget)

        self.pushButton.clicked.connect(self.word_get)
        self.pushButton_2.clicked.connect(MainWindow.close)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "登录"))
        self.lineEdit.setPlaceholderText(_translate("MainWindow", "请输入帐号"))
        self.lineEdit_2.setPlaceholderText(_translate("MainWindow", "请输入密码"))
        self.label.setText(_translate("MainWindow", "帐号"))
        self.label_2.setText(_translate("MainWindow", "密码"))
        self.pushButton.setText(_translate("MainWindow", "确定"))
        self.pushButton_2.setText(_translate("MainWindow", "取消"))

    def word_get(self):
        login_user = self.lineEdit.text()
        login_password = self.lineEdit_2.text()
        print(login_user)
        print(login_password)
        print("start:")
        conn = pymssql.connect(host='.', user='sa', password='sql', database='Test', charset="GBK")
        print("connected!")
        cur = conn.cursor()
        sqlstr = 'select count(*) from usertable where userid = \''+ str(login_user) +'\' and password = \''+ str(login_password) +'\''
        try:
            print(sqlstr)
            cur.execute(sqlstr)
            count = cur.fetchall()
            print(count[0][0])
        except:
            print("Error: unable to fetch data")
        finally:
            print("End of SQL")
        cur.close()
        conn.close()

        if count[0][0] == 1:
            ui_hello.show()
            MainWindow.close()
        else:
            QMessageBox.warning(self,
                    "警告",
                    "用户名或密码错误！",
                    QMessageBox.Yes)
            self.lineEdit.setFocus()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui_hello = hello_mainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

# 作者：一颗数据小白菜
# 链接：https://www.jianshu.com/p/d18ff36a78d6
# 來源：简书
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。