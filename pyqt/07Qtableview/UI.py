from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 537)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 611, 501))
        self.tabWidget.setStyleSheet("")
        self.tabWidget.setObjectName("tabWidget")
        self.tab_1 = QtWidgets.QWidget()
        self.tab_1.setStyleSheet("background:rgb(155, 122, 200)\n"
"")
        self.tab_1.setObjectName("tab_1")
        self.tableView = QtWidgets.QTableView(self.tab_1)
        self.tableView.setGeometry(QtCore.QRect(10, 10, 521, 301))
        self.tableView.setObjectName("tableView")
        self.tabWidget.addTab(self.tab_1, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setStyleSheet("background:orange")
        self.tab.setObjectName("tab")
        self.tabWidget.addTab(self.tab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 640, 23))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "对讲模块"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), _translate("MainWindow", "设备列表"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "分组码列表"))
        self.menu.setTitle(_translate("MainWindow", "设备查看"))
        self.menu_2.setTitle(_translate("MainWindow", "分组查看"))
        self.menu_3.setTitle(_translate("MainWindow", "主板信息查看"))

# 作者：Symbian米汤
# 链接：https://www.jianshu.com/p/7812da75db13
# 來源：简书
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。