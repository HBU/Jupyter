# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'formMain.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Main(object):
    def setupUi(self, Main):
        Main.setObjectName("Main")
        Main.setWindowModality(QtCore.Qt.NonModal)
        Main.resize(603, 457)
        self.tableView = QtWidgets.QTableView(Main)
        self.tableView.setGeometry(QtCore.QRect(50, 20, 391, 391))
        self.tableView.setObjectName("tableView")
        self.pushButton = QtWidgets.QPushButton(Main)
        self.pushButton.setGeometry(QtCore.QRect(490, 60, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(Main)
        self.label.setGeometry(QtCore.QRect(160, 420, 231, 16))
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(Main)
        self.lineEdit.setGeometry(QtCore.QRect(490, 260, 71, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton_2 = QtWidgets.QPushButton(Main)
        self.pushButton_2.setGeometry(QtCore.QRect(490, 130, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Main)
        self.pushButton_3.setGeometry(QtCore.QRect(490, 210, 75, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(Main)
        self.pushButton_4.setGeometry(QtCore.QRect(490, 170, 75, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(Main)
        self.pushButton_5.setGeometry(QtCore.QRect(490, 390, 75, 23))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(Main)
        self.pushButton_6.setGeometry(QtCore.QRect(490, 300, 75, 23))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(Main)
        self.pushButton_7.setGeometry(QtCore.QRect(490, 20, 75, 23))
        self.pushButton_7.setObjectName("pushButton_7")

        self.retranslateUi(Main)
        self.pushButton.clicked.connect(Main.btnAll)
        self.pushButton_2.clicked.connect(Main.btnInsert)
        self.pushButton_4.clicked.connect(Main.btnDelete)
        self.pushButton_3.clicked.connect(Main.btnUpdate)
        self.pushButton_6.clicked.connect(Main.btnQuery)
        self.pushButton_5.clicked.connect(Main.btnClose)
        self.pushButton_7.clicked.connect(Main.btnCreate)
        QtCore.QMetaObject.connectSlotsByName(Main)

    def retranslateUi(self, Main):
        _translate = QtCore.QCoreApplication.translate
        Main.setWindowTitle(_translate("Main", "SQLserver基本操作"))
        self.pushButton.setText(_translate("Main", "全部数据"))
        self.label.setText(_translate("Main", "Design by David 2018.3.21"))
        self.pushButton_2.setText(_translate("Main", "增加"))
        self.pushButton_3.setText(_translate("Main", "修改"))
        self.pushButton_4.setText(_translate("Main", "删除"))
        self.pushButton_5.setText(_translate("Main", "关闭"))
        self.pushButton_6.setText(_translate("Main", "查询"))
        self.pushButton_7.setText(_translate("Main", "建表"))

