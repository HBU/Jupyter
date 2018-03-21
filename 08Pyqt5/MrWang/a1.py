# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '1.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(571, 431)
        self.btn1 = QtWidgets.QPushButton(Dialog)
        self.btn1.setGeometry(QtCore.QRect(400, 340, 75, 23))
        self.btn1.setObjectName("btn1")
        self.btn2 = QtWidgets.QPushButton(Dialog)
        self.btn2.setGeometry(QtCore.QRect(100, 340, 75, 23))
        self.btn2.setObjectName("btn2")
        self.label1 = QtWidgets.QLabel(Dialog)
        self.label1.setGeometry(QtCore.QRect(230, 80, 54, 12))
        self.label1.setObjectName("label1")

        self.retranslateUi(Dialog)
        self.btn1.clicked.connect(Dialog.btn1_clicked)
        self.btn2.clicked.connect(Dialog.btn2_clicked)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.btn1.setText(_translate("Dialog", "PushButton"))
        self.btn2.setText(_translate("Dialog", "PushButton"))
        self.label1.setText(_translate("Dialog", "TextLabel"))

