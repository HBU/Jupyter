# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AA.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog2(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(545, 381)
        self.calendarWidget = QtWidgets.QCalendarWidget(Dialog)
        self.calendarWidget.setGeometry(QtCore.QRect(20, 10, 248, 197))
        self.calendarWidget.setObjectName("calendarWidget")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(323, 60, 181, 20))
        self.label.setObjectName("label")

        self.retranslateUi(Dialog)
        self.calendarWidget.clicked['QDate'].connect(Dialog.CClick)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "TextLabel"))

