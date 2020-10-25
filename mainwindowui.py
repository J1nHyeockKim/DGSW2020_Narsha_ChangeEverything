# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1000)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.originalVideo = QtWidgets.QLabel(self.centralwidget)
        self.originalVideo.setGeometry(QtCore.QRect(0, 270, 960, 540))
        self.originalVideo.setText("")
        self.originalVideo.setAlignment(QtCore.Qt.AlignCenter)
        self.originalVideo.setObjectName("originalVideo")
        self.changedVideo = QtWidgets.QLabel(self.centralwidget)
        self.changedVideo.setGeometry(QtCore.QRect(960, 270, 960, 540))
        self.changedVideo.setText("")
        self.changedVideo.setAlignment(QtCore.Qt.AlignCenter)
        self.changedVideo.setObjectName("changedVideo")
        self.changeButton = QtWidgets.QPushButton(self.centralwidget)
        self.changeButton.setGeometry(QtCore.QRect(1520, 80, 390, 50))
        self.changeButton.setObjectName("changeButton")
        self.text = QtWidgets.QLabel(self.centralwidget)
        self.text.setGeometry(QtCore.QRect(10, 220, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.text.setFont(font)
        self.text.setObjectName("text")
        self.text2 = QtWidgets.QLabel(self.centralwidget)
        self.text2.setGeometry(QtCore.QRect(960, 220, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.text2.setFont(font)
        self.text2.setObjectName("text2")
        self.saveButton = QtWidgets.QPushButton(self.centralwidget)
        self.saveButton.setGeometry(QtCore.QRect(1520, 20, 190, 50))
        self.saveButton.setObjectName("saveButton")
        self.loadButton = QtWidgets.QPushButton(self.centralwidget)
        self.loadButton.setGeometry(QtCore.QRect(1720, 20, 190, 50))
        self.loadButton.setObjectName("loadButton")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.changeButton.setText(_translate("MainWindow", "필터 넣기"))
        self.text.setText(_translate("MainWindow", "원본 영상"))
        self.text2.setText(_translate("MainWindow", "바뀐 영상"))
        self.saveButton.setText(_translate("MainWindow", "저장하기"))
        self.loadButton.setText(_translate("MainWindow", "불러오기"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

