# -*-coding:utf-8-*-
import sys
from myFunction import myFunction
from PyQt5 import QtWidgets

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myWindow = myFunction()
    myWindow.show()
    sys.exit(app.exec_())
