#8740C界面
import sys
from data import *
from solve import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QTextCursor

class MyWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('8740C')
        self.p1 = list(p1.keys())
        self.l1 = list(l1.keys())
        self.t1 = list(theta1.keys())
        self.init_ui()

    def init_ui(self):
        self.resize(1000,800)
        self.layout1 = QVBoxLayout()
        self.layout2 = QHBoxLayout()
        self.layout3 = QHBoxLayout()
        self.table_p1()
        self.table_l1()
        self.table_t1()
        self.textbrowser()
        self.button()
        self.layout1.addLayout(self.layout2)
        self.layout1.addLayout(self.layout3)
        self.layout2.addWidget(self.table_p1)
        self.layout2.addWidget(self.table_l1)
        self.layout2.addWidget(self.table_t1)
        self.layout3.addWidget(self.textbrowser1)
        self.layout3.addWidget(self.button1)
        self.setLayout(self.layout1)

    def table_p1(self):
        self.table_p1 = QTableWidget(len(self.p1),3)
        self.table_p1.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        p1_horizontalheaderitem = ['x','y','z']
        for i,hheaderitem in enumerate(p1_horizontalheaderitem):
            item = QTableWidgetItem()
            item.setText(hheaderitem)
            self.table_p1.setHorizontalHeaderItem(i,item)
        for i,vheaderitem in enumerate(self.p1):
            item = QTableWidgetItem()
            item.setText(vheaderitem)
            self.table_p1.setVerticalHeaderItem(i,item)
        for row,rowval in enumerate(self.p1):
            for col,val in enumerate(p1[rowval]):
                tableval = QTableWidgetItem(str(val))
                self.table_p1.setItem(row,col,tableval)
        return self.table_p1

    def table_l1(self):
        self.table_l1 = QTableWidget(len(self.l1),1)
        self.table_l1.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        l1_horizontalheaderitem = ['l']
        for i,hheaderitem in enumerate(l1_horizontalheaderitem):
            item = QTableWidgetItem()
            item.setText(hheaderitem)
            self.table_l1.setHorizontalHeaderItem(i,item)
        for i,vheaderitem in enumerate(self.l1):
            item = QTableWidgetItem()
            item.setText(vheaderitem)
            self.table_l1.setVerticalHeaderItem(i,item)
        for row,val in enumerate(self.l1):
            tableval = QTableWidgetItem(str(l1[val]))
            self.table_l1.setItem(row,0,tableval)
        return self.table_l1

    def table_t1(self):
        self.table_t1 = QTableWidget(len(self.t1),1)
        self.table_t1.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        l1_horizontalheaderitem = ['theta']
        for i,hheaderitem in enumerate(l1_horizontalheaderitem):
            item = QTableWidgetItem()
            item.setText(hheaderitem)
            self.table_t1.setHorizontalHeaderItem(i,item)
        for i,vheaderitem in enumerate(self.t1):
            item = QTableWidgetItem()
            item.setText(vheaderitem)
            self.table_t1.setVerticalHeaderItem(i,item)
        for row,val in enumerate(self.t1):
            tableval = QTableWidgetItem(str(theta1[val]*du))
            self.table_t1.setItem(row,0,tableval)
        return self.table_t1

    def textbrowser(self):
        self.textbrowser1 = QTextBrowser()
        self.textbrowser1.setFixedSize(800,200)

    def button(self):
        self.button1 = QPushButton('运行')
        self.button1.clicked.connect(self.solve)

    def solve(self):
        for row,val in enumerate(self.p1):
            for col in range(3):
                p1[val][col] = float(self.table_p1.item(row,col).text())
        for i in range(len(self.l1)):
            l1[self.l1[i]] = float(self.table_l1.item(i,0).text())
        for i in range(len(self.t1)):
            theta1[self.t1[i]] = float(self.table_t1.item(i,0).text())*hd
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        solve(p1,l1,theta1)
        sys.stdout = sys.__stdout__

    def normalOutputWritten(self, text):
        cursor = self.textbrowser1.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.textbrowser1.setTextCursor(cursor)
        self.textbrowser1.ensureCursorVisible()

class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w=MyWindow()
    w.show()
    sys.exit(app.exec())