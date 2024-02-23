import sys
import numpy as np
import matplotlib.pyplot as plt
from k6_data import *
from k6_solve import solve
from k6_plot import k6_plot
from function import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QTextCursor
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC

k6_data(p,l,theta)
n = np.linspace(0,359*hd,360).reshape(-1,1)
drawing_times = 0

class Plot_Thread(QThread):
	def __init__(self, canvas, fig, ax1, ax2):
		super(Plot_Thread, self).__init__()
		self.canvas = canvas
		self.fig = fig
		self.ax1 = ax1
		self.ax2 = ax2

	def run(self):
		k6_plot(self,p,l,theta,n)

class MyWindow(QWidget):
    def __init__(self):
        super(MyWindow,self).__init__()
        self.setWindowTitle('K6挑线')
        self.point = list(p.keys())
        self.length = list(l.keys())
        self.theta = list(theta.keys())
        self.init_ui()

    def init_ui(self):
        self.resize(1600,800)
        self.layout1 = QHBoxLayout()
        self.layout2 = QVBoxLayout()
        self.setTableData()
        self.setCanvas()
        self.setButton()
        self.layout1.addLayout(self.layout2)
        self.layout1.addWidget(self.canvas)
        self.layout2.addWidget(self.table_data)
        self.layout2.addWidget(self.button1)
        self.layout2.addWidget(self.button2)
        self.setLayout(self.layout1)
        
#        self.plot_thread = Plot_Thread(self.canvas, self.fig, self.ax1, self.ax2)

    def setTableData(self):
        self.table_data = QTableWidget(len(self.point)+len(self.length)+len(self.theta)+2,3)
        self.table_data.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        p_horizontalheaderitem = ['x','y','z']
        for i,hheaderitem in enumerate(p_horizontalheaderitem):
            item = QTableWidgetItem()
            item.setText(hheaderitem)
            self.table_data.setHorizontalHeaderItem(i,item)
        for row_point,vheaderitem in enumerate(self.point):
            item = QTableWidgetItem()
            item.setText(vheaderitem)
            self.table_data.setVerticalHeaderItem(row_point,item)
        for row_point,rowval in enumerate(self.point):
            for col,val in enumerate(p[rowval]):
                tableval = QTableWidgetItem(str(val))
                self.table_data.setItem(row_point,col,tableval)

        item = QTableWidgetItem()
        item.setText('')
        self.table_data.setVerticalHeaderItem(row_point+1,item)
        tableval = QTableWidgetItem('l')
        self.table_data.setItem(row_point+1,0,tableval)
        for row_length,vheaderitem in enumerate(self.length):
            item = QTableWidgetItem()
            item.setText(vheaderitem)
            self.table_data.setVerticalHeaderItem(row_length+row_point+2,item)
        for row_length,val in enumerate(self.length):
            tableval = QTableWidgetItem(str(l[val]))
            self.table_data.setItem(row_length+row_point+2,0,tableval)

        item = QTableWidgetItem()
        item.setText('')
        self.table_data.setVerticalHeaderItem(row_length+row_point+3,item)
        tableval = QTableWidgetItem('theta')
        self.table_data.setItem(row_length+row_point+3,0,tableval)
        for row_theta,vheaderitem in enumerate(self.theta):
            item = QTableWidgetItem()
            item.setText(vheaderitem)
            self.table_data.setVerticalHeaderItem(row_length+row_point+row_theta+4,item)
        for row_theta,val in enumerate(self.theta):
            tableval = QTableWidgetItem(str(theta[val]*du))
            self.table_data.setItem(row_length+row_point+row_theta+4,0,tableval)
        
        return self.table_data

    def setCanvas(self):
        self.fig = plt.Figure()
        self.canvas = FC(self.fig)
        self.canvas.setFixedSize(1100,700)
        self.ax1 = self.fig.add_subplot(1,2,1,projection="3d")
        self.ax2 = self.fig.add_subplot(1,2,2)

    def setButton(self):
        self.button1 = QPushButton('更新')
        self.button1.clicked.connect(self.solve)
        self.button2 = QPushButton('清空画布')
        self.button2.clicked.connect(self.clearAxes)

    def solve(self):
        global drawing_times
        k6_data(p,l,theta)
        self.refreshData()
        solve(p,l,theta,n)
#        self.plot_thread.start()
        k6_plot(self,p,l,theta,n,drawing_times)
        drawing_times += 1
    
    def refreshData(self):
        for row_point,val in enumerate(self.point):
            for col in range(3):
                p[val][col] = float(self.table_data.item(row_point,col).text())
        for row_length in range(len(self.length)):
            l[self.length[row_length]] = float(self.table_data.item(row_point+row_length+2,0).text())
        for row_theta in range(len(self.theta)):
            theta[self.theta[row_theta]] = float(self.table_data.item(row_point+row_length+row_theta+4,0).text())*hd
    
    def clearAxes(self):
        global drawing_times
        drawing_times = 0
        self.ax1.cla()
        self.ax2.cla()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w=MyWindow()
    w.show()
    sys.exit(app.exec())