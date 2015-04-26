#!/usr/bin/env python
import random
import pdb
import sys
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtCore import (QLineF, QPointF, QRectF, Qt, QTimer)
from PyQt5.QtGui import (QBrush, QColor, QPainter, QIntValidator)
from PyQt5.QtWidgets import (QApplication, QWidget, QGraphicsView, QGraphicsScene, QGraphicsItem,
                             QGridLayout, QVBoxLayout, QHBoxLayout, QSizePolicy,
                             QLabel, QLineEdit, QPushButton, QComboBox)

class MatplotlibWindow(FigureCanvas):
    def __init__(self, parent=None, width=500, height=500, dpi=100):
        # initializer
        # matplotlib
        fig = Figure(figsize=(width, height), dpi=dpi)        
        self.axes = fig.add_subplot(111)
        self.axes.hold(False)                
        super(MatplotlibWindow, self).__init__(fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.x  = np.arange(0, 4*np.pi, 0.1)
        self.y  = np.sin(self.x)
        
        self.width = width
        self.height = height
        self.draw_graph()
        # initializer

    def draw_graph(self):
        self.axes.plot(self.x, self.y)
        self.draw()
        
    def update_graph(self):
        self.update()
        
    def paint(self, painter, option, widget):
        pass
                    
    def boundingRect(self):
        return QRectF(0,0,self.width,self.height)                    

class MainWindow(QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Layout
        LeftBoxLayout  = QVBoxLayout()
        RightBoxLayout = QVBoxLayout()
        MainLayout     = QHBoxLayout()
        LeftBoxLayout.setAlignment(Qt.AlignTop)
        RightBoxLayout.setAlignment(Qt.AlignTop)
        MainLayout.setAlignment(Qt.AlignTop)
        
        # for matplotlib graph
        self.MatplotlibWindow = MatplotlibWindow(width=200,height=200)
        LeftBoxLayout.addWidget(self.MatplotlibWindow)
        
        # ComboBox
        self.combo = QComboBox(self)
        self.combo.addItem("Ubuntu")
        self.combo.addItem("Mandriva")
        self.combo.addItem("Fedora")
        self.combo.addItem("Red Hat")
        self.combo.addItem("Gentoo")
        self.combo.activated.connect(self.onActivated)
        self.combo.activated[str].connect(self.onActivatedstr)
        RightBoxLayout.addWidget(self.combo)
        
        # button
        self.update = QPushButton("&Update")
        self.update.clicked.connect(self.update_graph)
        RightBoxLayout.addWidget(self.update)

        MainLayout.addLayout(LeftBoxLayout)
        MainLayout.addLayout(RightBoxLayout)
        
        self.setLayout(MainLayout)
        self.setWindowTitle("Matplotlib Window")
        
    def update_graph(self):
        self.MatplotlibWindow.update_graph()
        
    def onActivated(self,number):
        print(number)#0,1,2,3,4
        
    def onActivatedstr(self,text):
        print(text)#Ubuntu,Mandriva,Fedora,Red Hat,Gentoo        

    #def keyPressEvent(self, event):
    #    key = event.key()
    #    super(MainWindow, self).keyPressEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()

    mainWindow.show()
    sys.exit(app.exec_())
