#!/usr/bin/env python
import random
import sys
from PyQt5.QtCore import (QLineF, QPointF, QRectF, Qt, QTimer)
from PyQt5.QtGui import (QBrush, QColor, QPainter, QIntValidator)
from PyQt5.QtWidgets import (QApplication, QWidget, QGraphicsView, QGraphicsScene, QGraphicsItem,
                             QGridLayout, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox)

class MatplotlibWindow(QGraphicsItem):
    def __init__(self, width=500, height=500, size=5):
        super(MatplotlibWindow, self).__init__()
        self.width = width
        self.height = height
        self.size = size
        self.NH = self.height//size
        self.NW = self.width//size
        self.board = []
        for y in range(self.NH):
            self.board.append([0] * self.NW)
        self.board[0][self.NW//2] = 1
        self.pos = 0

    def update_graph(self):
        for y in range(self.NH):
            for x in range(self.NW):
                self.board[y][x] = 0
        self.board[0][self.NW//2] = 1
        self.pos = 0
        self.update()
        
    def paint(self, painter, option, widget):
        painter.setPen(QColor(220,220,220))
        for y in range(self.NH):
            painter.drawLine(0, y*self.size, self.width, y*self.size)
        for x in range(self.NW):
            painter.drawLine(x*self.size, 0, x*self.size, self.height)

        painter.setBrush(Qt.black)
        for y in range(self.NH):
            for x in range(self.NW):
                if self.board[y][x] == 1:
                    painter.drawRect(self.size*x, self.size*y, self.size, self.size)
                    
    def boundingRect(self):
        return QRectF(0,0,self.width,self.height)                    

class MainWindow(QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.graphicsView = QGraphicsView()
        scene = QGraphicsScene(self.graphicsView)
        scene.setSceneRect(0, 0, 400, 400)
        self.graphicsView.setScene(scene)
        self.MatplotlibWindow = MatplotlibWindow(400,400)
        scene.addItem(self.MatplotlibWindow)


        buttonLayout = QVBoxLayout()
        
        # ComboBox
        self.combo = QComboBox(self)
        self.combo.addItem("Ubuntu")
        self.combo.addItem("Mandriva")
        self.combo.addItem("Fedora")
        self.combo.addItem("Red Hat")
        self.combo.addItem("Gentoo")
        self.combo.activated.connect(self.onActivated)
        self.combo.activated[str].connect(self.onActivatedstr)
        
        buttonLayout.addWidget(self.combo)
        
        # button
        self.update = QPushButton("&Update")
        self.update.clicked.connect(self.update_graph)
        buttonLayout.addWidget(self.update)

       
        propertyLayout = QVBoxLayout()
        propertyLayout.setAlignment(Qt.AlignTop)
        propertyLayout.addLayout(buttonLayout)

        mainLayout = QHBoxLayout()
        mainLayout.setAlignment(Qt.AlignTop)
        mainLayout.addWidget(self.graphicsView)
        mainLayout.addLayout(propertyLayout)
        

        self.setLayout(mainLayout)
        self.setWindowTitle("Matplotlib Window")
        
        #combo.activated.connect(self.onActivated)
        #combo.activated[str].connect(self.onActivatedstr)

        
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
