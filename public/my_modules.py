# import common modules #
import sys
import pdb
import datetime
import numpy as np
import matplotlib.pyplot as plt
# import common modules #

def graphInitializer(title, x_label, y_label):
    # clear graph
    plt.clf()
    # display grid
    plt.grid(True)
    # label
    title = title + '\n'
    plt.title (title.title(), fontweight="bold")
    plt.xlabel(x_label,       fontweight="bold")
    plt.ylabel(y_label,       fontweight="bold")
    # draw origin line
    plt.axhline(linewidth=1.5, color='k')

def mkdate(text):
    return datetime.datetime.strptime(text, '%Y/%m/%d')

# load history data of crude oil
def load_history_data():
    history_data_path = '../data/crude_oil_history.csv'
    # read data
    dt   = np.dtype({'names': ('date', 'price'),
                   'formats': ('S10' , np.float)})
    data = np.genfromtxt(history_data_path,
                         delimiter=',',
                         dtype=dt,
                         usecols=[0,1],
                         skiprows=1)
    return data
