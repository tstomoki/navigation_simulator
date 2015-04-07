# import common modules #
import sys
import pdb
import datetime
import numpy as np
# import common modules #

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
