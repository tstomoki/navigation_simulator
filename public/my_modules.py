# import common modules #
import math
import sys
import pdb
import datetime
import numpy as np
import matplotlib.pyplot as plt
import calendar as cal
import random
# import common modules #

# import own modules #
sys.path.append('../public')
from constants  import *
# import own modules #

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

def load_hull_list(path=None):
    if path is None:
        path = '../data/components_lists/hull_list.csv'
        
    # read data
    dt = np.dtype({'names'  : ('id'  , 'Loa'   , 'Lpp'   , 'Disp'  , 'DWT'   , 'Bmld'  , 'Dmld'  , 'draft_full', 'draft_ballast', 'Cb'    , 'S'     , 'ehp0_ballast', 'ehp1_ballast', 'ehp2_ballast', 'ehp3_ballast', 'ehp4_ballast', 'ehp0_full', 'ehp1_full', 'ehp2_full', 'ehp3_full', 'ehp4_full'),
                   'formats': (np.int16, np.float, np.float, np.float, np.float, np.float, np.float, np.float    ,  np.float      , np.float, np.float, np.float      , np.float      , np.float      , np.float      , np.float      , np.float   , np.float   , np.float   , np.float   , np.float)})

    hull_list = np.genfromtxt(path,
                              delimiter=',',
                              dtype=dt,
                              skiprows=1)    
    return hull_list

def load_engine_list(path=None):
    if path is None:
        path = '../data/components_lists/engine_list.csv'
        
    # read data
    dt = np.dtype({'names'  : ('id'    , 'name', 'sfoc0' , 'sfoc1' , 'sfoc2' , 'bhp0'  , 'bhp1'  , 'bhp2'  , 'N_max' , 'max_load'),
                   'formats': (np.int16, 'S10' , np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float)})

    engine_list = np.genfromtxt(path,
                                delimiter=',',
                                dtype=dt,
                                skiprows=1)    
    return engine_list

def load_propeller_list(path=None):
    if path is None:
        path = '../data/components_lists/propeller_list.csv'
        
    # read data
    dt = np.dtype({'names'  : ('id'    , 'P_D'   , 'EAR'   , 'D'     , 'blade_num', 'Rn'),
                   'formats': (np.int16, np.float, np.float, np.float, np.float   , np.float)})

    propeller_list = np.genfromtxt(path,
                                   delimiter=',',
                                   dtype=dt,
                                   skiprows=1)    
    return propeller_list

# load history data of crude oil
## daily
def load_history_data(from_date=None, to_date=None):
    history_data_path = '../data/crude_oil_history.csv'
    # read data
    dt   = np.dtype({'names': ('date', 'price'),
                   'formats': ('S10' , np.float)})
    data = np.genfromtxt(history_data_path,
                         delimiter=',',
                         dtype=dt,
                         usecols=[0,1],
                         skiprows=1)

    if not from_date is None:
        from_datetime = datetime.datetime.strptime(from_date, "%Y/%m/%d")
        for index in range(len(data)):
            datetime_key = datetime.datetime.strptime(data['date'][index], "%Y/%m/%d")
            if from_datetime <= datetime_key:
                break
    return data[index:]

# load history data of crude oil
## monthly
def load_monthly_history_data(from_date=None, to_date=None):
    history_data_path = '../data/crude_oil_monthly_history.csv'
    # read data
    dt   = np.dtype({'names': ('date', 'price'),
                   'formats': ('S10' , np.float)})
    data = np.genfromtxt(history_data_path,
                         delimiter=',',
                         dtype=dt,
                         usecols=[0,1],
                         skiprows=1)

    # from_date
    if not from_date is None:
        from_datetime = datetime.datetime.strptime(from_date, "%Y/%m/%d")
        for index in range(len(data)):
            datetime_key = datetime.datetime.strptime(data['date'][index], "%Y/%m/%d")
            if from_datetime <= datetime_key:
                break
        data = data[index:]

    # to_date
    if not to_date is None:
        to_datetime = datetime.datetime.strptime(to_date, "%Y/%m/%d")
        for index in range(len(data)):
            datetime_key = datetime.datetime.strptime(data['date'][index], "%Y/%m/%d")
            if to_datetime <= datetime_key:
                break
        data = data[:index]
    
    return data

# substitute inf to nan in values
def inf_to_nan_in_array(values):
    inf_induces = np.where(values==float('-inf'))[0]

    for i in range(len(inf_induces)):
        values[inf_induces[i]] = float('nan')

    return values

# load history data of world scale
## monthly
def load_world_scale_history_data(from_date=None, to_date=None):
    history_data_path = '../data/world_scale.csv'
    # read data
    dt   = np.dtype({'names': ('date', 'ws'),
                   'formats': ('S10' , np.float)})
    data = np.genfromtxt(history_data_path,
                         delimiter=',',
                         dtype=dt,
                         usecols=[0,1],
                         skiprows=1)
    index = 0
    if not from_date is None:
        from_datetime = datetime.datetime.strptime(from_date, "%Y/%m/%d")
        for index in range(len(data)):
            datetime_key = datetime.datetime.strptime(data['date'][index], "%Y/%m/%d")
            if from_datetime <= datetime_key:
                break
    return data[index:]

def add_month(current_date):
    _d, days = cal.monthrange(current_date.year, current_date.month)
    return current_date + datetime.timedelta(days=days)

# return true with a prob_value [%] possibility
def prob(prob_value):
    N = 100
    array  = [False] * (N - prob_value) + [True] * (N)
    random.shuffle(array)
    return random.choice(array)
