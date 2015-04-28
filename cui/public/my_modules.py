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
    dt = np.dtype({'names'  : ('id'    , 'name', 'P_D'   , 'EAR'   , 'blade_num', 'Rn'    , 'D'     , 'KT0'   , 'KT1'   , 'KT2'   , 'KQ0'   , 'KQ1'   , 'KQ2'),
                   'formats': (np.int16, 'S10' , np.float, np.float, np.float   , np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float)})

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

def first_day_of_month(d):
    return datetime.date(d.year, d.month, 1)

def add_month(current_date):
    _d, days = cal.monthrange(current_date.year, current_date.month)
    return current_date + datetime.timedelta(days=days)

def add_year(start_date, year_num=1):
    current_date = start_date
    for year_index in range(year_num):
        for month_index in range(12):
            current_date = add_month(current_date)
    return current_date

# return true with a prob_value [%] possibility
def prob(prob_value):
    N = 100
    array  = [False] * (N - prob_value) + [True] * (N)
    random.shuffle(array)
    return random.choice(array)

# preconditions: there is only 2 conditions
def get_another_condition(load_condition):
    if not load_condition in LOAD_CONDITION.keys():
        raise 'UNVALID CONDITION ERROR'

    return [condition for condition in LOAD_CONDITION.keys() if not condition == load_condition][0]
    
def load_condition_to_human(load_condition):
    return LOAD_CONDITION[load_condition]

def is_ballast(load_condition):
    return LOAD_CONDITION[load_condition] == 'ballast'

def is_full(load_condition):
    return not is_ballast(load_condition)

def km2mile(km):
    return km * 0.62137

def mile2km(mile):
    return mile * 1.6093

def ms2knot(ms):
    return ms / 0.5144444

def knot2ms(knot):
    return knot * 0.5144444

def ms2mileday(ms):
    return km2mile( ms / 1000.0 * 3600.0 * 24)

# return speed [mile/day]
def knot2mileday(knot):
    ms = knot2ms(knot)
    return km2mile( ms / 1000.0 * 3600.0 * 24)

def init_dict_from_keys_with_array(keys, dtype=None):
    ret_dict = {}
    for key in keys:
        if dtype is None:
            ret_dict[key] = np.array([])
        else:
            ret_dict[key] = np.array([], dtype=dtype)
    return ret_dict

def rpm2rps(rpm):
    return rpm / 60.0

# append for np_array
def append_for_np_array(base_array, add_array):
    if len(base_array) == 0:
        base_array = np.append(base_array, add_array)
    else:
        base_array = np.vstack((base_array, add_array))
    return base_array

def print_with_notice(display_str):
    notice_str = '*' * 60
    print "%60s" % (notice_str)
    print "%10s %s %10s" % ('*' * 10, display_str, '*' * 10)
    print "%60s" % (notice_str)
    return 

def datetime_to_human(datetime_var):
    return datetime.datetime.strftime(datetime_var, "%Y/%m/%d")

def str_to_datetime(datetime_str):
    return datetime.datetime.strptime(datetime_str, "%Y/%m/%d")

def search_near_index(current_date, date_list):
    tmp_array = np.array([])
    for index, date_elem in enumerate(date_list):
        day_delta = ( date2datetime(current_date) - str_to_datetime(date_elem) ).days
        day_delta = np.abs(day_delta)
        tmp_array = append_for_np_array(tmp_array, [day_delta, index])
    day_delta, index = tmp_array[np.argmin(tmp_array, axis=0)[0]]
    return date_list[index]

def date2datetime(current_date):
    return datetime.datetime.combine(current_date, datetime.datetime.min.time())
