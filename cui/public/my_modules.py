# -*- coding: utf-8 -*-
# import common modules #
import math
import os
import sys
from pdb import *
import datetime
import numpy as np
import matplotlib.pyplot as plt
import calendar as cal
import random
import csv
import json
import operator
import re
from pylab import *
import pandas as pd
from scipy.interpolate import spline
import operator
# import common modules #

# import own modules #
sys.path.append('../public')
sys.path.append('../models')
from constants  import *
from cubic_module import *
# import own modules #

#initialize dir_name
def initializeDir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return

#initialize dir_name
def initializeDirHierarchy(dir_name):
    split_dir_names = dir_name.split('/')
    if len(split_dir_names) == 0:
        return 
    initialize_dir = split_dir_names[0]
    initializeDir(initialize_dir)
    for split_dir_name in split_dir_names[1:]:
        initialize_dir += '/' + split_dir_name
        initializeDir(initialize_dir)
    return

def graphInitializer(title, x_label, y_label):
    # clear graph
    plt.clf()
    # display grid
    plt.grid(True)
    # label
    title = title + '\n'
    plt.title(title,       fontweight="bold")
    plt.xlabel(x_label,       fontweight="bold")
    plt.ylabel(y_label,       fontweight="bold")
    # draw origin line
    plt.axhline(linewidth=1.5, color='k')
    return

def mkdate(text):
    return datetime.datetime.strptime(text, '%Y/%m/%d')

def load_hull_list(path=None):
    if path is None:
        path = '../data/components_lists/hull_list.csv'
        
    # read data
    dt = np.dtype({'names'  : ('id'  , 'Loa'   , 'Lpp'   , 'Disp'  , 'DWT'   , 'Bmld'  , 'Dmld'  , 'draft_full', 'draft_ballast', 'Cb'    , 'S'     , 'ehp0_ballast', 'ehp1_ballast', 'ehp2_ballast', 'ehp3_ballast', 'ehp4_ballast', 'ehp0_full', 'ehp1_full', 'ehp2_full', 'ehp3_full', 'ehp4_full', 'with_bow'),
                   'formats': (np.int16, np.float, np.float, np.float, np.float, np.float, np.float, np.float    ,  np.float      , np.float, np.float, np.float      , np.float      , np.float      , np.float      , np.float      , np.float   , np.float   , np.float   , np.float   , np.float, 'S10')})

    hull_list = np.genfromtxt(path,
                              delimiter=',',
                              dtype=dt,
                              skiprows=1)    
    return hull_list

def load_engine_list(path=None):
    if path is None:
        path = '../data/components_lists/engine_list.csv'
        
    # read data
    dt = np.dtype({'names'  : ('id'    , 'name', 'specific_name', 'sfoc0' , 'sfoc1' , 'sfoc2' , 'bhp0'  , 'bhp1'  , 'bhp2'  , 'N_max' , 'max_load', 'sample_rpm0', 'sample_bhp0', 'sample_rpm1', 'sample_bhp1'),
                   'formats': (np.int16, 'S10' , 'S20'          , np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float  ,  np.float    ,  np.float    ,  np.float    , np.float)})

    engine_list = np.genfromtxt(path,
                                delimiter=',',
                                dtype=dt,
                                skiprows=1)    
    return engine_list

def load_propeller_list(path=None):
    if path is None:
        #path = '../data/components_lists/propeller_list.csv'
        path = '../data/components_lists/results/selected_propeller.csv'
        
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

# load history data of flat rate
## monthly
def load_flat_rate_history_data(from_date=None, to_date=None):
    history_data_path = '../data/flat_rate.csv'
    # read data
    dt   = np.dtype({'names': ('date', 'fr'),
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

def get_days_num_in_the_month(current_date):
    _, days_num = cal.monthrange(current_date.year, current_date.month)
    return days_num

def first_day_of_month(d):
    return datetime.date(d.year, d.month, 1)

def add_month(current_date, num=1):
    for _n in range(num):    
        _d, days      = cal.monthrange(current_date.year, current_date.month)
        current_date += datetime.timedelta(days=days)
    return current_date

def add_year(start_date, year_num=1):
    current_date = start_date
    if year_num < 1:
        current_date = add_month(current_date, int(year_num * 12))
        return current_date
    
    for year_index in range(year_num):
        current_date = add_month(current_date, 12)
    return current_date

# return true with a prob_value [%] possibility
def prob(prob_value):
    N = 10000
    nonzero_num = np.count_nonzero(np.random.binomial(1, prob_value, N))
    threshold   = N * prob_value
    return (nonzero_num > threshold)

# {c0: 0.1, c2:0.3.....}
def prob_with_weight(weight_dict):
    # normalization
    data_sum = sum(weight_dict.values())
    return np.random.choice(weight_dict.keys(), p=[ _v / data_sum for _v in weight_dict.values()])

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
    notice_str = '*' * 80
    print "%80s" % (notice_str)
    print "%10s %s %10s" % ('*' * 10, display_str, '*' * 10)
    print "%80s" % (notice_str)
    return 

def datetime_to_human(datetime_var):
    return datetime.datetime.strftime(datetime_var, "%Y/%m/%d")

def detailed_datetime_to_human(datetime_var):
    return datetime.datetime.strftime(datetime_var, "%Y/%m/%d %H:%M:%S")

def daterange_to_str(start_date, end_date):
    return "%s-%s" % (start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))

def str_to_datetime(datetime_str):
    return datetime.datetime.strptime(datetime_str, "%Y/%m/%d")

def str_to_date(datetime_str):
    tdatetime = str_to_datetime(datetime_str)
    return datetime.date(tdatetime.year, tdatetime.month, tdatetime.day)

def search_near_index(current_date, date_list):
    tmp_array = np.array([])
    for index, date_elem in enumerate(date_list):
        day_delta = ( date2datetime(current_date) - str_to_datetime(date_elem) ).days
        day_delta = np.abs(day_delta)
        tmp_array = append_for_np_array(tmp_array, [day_delta, index])
    day_delta, index = tmp_array[np.argmin(tmp_array, axis=0)[0]] if not isinstance(np.argmin(tmp_array, axis=0), np.int64) else tmp_array
    return date_list[index]

def date2datetime(current_date):
    return datetime.datetime.combine(current_date, datetime.datetime.min.time())

# return string num
def number_with_delimiter(num):
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US')
    return locale.format("%0.4lf", num, grouping=True)

def generate_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d%H%M')

def write_csv(column_names, write_data, output_file_path):
    # for initial write
    write_column_flg = False if os.path.exists(output_file_path) else True
    
    # write file
    f = open(output_file_path, 'a')
    csvWriter = csv.writer(f)
    if write_column_flg:
        csvWriter.writerow(column_names)
    csvWriter.writerow(write_data)
    
    f.close()
    return

def write_simple_array_csv(column_names, write_data, output_file_path):
    # for initial write
    write_column_flg = False if os.path.exists(output_file_path) else True
    
    # write file
    f = open(output_file_path, 'a')
    csvWriter = csv.writer(f)
    if write_column_flg:
        csvWriter.writerow(column_names)
    csvWriter.writerows(write_data)
    
    f.close()
    return

def write_array_to_csv(column_names, write_array, output_file_path):
    # write file
    f = open(output_file_path, 'w')
    csvWriter = csv.writer(f)
    # write row
    csvWriter.writerow(column_names)

    # write array data based on given column names
    for write_row in write_array:
        write_row_data = []
        for column_name in column_names:
            write_data = write_row[column_name]
            # for numpy array
            if isinstance(write_data, np.ndarray):
                write_data = write_data[0]
            write_row_data.append(write_data)
        csvWriter.writerow(write_row_data)
    f.close()
    return 

# for the callback results
def flatten_3d_to_2d(array_3d):
    ret_combinations = np.array([])
    for array_2d in array_3d:
        # ignore the vacant array
        if len(array_2d) == 0:
            continue

        if len(ret_combinations) == 0:
            ret_combinations = array_2d
        else:
            try:
                ret_combinations = np.r_[ret_combinations, array_2d]
            except:
                print "error occured at "
    return ret_combinations

'''
def devide_array(combinations, devide_num):
    ret_combinations   = np.array([])
    combinations_num   = len(combinations)
    stride             = math.floor( combinations_num / devide_num)
    combinations_index = 0
    while True:
        if combinations_index + stride >= combinations_num:
            break
        start_index   = combinations_index
        end_index     = combinations_index + stride
        devided_array = combinations[start_index:end_index]
        pdb.set_trace()
        ret_combinations = np.vstack((ret_combinations, devided_array))        
        combinations_index += stride
    pdb.set_trace()
    ret_combinations  = np.vstack((ret_combinations, combinations[start_index:]))
    return retu_combinations
'''

def error_printer(exception):
    print '================================= Error detail ================================='
    print "%10s: %s" % ('type'   , str(type(exception)))
    print "%10s: %s" % ('args'   , str(exception.args))
    print "%10s: %s" % ('message', exception.message)
    print "%10s: %s" % ('error'  , str(exception))
    print '================================= Error detail ================================='

# seconds -> MM:HH:SS
def convert_second(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes   = divmod(minutes, 60)
    return "%02d:%02d:%02d" % (hours, minutes, seconds)

def draw_BHP_approxmate_graph():
    from engine      import Engine
    engine_list = load_engine_list()
    engine_1    = Engine(engine_list, 2)
    engine_2    = Engine(engine_list, 3)
    title       = "engine details".title()
    x_label     = "rpm"
    y_label     = "Engine Output [kW]"
    graphInitializer(title, x_label, y_label)

    # E1
    rpm_array = np.arange(50.0, 95, RPM_RANGE['stride'])            
    draw_data = np.array([ [rpm, engine_1.calc_bhp(rpm)] for rpm in rpm_array])
    plt.plot(draw_data.transpose()[0],
             draw_data.transpose()[1],
             color='b', lw=5, label='Engine 1')

    # E2
    rpm_array = np.arange(50.0, 105, RPM_RANGE['stride'])            
    draw_data = np.array([ [rpm, engine_2.calc_bhp(rpm)] for rpm in rpm_array])
    plt.plot(draw_data.transpose()[0],
             draw_data.transpose()[1],
             color='g', lw=5, label='Engine 2')    

    plt.xlim([45, 111])
    plt.ylim([8000, 25000])
    plt.legend(shadow=True)
    plt.legend(loc='upper left')    
    plt.savefig('engine_detail.png')
    plt.close()

def draw_SFOC_approxmate_graph():
    from engine      import Engine
    engine_list = load_engine_list()
    engine_1    = Engine(engine_list, 2)
    engine_2    = Engine(engine_list, 3)
    title       = "engine details".title()
    x_label     = "Load [%]"
    y_label     = "SFOC [g/kW]"
    graphInitializer(title, x_label, y_label)

    # E1
    rpm_array = np.arange(50.0, 95, RPM_RANGE['stride'])
    draw_data = np.array([ [engine_1.calc_load(engine_1.calc_bhp(rpm)) * 100, engine_1.calc_sfoc(engine_1.calc_bhp(rpm))] for rpm in rpm_array])
    plt.plot(draw_data.transpose()[0],
             draw_data.transpose()[1],
             color='b', lw=5, label='Engine 1')

    # E2
    rpm_array = np.arange(50.0, 105, RPM_RANGE['stride'])
    draw_data_2 = np.array([ [engine_2.calc_load(engine_2.calc_bhp(rpm)) * 100, engine_2.calc_sfoc(engine_2.calc_bhp(rpm))] for rpm in rpm_array])
    plt.plot(draw_data_2.transpose()[0],
             draw_data_2.transpose()[1],
             color='g', lw=5, label='Engine 2')    

    plt.xlim([40, 109])
    plt.ylim([162, 172])
    plt.legend(shadow=True)
    plt.legend(loc='upper left')
    plt.savefig('engine_sfoc_detail.png')
    plt.close()

def write_file_as_text(string, output_path, open_mode):
    if open_mode == 'a':
        string += "\n"
    f = open(output_path, open_mode)
    f.write(string)
    f.close()     
    return 
    
def write_file_as_json(dict_file, output_path):
    f = open(output_path, 'w')
    json_data = json.dumps(dict_file, indent=4)
    f.write(json_data)
    f.close()     
    return 

def load_json_file(json_file_path):
    f = open(json_file_path, 'r')
    json_data = json.load(f)
    f.close()
    return json_data

def check_combinations_exists(hull, engine, propeller):
    ret_combinations = None
    combination_str  = generate_combination_str(hull, engine, propeller)
    file_path        = "%s/designs/%s/%s_combinations.json" % (COMBINATIONS_DIR_PATH, combination_str, combination_str)

    if os.path.exists(file_path):
        ret_combinations = {}
        raw_data         = load_json_file(file_path)
        ret_combinations = init_dict_from_keys_with_array(LOAD_CONDITION.values())
        for key, load_condition in LOAD_CONDITION.items():
            for combination in raw_data[load_condition]:
                add_combination = np.array(combination)
                ret_combinations[load_condition] = append_for_np_array(ret_combinations[load_condition], add_combination)
    return ret_combinations

def generate_combination_str(hull, engine, propeller):
    return "H%dE%dP%d" % (hull.base_data['id'], engine.base_data['id'], propeller.base_data['id'])

def generate_combination_str_with_id(hull_id, engine_id, propeller_id):
    hull_id_str      = str(hull_id[0]) if isinstance(hull_id, np.ndarray) else str(hull_id)
    engine_id_str    = str(engine_id[0]) if isinstance(engine_id, np.ndarray) else str(engine_id)
    propeller_id_str = str(propeller_id[0]) if isinstance(propeller_id, np.ndarray) else str(propeller_id)
    return "H%sE%sP%s" % (hull_id_str, engine_id_str, propeller_id_str)

def select_retrofits_design(temp_npv):
    retrofit_design = {}

    # consider appearance time
    appr_time = {}
    for combination_key, npv_array in temp_npv.items():
        appr_time[combination_key]  = len(npv_array)
    appr_ranked_keys = sorted(appr_time, key=appr_time.get)
    appr_ranked_dict = {}
    for index, ranked_key in enumerate(appr_ranked_keys):
        appr_ranked_dict[ranked_key] = index

    # get maximum average NPV for each design
    design_npv = {}
    for combination_key, npv_array in temp_npv.items():
        design_npv[combination_key] = np.average(npv_array)
    NPV_ranked_keys = sorted(appr_time, key=appr_time.get)
    NPV_ranked_dict = {}
    for index, ranked_key in enumerate(NPV_ranked_keys):
        NPV_ranked_dict[ranked_key] = index

    # calc total score
    total_rank = {}
    for combination_key, npv_array in temp_npv.items():
        total_rank[combination_key] = calc_retrofit_score(appr_ranked_dict[combination_key],
                                                          NPV_ranked_dict[combination_key],
                                                          len(temp_npv))
    retrofit_design_key = max(total_rank.iteritems(), key=operator.itemgetter(1))[0]

    retrofit_design[retrofit_design_key] = design_npv[retrofit_design_key]
    return retrofit_design

def calc_retrofit_score(appr_rank, NPV_rank, rank_length):
    # appearance time
    appearance_score = calc_base_score(appr_rank, rank_length) * APPR_RANK_WEIGHT

    # averaged NPV
    NPV_score        = calc_base_score(NPV_rank, rank_length)  * NPV_RANK_WEIGHT
    
    total_score = appearance_score + NPV_score
    return total_score

def calc_base_score(rank, rank_length):
    if rank_length == 1:
        return 1.0
    
    return math.exp( (2 * rank) / float(1 - rank_length) * math.log(10) )

# retrofit_design: (hull_id, engine_id, propeller_id, NPV)
def change_design(retrofit_design):
    # import models #
    from hull        import Hull
    from engine      import Engine
    from propeller   import Propeller
    # import models #
    hull_id, engine_id, propeller_id, NPV = retrofit_design

    # load components list
    hull_list           = load_hull_list()
    engine_list         = load_engine_list()
    propeller_list      = load_propeller_list()

    hull      = Hull(hull_list,           int(hull_id))
    engine    = Engine(engine_list,       int(engine_id))
    propeller = Propeller(propeller_list, int(propeller_id))
    return hull, engine, propeller

def update_retrofit_design_log(retrofit_design_log, combinations, scenario_num):
    for combination in combinations:
        combination_key = generate_combination_str_with_id(combination['hull_id'],
                                                           combination['engine_id'],
                                                           combination['propeller_id'])
        if not retrofit_design_log.has_key(combination_key):
            retrofit_design_log[combination_key] = {}
        retrofit_design_log[combination_key][scenario_num] = combination['NPV']
    return retrofit_design_log

def draw_NPV_for_retrofits(retrofit_design_log, output_dir_path, dockin_date_str, current_combinations):
    hull, engine, propeller = current_combinations
    current_combination_key = generate_combination_str(hull, engine, propeller)
    png_filename = "%s/%s.png" % (output_dir_path, dockin_date_str)
    title    = "NPV comparison with current design during %s\n" % (dockin_date_str)
    x_label = "combination id".upper()
    y_label = "%s %s" % ('npv'.upper(), '[$]')
    # initialize graph
    graphInitializer(title, x_label, y_label)
    # overwrite
    plt.title (title, fontweight="bold")
    # draw current_design
    current_design_NPV = np.average(retrofit_design_log[current_combination_key].values())
    plt.axhline(current_design_NPV, 
                xmin=0, xmax=len(retrofit_design_log.keys()),                    
                linewidth=1.5, color='r',
                linestyle="--")
    # draw other NPV
    draw_data     = np.array([])
    for_json_data = {'current_design': {'combination': current_combination_key,  'average_NPV': current_design_NPV}, 'other_design': {}}
    for index, combination_key in enumerate(retrofit_design_log.keys()):
        # exclude current design
        if combination_key == current_combination_key:
            continue
        ave_npv       = np.average(retrofit_design_log[combination_key].values())
        add_elem      = np.array([index, ave_npv])
        draw_data     = append_for_np_array(draw_data, add_elem)
        for_json_data['other_design'][combination_key] = ave_npv
        
    x_data    = draw_data.transpose()[0]
    y_data    = draw_data.transpose()[1]
    plt.bar(x_data, y_data)
    plt.ylim([0, np.max(y_data) * 1.1])
    plt.text(0, current_design_NPV*1.01, "%0.2lf" % (round(current_design_NPV, 2)), fontsize=10)
    plt.savefig(png_filename)
    plt.close()
    
    # output as json
    json_filename = "%s/%s.json" % (output_dir_path, dockin_date_str)
    write_file_as_json(for_json_data, json_filename)
    return

def read_csv(filepath):
    load_data = np.array([])
    with open(filepath, 'r') as f:
        reader    = csv.reader(f)
        header    = next(reader)
        for row in reader:
            load_data = append_for_np_array(load_data, row)
    return header, load_data

def draw_initial_design_graph(filepath):
    header, load_data = read_csv(filepath)
    draw_data = np.array([])
    for index, combination in enumerate(load_data):
        add_elem  = np.array([index, combination[3]])
        draw_data = append_for_np_array(draw_data, add_elem)
        
    png_filename = "./%s.png" % ('initial_design')
    title   = "NPV comparison with initial design\n"
    x_label = "combination id".upper()
    y_label = "%s %s" % ('npv'.upper(), '[$]')
    # initialize graph
    graphInitializer(title, x_label, y_label)
    # overwrite
    plt.title (title, fontweight="bold")
    
    x_data    = draw_data.transpose()[0].astype(np.int32)
    y_data    = draw_data.transpose()[1].astype(np.float)
    plt.bar(x_data, y_data)
    plt.ylim([np.max(y_data) / 1.2, np.max(y_data) * 1.1])
    plt.savefig(png_filename)
    plt.close()    
    return 

def generate_date_array(start_date, end_date):
    number_of_days = (end_date - start_date).days + 1
    ret_array      = [start_date + datetime.timedelta(days=x) for x in range(0, number_of_days)]
    return ret_array

def aggregate_combinations(raw_combinations, output_dir_path):
    dtype  = np.dtype({'names': ('hull_id', 'engine_id', 'propeller_id', 'averaged_NPV', 'std'),
                       'formats': (np.int, np.int, np.int , np.float, np.float)})
    ret_combinations = np.array([], dtype=dtype)

    hull_ids      = np.unique(raw_combinations['hull_id'])
    engine_ids    = np.unique(raw_combinations['engine_id'])
    propeller_ids = np.unique(raw_combinations['propeller_id'])

    for hull_id in hull_ids:
        for engine_id in engine_ids:
            for propeller_id in propeller_ids:
                target_induces  = np.where( (raw_combinations['hull_id']==hull_id) &
                                            (raw_combinations['engine_id']==engine_id) &
                                            (raw_combinations['propeller_id']==propeller_id) )
                target_results  = raw_combinations[target_induces]
                if len(target_results) == 0:
                    continue
                averaged_NPV    = np.average(target_results['NPV'])
                std             = np.std(target_results['NPV'])
                add_combination = np.array([(hull_id,
                                             engine_id,
                                             propeller_id,
                                             averaged_NPV,
                                             std)],
                                           dtype=dtype)
            ret_combinations = append_for_np_array(ret_combinations, add_combination)
            output_NPV_log_to_json(generate_combination_str_with_id(hull_id, engine_id, propeller_id),
                                   averaged_NPV,
                                   std,
                                   target_results,
                                   output_dir_path)
    return ret_combinations

# curve fitting #
def calc_y(x, wlist, M):
    ret = wlist[0]
    for i in range(1, M+1):
        ret += wlist[i] * (x ** i)
    return ret

# estimate params with training data #
def estimate(xlist, tlist, M):
    # (M+1) params exists for the Mth polynomial expression
    A = []
    for i in range(M+1):
        for j in range(M+1):
            temp = (xlist**(i+j)).sum()
            A.append(temp)
    A = array(A).reshape(M+1, M+1)
        
    T = []
    for i in range(M+1):
        T.append(((xlist**i) * tlist).sum())
    T = array(T)
    # w is the solution
    wlist = np.linalg.solve(A, T)
    return wlist

def draw_approx_curve(coefficients, title, dir_path, xlist, degree, x_label, y_label):
    output_file_path = "%s/engine_%s.png" % (dir_path, title_to_snake(title))
    graphInitializer(title,
                     x_label,
                     y_label)
    plt.savefig(output_file_path)
    draw_data = np.array([ (_x, calc_y(_x, coefficients, degree)) for _x in xlist])
    plt.plot(draw_data.transpose()[0],
             draw_data.transpose()[1],
             color='b')
    plt.savefig(output_file_path)
    plt.close()    
    return
# curve fitting #

def title_to_snake(title):
    return re.sub(' ', r'_', title).lower()

def analyze_correlation(oil_price_history_data, world_scale_history_data, date_range=None):
    # define date_range #
    if not date_range is None:
        raw_date        = np.array([str_to_datetime(_date) for _date in oil_price_history_data['date'][:-1]])
        target_induces  = np.where( (raw_date > date_range['start']) & ( raw_date < date_range['end']) )
        target_oil_data = oil_price_history_data[target_induces]
    else:
        target_oil_data = oil_price_history_data
    # cull data for the oilprice date
    culled_world_scale_data = []
    for oil_price_date in target_oil_data['date']:
        target_index            = np.where(world_scale_history_data['date']==oil_price_date)[0]
        if len(target_index) == 0:
            continue
        target_data = world_scale_history_data[target_index]
        target_date = target_data['date'][0]
        target_ws   = target_data['ws'][0]
        add_elem    = (target_date, target_ws)
        culled_world_scale_data.append(add_elem)
    culled_world_scale_data = np.array(culled_world_scale_data,
                                       dtype=world_scale_history_data.dtype)
    panda_frame = pd.DataFrame({'date': [str_to_datetime(_date) for _date in target_oil_data['date']],
                                'oilprice': target_oil_data['price'],
                                'world_scale': culled_world_scale_data['ws']})

    title           = "correlation between oilprice and world_scale"
    output_dir_path = "%s/oilprice_ws/%s" % (CORRELATION_DIR_PATH, daterange_to_str(str_to_datetime(target_oil_data['date'][0]), str_to_datetime(target_oil_data['date'][-1])))
    initializeDirHierarchy(output_dir_path)
    text_file_path  = "%s/describe.txt" % output_dir_path
    write_file_as_text(str(panda_frame.corr())    , text_file_path, 'w')
    write_file_as_text(str(panda_frame.describe()), text_file_path, 'a')
    draw_statistical_graphs(panda_frame, title, "oilprice_ws_crr", output_dir_path)
    return

def draw_statistical_graphs(panda_frame, title, filename, output_dir_path):
    # scatter_matrix
    filepath = "%s/%s_scatter_matrix.png" % (output_dir_path, filename)
    plt.figure()
    plt.title(title, fontweight="bold")
    pd.scatter_matrix(panda_frame)
    plt.savefig(filepath)
    plt.clf()
    plt.close()

    # plot
    filepath = "%s/%s_plot.png" % (output_dir_path, filename)
    plt.figure()
    plt.title(title, fontweight="bold")
    panda_frame.plot()
    plt.savefig(filepath)
    plt.clf()
    plt.close()

    '''
    # area
    filepath = "%s/%s_area.png" % (output_dir_path, filename)
    plt.figure()
    plt.title(title, fontweight="bold")
    panda_frame.plot(kind='area', legend=True)
    plt.savefig(filepath)
    plt.clf()    
    '''

    # hist
    filepath = "%s/%s_hist.png" % (output_dir_path, filename)
    plt.figure()
    plt.title(title, fontweight="bold")
    panda_frame['oilprice'].hist(color="#5F9BFF", alpha=.5, label='oilprice')
    panda_frame['world_scale'].hist(color="#F8766D", alpha=.5, label='world_scale')
    plt.legend(shadow=True)
    plt.legend(loc='upper right')        
    plt.savefig(filepath)
    plt.clf()
    plt.close()
    return

def calc_change_rate(previous_value, predicted_value):
    return ( float(predicted_value) / float(previous_value) ) - 1.0

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def output_NPV_log_to_json(design_str, averaged_NPV, std, raw_results, output_dir_path):
    output_file_path = "%s/%s.json" % (output_dir_path, design_str)
    raw_results_dict = dict((raw_result['scenario_num'], raw_result['NPV']) for raw_result in raw_results)
    write_dict       = {'averaged_NPV': averaged_NPV,
                        'std'         : std,
                        'raw_results' : raw_results_dict}
    write_file_as_json(write_dict, output_file_path)    
    return

def unleash_np_array_array(np_array_array):
    return np.array([_d[0] for _d in np_array_array])

def get_component_from_id_array(component_ids, hull_list, engine_list, propeller_list):
    # import models #
    from hull        import Hull
    from engine      import Engine
    from propeller   import Propeller
    # import models #    
    hull_id, engine_id, propeller_id = component_ids
    hull      = Hull(hull_list, hull_id)
    engine    = Engine(engine_list, engine_id) 
    propeller = Propeller(propeller_list, propeller_id)    
    return hull, engine, propeller

def draw_NPV_histogram(json_filepath, output_filepath):
    data         = load_json_file(json_filepath)
    averaged_NPV = data['averaged_NPV']
    
    title   = "NPV histogram\n"
    x_label = "%s [$]" % "npv".upper()
    y_label = "frequency".upper()

    # initialize graph
    graphInitializer(title, x_label, y_label)
    # overwrite title
    plt.title (title, fontweight="bold")

    # draw averaged pv
    plt.axvline(x=averaged_NPV, color='r', linewidth=3, linestyle='--')
    
    draw_data = np.array([])
    for scenario_num, npv in data['raw_results'].items():
        draw_data = np.append(draw_data, npv)
    
    range_minimum = np.amin(draw_data)
    range_maximum = np.amax(draw_data)    
    plt.hist(draw_data, normed=False, bins=20, alpha=0.8, range=(range_minimum, range_maximum))
    plt.savefig(output_filepath)
    plt.close()
    return

def draw_each_NPV_distribution(designs_data, output_dirpath):
    # create dir
    distribution_dir_path = "%s/distribution" % (output_dirpath)
    initializeDirHierarchy(distribution_dir_path)

    x_label = 'NPV [$]'
    y_label = 'frequency'.upper()
    for design_key, design_data in designs_data.items():
        draw_data = []
        title = "NPV Distribution of design %s" % (design_key)
        for key, data in design_data['raw_results'].items():
            draw_data.append([int(key), data])
        draw_data      = np.array(draw_data)
        panda_frame    = pd.DataFrame({'scenario_num': draw_data.transpose()[0],
                                       'NPV': draw_data.transpose()[1]})
        text_file_path = "%s/describe_%s.txt" % (distribution_dir_path, design_key)
        write_file_as_text(str(panda_frame.describe()), text_file_path, 'w')
        # hist
        filepath = "%s/hist_%s.png" % (distribution_dir_path, design_key)
        plt.figure()
        graphInitializer(title, x_label, y_label)

        panda_frame['NPV'].hist(color="#5F9BFF", alpha=.5)
        plt.savefig(filepath)
        plt.clf()
        plt.close()        
    return 

def draw_NPV_for_each3D(designs_data, output_dirpath, ylim=None, zlim=None):
    title   = "PV histogram for each design\n"
    x_label = "engine_id".upper()
    y_label = "propeller_id".upper()    
    z_label = "averaged_NPV".upper()

    draw_data = []
    for design_key, design_data in designs_data.items():
        hull_id, engine_id, propeller_id = get_component_ids_from_design_key(design_key) 
        append_data = [hull_id, engine_id, propeller_id, design_data['averaged_NPV'] ]
        draw_data.append(append_data)
    draw_data = np.array(clean_draw_data(draw_data))

    xlist     = draw_data.transpose()[1]
    ylist     = draw_data.transpose()[2]
    zlist     = draw_data.transpose()[3]

    column_names = [ "%s%s" % ('engine'.upper(), _x) for _x in np.unique(xlist)]
    row_names    = [ "%s%s" % ('propeller'.upper(), _x) for _x in np.unique(ylist)]

    # draw scatter
    output_filepath = "%s/NPV_for_each_design3D_scatter.png" % (output_dirpath)
    draw_3d_scatter(xlist.astype(np.int64), ylist.astype(np.int64), zlist.astype(np.float), x_label, y_label, z_label, column_names, row_names, ylim, zlim, 0.4)
    xticks       = np.unique(xlist).astype(np.int64)
    plt.xticks(xticks-0.5, xticks)
    plt.savefig(output_filepath)
    plt.close()
    
    # draw bar
    output_filepath = "%s/NPV_for_each_design3D_bar.png" % (output_dirpath)
    draw_3d_bar(xlist.astype(np.int64), ylist.astype(np.int64), zlist.astype(np.float), x_label, y_label, z_label, column_names, row_names, ylim, zlim, 0.4)
    xticks       = np.unique(xlist).astype(np.int64)
    plt.xticks(xticks-0.5, xticks)
    plt.savefig(output_filepath)
    plt.close()

    # describe
    panda_frame = pd.DataFrame({'engine_list': xlist.astype(np.int64),
                                'propeller_list': ylist.astype(np.int64),
                                'averaged_NPV': zlist.astype(np.float)})
    text_file_path  = output_filepath[:-4] + '_describe.txt'
    write_file_as_text(str(panda_frame.describe()), text_file_path, 'w')
    return

def draw_NPV_histogram_m(json_filepath, output_filepath):
    data         = load_json_file(json_filepath)
    averaged_NPV = data['averaged_NPV']
    
    title   = "NPV histogram\n"
    x_label = "%s [$]" % "npv".upper()
    y_label = "frequency".upper()

    # initialize graph
    graphInitializer(title, x_label, y_label)
    # overwrite title
    plt.title (title, fontweight="bold")

    # draw averaged pv
    plt.axvline(x=averaged_NPV, color='r', linewidth=3, linestyle='--')
    
    draw_data = np.array([])
    for scenario_num, npv in data['raw_results'].items():
        draw_data = np.append(draw_data, npv)
    
    range_minimum = np.amin(draw_data)
    range_maximum = np.amax(draw_data)    
    plt.hist(draw_data, normed=False, bins=20, alpha=0.8, range=(range_minimum, range_maximum))
    plt.savefig(output_filepath)
    plt.close()
    return

def get_component_ids_from_design_key(design_key):
    p = re.compile(r'H(\d+)E(\d+)P(\d+)')
    try:
        a = p.search(design_key)
    except:
        pdb.set_trace()
    hull_id, engine_id, propeller_id = a.groups()

    return hull_id, engine_id, propeller_id

def load_result(result_path):
    # return vacant dict if path doesn't exist
    if not os.path.exists(result_path):
        return {}

    ret_result = {}
    result_data = np.array([])
    result_files = [ "%s/%s" % (result_path, _f) for _f in os.listdir(result_path) if _f[:14] == 'initial_design']

    for result_file in result_files:
        header, load_data = read_csv(result_file)
        if len(result_data) == 0:
            result_data = load_data
        else:
            for _e in load_data:
                result_data = np.vstack((result_data, _e))

    for element in result_data:
        scenario_num, hull_id, engine_id, propeller_id, NPV, lap_time = element
        combination_key = generate_combination_str_with_id(hull_id, engine_id, propeller_id)
        if not ret_result.has_key(combination_key):
            ret_result[combination_key] = {}
        ret_result[combination_key][int(scenario_num)] = float(NPV)
    return ret_result

# display maximum_designs
def display_maximum_designs(designs_data, display_num):
    each_data = []
    for design_key, design_data in designs_data.items():
        each_data.append((design_key, np.average([val for key, val in design_data['raw_results'].items()])))
    dtype           = np.dtype({'names': ('key', 'averaged_NPV'),
                                'formats': ('S10', np.float)})
    each_data       = np.array(each_data, dtype=dtype)
    maximum_designs = each_data[each_data['averaged_NPV'].argsort()[-display_num:][::-1]]


    for maximum_design in maximum_designs:
        design_key, averaged_NPV = maximum_design
        std = float(designs_data[design_key]['std'])
        hull_id, engine_id, propeller_id = get_component_ids_from_design_key(design_key)        
        print_str = "hull_id: %s, engine_id: %s, propeller_id: %s, averaged_NPV: %lf, standard deviation: %lf, variance: %lf" % (hull_id, engine_id, propeller_id,
                                                                                                                                 averaged_NPV,
                                                                                                                                 std,
                                                                                                                                 math.pow(std, 2))
        print print_str.upper()
    return
    
def draw_whole_NPV(designs_data, output_dir_path):
    each_data = []
    for design_key, design_data in designs_data.items():
        each_data.append((design_key, np.average([val for key, val in design_data['raw_results'].items()])))
    dtype     = np.dtype({'names': ('key', 'averaged_NPV'),
                                'formats': ('S10', np.float)})
    each_data = np.array(each_data, dtype=dtype)
    title     = "NPV comparison with top designs\n"
    x_label   = "combination id".upper()
    y_label   = "%s %s" % ('npv'.upper(), '[$]')
    # initialize graph
    graphInitializer(title, x_label, y_label)
    # overwrite
    plt.title (title, fontweight="bold")

    png_filename = "%s/comparison_whole_NPV.png" % (output_dir_path)
    draw_data = []
    for index, data in enumerate(each_data):
        design_key, averaged_NPV = data
        draw_data.append((index, averaged_NPV))
    draw_data = np.array(draw_data)
    x_data    = draw_data.transpose()[0]
    y_data    = draw_data.transpose()[1]
    plt.bar(x_data, y_data)
    plt.xlim([0, len(x_data)])
    plt.ylim([floor(y_data.min()), np.max(y_data) * 1.001])
    plt.savefig(png_filename)
    plt.close()
    return

def clean_draw_data(draw_data):
    # load components list
    engine_list         = load_engine_list()
    propeller_list      = load_propeller_list()

    hull_id = '1'
    for engine_info in engine_list:
        engine_id = str(engine_info[0])
        for propeller_info in propeller_list:
            propeller_id = str(propeller_info[0])
            if len( [_d for _d in draw_data if _d[0] == hull_id and _d[1] == engine_id and _d[2] == propeller_id]) == 0:
                append_data = [hull_id, engine_id, propeller_id, 0 ]
                draw_data.append(append_data)
    return draw_data

def generate_market_scenarios(scenario, world_scale, flat_rate, sinario_mode, simulation_duration_years):
    scenario.generate_sinario(sinario_mode, simulation_duration_years)
    world_scale.generate_sinario_with_oil_corr(sinario_mode, scenario.history_data[-1], scenario.predicted_data)
    flat_rate.generate_flat_rate(sinario_mode, simulation_duration_years)
    return

def separate_list(raw_list, num):
    ret_data = []
    delta = round( len(raw_list) / float(num) )
    index = 0
    while index < len(raw_list):
        index = int(index)
        if (index+delta) > len(raw_list):
            ret_data.append(raw_list[index:])
        else:
            ret_data.append(raw_list[index:int(index+delta)])
        index += delta
    return ret_data

def get_wave_height(current_bf):
    bf_info_path = "%s/beaufort_info.csv" % (DATA_PATH)
    dt   = np.dtype({'names': ('BF', 'wind_speed', 'wave_height', 'wave_period'),
                   'formats': ('S5', np.float, np.float, np.float)})
    bf_info = np.genfromtxt(bf_info_path,
                            delimiter=',',
                            dtype=dt,
                            skiprows=1)
    current_bf_info = bf_info[np.where(bf_info['BF'] == current_bf)]
    if len(current_bf_info) == 0:
        current_wave_height = 0
    else:
        current_wave_height = current_bf_info['wave_height'][0]
    return current_wave_height

# consider bow for velocity
def consider_bow_for_v(hull, velocity, load_condition):
    if hull.base_data['with_bow'] == 'FALSE':
        return velocity
    index = 3.0 if LOAD_CONDITION[load_condition] == 'ballast' else 16
    velocity *= ( (100 - math.pow(index, 1.0/3)) / 100 )    
    return velocity

def compare_hull_design(npv_result, initial_engine_id, initial_propeller_id):
    x_label    = 'NPV [$]'
    y_label    = 'frequency'.upper()
    title      = "NPV Distribution of design"
    data_frame = {}
    color_dict = {'A': 'r', 'B': 'b'}
    for c_key, values in npv_result.items():
        hull_id, engine_id, propeller_id = get_component_ids_from_design_key(c_key)
        if ( int(engine_id) == initial_engine_id) and (int(propeller_id) == initial_propeller_id) :
            hull_name = 'A' if int(hull_id) == 1 else 'B'
            data_frame[hull_name] = values
            
    # consider delta
    '''
    draw_data = []
    for index, val in enumerate(data_frame['A']):
        delta = val - data_frame['B'][index]
        draw_data.append(delta)
    # draw delta histgram
    delta_frame = pd.DataFrame(draw_data)
    delta_frame.hist()
    plt.ylim([0, 15])
    plt.xlim([-30000000, 30000000])
    # draw origin line
    plt.axvline(linewidth=1.7, color="k")
    plt.savefig('../delta.png')
    plt.clf()
    '''

    df = pd.DataFrame(data_frame, columns=data_frame.keys())
    

    graphInitializer(title, x_label, y_label)    
    fig, ax = plt.subplots()

    a_heights, a_bins = np.histogram(df['A'])
    b_heights, b_bins = np.histogram(df['B'], bins=a_bins)
    width = (a_bins[1] - a_bins[0])/3

    ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue')
    ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen')

    # hist
    filepath = "./hist_case.png"
    #panda_frame.plot(kind='hist', alpha=.5, bins=10)
    plt.ylim([0, 12])
    plt.legend(shadow=True)
    plt.legend(loc='upper right')
    plt.savefig(filepath)
    plt.clf()
    plt.close()

    return

def output_ratio_dict_sub():
    sub_ratio_dict = { 1: {'A': 518652960.15,'B': 519246707.05},
                       2: {'A': 605747467.48,'B': 604835404.62},
                       3: {'A': 604789731.48,'B': 601112254.09}}
    data_labels = ['1', '2-1', '2-2']
    draw_data   = []
    hull_various = ['A', 'B']    
    display_range = range(1,4)
    for case_key in display_range:
        draw_element = []
        for hull_key in hull_various:
            draw_element.append(sub_ratio_dict[case_key][hull_key])
        draw_data.append(draw_element)
        df2 = pd.DataFrame(draw_data, columns=hull_various)
    df2.plot(kind='bar', alpha=0.8, color=['r', 'b'])
    plt.ylim([0, 700000000])
    plt.ylabel('Averaged NPV [$]', fontweight="bold")
    #plt.xticks(np.array(display_range)-0.35, ["Case %d" % (_d) for _d in display_range], rotation=0, fontsize=20)
    plt.xticks(np.array(display_range)-0.35, ["Case %s" % (_d) for _d in data_labels], rotation=0, fontsize=20)    
    
    plt.savefig('../result.png')
    return

def output_ratio_dict_sub_for_sig_kst():
    sub_ratio_dict = { 1: {'A': 605747467.48,'B': 604835404.62},
                       2: {'A': 604789731.48,'B': 601112254.09}}
    data_labels = ['1-1', '1-2']
    draw_data   = []
    hull_various = ['A', 'B']    
    display_range = range(1,3)
    for case_key in display_range:
        draw_element = []
        for hull_key in hull_various:
            draw_element.append(sub_ratio_dict[case_key][hull_key])
        draw_data.append(draw_element)
        df2 = pd.DataFrame(draw_data, columns=hull_various)
    df2.plot(kind='bar', alpha=0.8, color=['r', 'b'])
    plt.ylim([0, 700000000])
    plt.ylabel('Averaged NPV [$]', fontweight="bold")
    #plt.xticks(np.array(display_range)-0.35, ["Case %d" % (_d) for _d in display_range], rotation=0, fontsize=20)
    plt.xticks(np.array(display_range)-0.35, ["Case %s" % (_d) for _d in data_labels], rotation=0, fontsize=20)    
    
    plt.savefig('../result.png')
    return

def generate_operation_date(start_month, operation_end_date=None):
    operation_date_array = np.array([])
    input_start_month = str_to_datetime(start_month) if isinstance(start_month, str) else start_month
    operation_start_date = first_day_of_month(input_start_month)
    if operation_end_date is None:
        operation_end_date   = add_year(operation_start_date, OPERATION_DURATION_YEARS)

    current_date = operation_start_date
    while True:
        operation_date_array = np.append(operation_date_array, current_date)
        current_date += datetime.timedelta(days=1)
        if current_date >= operation_end_date:
            break
    return operation_date_array

def display_sorted_result(data, n):
    dt   = np.dtype({'names': ('c_key','npv','std'),
                     'formats': ('S10', np.float, np.float)})
    data = np.array([ (_k, _v['npv'], _v['std'] ) for _k, _v in data.items()], dtype=dt)

    # NPV descending
    sorted_data = data[data['npv'].argsort()[-n:][::-1]]
    print "---------- %20s ----------" % ('NPV descending order')
    print "%20s %10s %10s" % ('combination key', 'NPV', 'STD')
    for _d in sorted_data:
        key, npv, std = _d
        print "%20s %10.2lf %10.2lf" % (key, npv, std)
    
    # STD ascending
    sorted_data = data[data['std'].argsort()[:n]]
    print "---------- %20s ----------" % ('STD ascending order')
    print "%20s %10s %10s" % ('combination key', 'NPV', 'STD')
    for _d in sorted_data:
        key, npv, std = _d
        print "%20s %10.2lf %10.2lf" % (key, npv, std)
    return

def calc_simple_oilprice(length, index, constant, ratio):
    # oilprice = ( c(t-1) / n )*x + c
    if length == 0:
        return 0
    oilprice = ( constant * (ratio-1) / length ) * index + constant
    return oilprice

def generate_significant_modes(oilprice_mode, 
                               oil_price_history_data, 
                               world_scale_history_data, 
                               flat_rate_history_data):
    from sinario     import Sinario
    from world_scale import WorldScale
    from flat_rate   import FlatRate
    sinario      = Sinario(oil_price_history_data)
    world_scale  = WorldScale(world_scale_history_data)
    flat_rate    = FlatRate(flat_rate_history_data)

    if oilprice_mode == 'oilprice_medium':
        # set modes
        world_scale_mode = 'medium'
        flat_rate_mode   = 'medium'
        significant_oilprice = significant_world_scale = significant_flat_rate = None
    elif oilprice_mode == 'oilprice_low':
        # set modes
        world_scale_mode = 'high'
        flat_rate_mode   = 'high'
        significant_oilprice = np.min(oil_price_history_data['price']) / 2.0
        # calc rate
        rate = significant_oilprice / oil_price_history_data['price'][-1]
        significant_world_scale = min(world_scale_history_data['ws'][-1] * (1/rate/3), 150)
        significant_flat_rate   = min(flat_rate_history_data['fr'][-1] * (1/rate/3), 100)
    elif oilprice_mode == 'oilprice_high':
        # set modes
        world_scale_mode = 'low'
        flat_rate_mode   = 'low'
        significant_oilprice = np.max(oil_price_history_data['price']) * 2.0
        # calc rate
        rate = significant_oilprice / oil_price_history_data['price'][-1]
        significant_world_scale = max(world_scale_history_data['ws'][-1] * (3/rate), 20)
        significant_flat_rate = min(flat_rate_history_data['fr'][-1] * (3/rate), 100)

    # generate sinario
    sinario.generate_significant_sinario(oilprice_mode, significant_oilprice)
    world_scale.generate_significant_sinario(world_scale_mode, significant_world_scale)
    flat_rate.generate_significant_flat_rate(flat_rate_mode, significant_flat_rate)


    return sinario, world_scale, flat_rate

def generate_final_significant_modes(oilprice_mode, 
                                    oil_price_history_data, 
                                    world_scale_history_data, 
                                    flat_rate_history_data):
    from sinario     import Sinario
    from world_scale import WorldScale
    from flat_rate   import FlatRate
    sinario      = Sinario(oil_price_history_data)
    world_scale  = WorldScale(world_scale_history_data)
    flat_rate    = FlatRate(flat_rate_history_data)

    # oilprice
    significant_high_oilprice_index = oil_price_history_data[np.argmax(oil_price_history_data['price'])]
    significant_high_oilprice       = significant_high_oilprice_index['price'] * MULTIPLY_INDEX
    significant_low_oilprice_index  = oil_price_history_data[np.argmin(oil_price_history_data['price'])]
    significant_low_oilprice        = significant_low_oilprice_index['price']
    # world_scale
    significant_high_world_scale_index = search_near_index(str_to_date(significant_high_oilprice_index['date']), world_scale_history_data['date'])
    significant_high_world_scale       = world_scale_history_data[np.where(world_scale_history_data['date']==significant_high_world_scale_index)[0]]['ws'][0] * DEVIDE_INDEX
    significant_low_world_scale_index  = search_near_index(str_to_date(significant_low_oilprice_index['date']), world_scale_history_data['date'])
    significant_low_world_scale        = world_scale_history_data[np.where(world_scale_history_data['date']==significant_low_world_scale_index)[0]]['ws'][0]
    # flat_rate
    significant_high_flat_rate_index = search_near_index(str_to_date(significant_high_oilprice_index['date']), flat_rate_history_data['date'])
    significant_high_flat_rate       = flat_rate_history_data[np.where(flat_rate_history_data['date']==significant_high_flat_rate_index)[0]]['fr'][0]                
    significant_low_flat_rate_index  = search_near_index(str_to_date(significant_low_oilprice_index['date']), flat_rate_history_data['date'])
    significant_low_flat_rate        = flat_rate_history_data[np.where(flat_rate_history_data['date']==significant_low_flat_rate_index)[0]]['fr'][0]            

    if oilprice_mode == 'oilprice_low':
        # set modes
        significant_oilprice    = significant_low_oilprice
        significant_world_scale = significant_low_world_scale
        significant_flat_rate   = significant_low_flat_rate        
    elif oilprice_mode == 'oilprice_high':
        # set modes
        significant_oilprice    = significant_high_oilprice
        significant_world_scale = significant_high_world_scale
        significant_flat_rate   = significant_high_flat_rate
    elif oilprice_mode == 'oilprice_dec':
        # set modes
        significant_oilprice    = [significant_high_oilprice, significant_low_oilprice]
        significant_world_scale = [significant_high_world_scale, significant_low_world_scale]
        significant_flat_rate   = [significant_high_flat_rate, significant_low_flat_rate]        
    elif oilprice_mode == 'oilprice_inc':
        # set modes
        significant_oilprice    = [significant_low_oilprice, significant_high_oilprice]
        significant_world_scale = [significant_low_world_scale, significant_high_world_scale]
        significant_flat_rate   = [significant_low_flat_rate, significant_high_flat_rate]        
        
    # generate sinario
    sinario.generate_significant_sinario(oilprice_mode, significant_oilprice)
    world_scale.generate_significant_sinario(oilprice_mode, significant_world_scale)
    flat_rate.generate_significant_flat_rate(oilprice_mode, significant_flat_rate)

    return sinario, world_scale, flat_rate

def count_whole_designs():
    # import models #
    from hull        import Hull
    from engine      import Engine
    from propeller   import Propeller
    # import models #
    # load components list
    hull_list           = load_hull_list()
    engine_list         = load_engine_list()
    propeller_list      = load_propeller_list()
    return hull_list.size * engine_list.size * propeller_list.size
