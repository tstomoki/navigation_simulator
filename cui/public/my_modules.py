# import common modules #
import math
import os
import sys
import pdb
import datetime
import numpy as np
import matplotlib.pyplot as plt
import calendar as cal
import random
import csv
import json
import operator
import re
# import common modules #

# import own modules #
sys.path.append('../public')
sys.path.append('../models')
from constants  import *
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
    for year_index in range(year_num):
        current_date = add_month(current_date, 12)
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
    notice_str = '*' * 80
    print "%80s" % (notice_str)
    print "%10s %s %10s" % ('*' * 10, display_str, '*' * 10)
    print "%80s" % (notice_str)
    return 

def datetime_to_human(datetime_var):
    return datetime.datetime.strftime(datetime_var, "%Y/%m/%d")

def detailed_datetime_to_human(datetime_var):
    return datetime.datetime.strftime(datetime_var, "%Y/%m/%d %H:%M:%S")

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
    file_path        = "%s/%s/%s_combinations.json" % (COMBINATIONS_DIR_PATH, combination_str, combination_str)

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
    return "H%dE%dP%d" % (hull_id, engine_id, propeller_id)

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

def change_design(design_key):
    # import models #
    from hull        import Hull
    from engine      import Engine
    from propeller   import Propeller
    # import models #
    # get component_ids
    p = re.compile(r'H(\d+)E(\d+)P(\d+)')
    try:
        a = p.search(design_key)
    except:
        pdb.set_trace()
    hull_id, engine_id, propeller_id = a.groups()

    # load components list
    hull_list           = load_hull_list()
    engine_list         = load_engine_list()
    propeller_list      = load_propeller_list()

    hull      = Hull(hull_list,           int(hull_id))
    engine    = Engine(engine_list,       int(engine_id))
    propeller = Propeller(propeller_list, int(propeller_id))
    return hull, engine, propeller

def update_retrofit_design_log(retrofit_design_log, combinations):
    for combination in combinations:
        combination_key = generate_combination_str_with_id(combination['hull_id'],
                                                           combination['engine_id'],
                                                           combination['propeller_id'])
        if not retrofit_design_log.has_key(combination_key):
            retrofit_design_log[combination_key] = np.array([])
        retrofit_design_log[combination_key] = np.append(retrofit_design_log[combination_key], combination['NPV'])
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
    current_combination_key = 'H1E2P1'
    current_design_NPV = np.average(retrofit_design_log[current_combination_key])
    plt.axhline(current_design_NPV, 
                xmin=0, xmax=len(retrofit_design_log),                    
                linewidth=1.5, color='r',
                linestyle="--")
    # draw other NPV
    draw_data     = np.array([])
    for_json_data = {'current_design': current_design_NPV, 'other_design': {}}
    for index, combination_key in enumerate(retrofit_design_log.keys()):
        # exclude current design
        if combination_key == current_combination_key:
            continue
        ave_npv       = np.average(retrofit_design_log[combination_key])
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
