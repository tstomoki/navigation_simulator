# import common modules #
import time
import sys
from operator import itemgetter
from pdb import *
import matplotlib
# server configuration #
import getpass
current_user = getpass.getuser()
if current_user == 'tsaito':
    matplotlib.use('Agg')
# server configuration #
from optparse import OptionParser
# import common modules #
# import own modules #
sys.path.append('../public')
from my_modules import *
from constants  import *
from cubic_module import *
# import own modules #

# import models #
from hull        import Hull
from sinario     import Sinario
from engine      import Engine
from propeller   import Propeller
from flat_rate   import FlatRate
from world_scale import WorldScale
# import models #


def run(options):
    result_dir_path = options.result_dir_path
    if options.aggregate:
        aggregate_output(result_dir_path)
    if options.significant:
        aggregate_significant_output(result_dir_path)
    if options.fuel_cost:
        draw_fuel_cost(result_dir_path)
    if options.retrofit:
        draw_retrofit_result(result_dir_path)
    sys.exit()        

    draw_hull_features()
    #draw_propeller_features()
    draw_engine_features()
    #aggregate_results(result_dir_path)
    draw_ct_fn()
    draw_ct_fn(True, 'BF6')
    return

def draw_fuel_cost(result_dir_path):
    optimal_design_key_low  = "H1E1P514"
    optimal_design_key_high = "H2E3P514"
    if result_dir_path is None:
        print 'No result_dir_path. abort'
        return
    target_dirs      = os.listdir(result_dir_path)
    target_dirs      = [ d for d in target_dirs if (d != 'aggregated_results') and (d[-4:] != 'xlsx')]
    output_file_path = "%s/fuel_cost_transition.png" % (result_dir_path)

    # for csv import
    # initialize
    dt   = np.dtype({'names': ('date','fuel_cost'),
                     'formats': ('S10', np.float)})
    # draw graph
    title       = "fuel cost".title()
    x_label     = "date".upper()
    y_label     = "fuel cost [USD]".upper()
    line_colors = {'low': {'high': 'k', 'low':'r'}, 'high': {'high': 'b', 'low':'g'}}
    line_styles = {'low': {'high': '-', 'low':'--'}, 'high': {'high': '-.', 'low':':'}}
    
    for target_dir in target_dirs:
        desti_dir = "%s/%s" % (result_dir_path,
                               target_dir)
        if os.path.exists(desti_dir):
            oilprice_mode    = re.compile(r'oilprice_(.+)').search(target_dir)
            if oilprice_mode is None:
                continue
            else:
                oilprice_mode = oilprice_mode.groups()[0]
            npv_result       = {}
            fuel_cost_result = {}
            files            = os.listdir(desti_dir)
            # for optimal high and design
            for mode in ['low', 'high']:
                # for low design
                optimal_design_key      = eval("optimal_design_key_%s" % (mode))
                optimal_design_dir_path = "%s/%s" % (desti_dir, optimal_design_key)
                fuel_file_path          = "%s/fuel_cost.csv" % (optimal_design_dir_path)
                data                    = np.genfromtxt(fuel_file_path,
                                                        delimiter=',',
                                                        dtype=dt,
                                                        skiprows=1)
                draw_data = [ [datetime.datetime.strptime(_d['date'], '%Y/%m/%d'), _d['fuel_cost']] for _d in data]
                draw_data = np.array(sorted(draw_data, key= lambda x : x[0]))
                draw_label = "%s(at %s oilprice)" % (optimal_design_key, oilprice_mode)
                plt.plot(draw_data.transpose()[0],
                         draw_data.transpose()[1],
                         label=draw_label,
                         color=line_colors[oilprice_mode][mode], linestyle=line_styles[oilprice_mode][mode])

    start_date = draw_data.transpose()[0].min()
    end_date   = start_date + datetime.timedelta(days=365)
    days_delta = (end_date - start_date).days
    title = "%s (%d days)\n" % (title, days_delta)
    plt.title(title,          fontweight="bold")
    plt.xlabel(x_label,       fontweight="bold")
    plt.ylabel(y_label,       fontweight="bold")    
    plt.legend(shadow=True)
    plt.legend(loc='upper right')
    plt.axhline(linewidth=1.0, color='k')
    #plt.xlim([draw_data.transpose()[0].min(), draw_data.transpose()[0].max()])
    plt.xlim(start_date, end_date)
    plt.ylim(0, 170000)
    plt.savefig(output_file_path)
    return 

def aggregate_results(result_dir_path):
    whole_result = {}

    for filename in os.listdir(result_dir_path):
        try:
            mode_str  = re.compile(r'(mode.+).json').search(filename).groups()[0]
            result_file_path = "%s/%s" % (result_dir_path, filename)
            whole_result[mode_str] = load_json_file(result_file_path)
        except:
            print "file %20s is ignored" % (filename)

    # display delta between Hull A and B
    for target_mode in sorted(whole_result.keys()):
        delta_array = {}
        for c_key, values in whole_result[target_mode].items():
            hull_id, engine_id, propeller_id = get_component_ids_from_design_key(c_key)
            if hull_id == '1':
                target_c_key = generate_combination_str_with_id(2, engine_id, propeller_id)
                delta = values['npv'] - whole_result[target_mode][target_c_key]['npv']
                delta_array[c_key] =delta
        average_delta = np.average(delta_array.values())
        print "%10s: %20.4lf" % (target_mode, average_delta)
        
    '''
    # display delta between mode1 and mode2
    delta_array = {}
    for c_key, values in whole_result['mode1'].items():
        target_val         = whole_result['mode2'][c_key]
        delta_array[c_key] = values['npv'] - target_val['npv']
    '''
    return

def draw_ct_fn(wave=None, beaufort=None):
    hull_list = load_hull_list()
    draw_data = dict.fromkeys(LOAD_CONDITION.values())
    colors = {0: 'r', 1:'b'}
    for load_condition_num, load_condition in LOAD_CONDITION.items():
        draw_data[load_condition] = {}        
        for hull_info in hull_list:
            hull_id                            = hull_info['id']
            draw_data[load_condition][hull_id] = {}
            hull                               = Hull(hull_list, hull_id)    
            # v range [knot]
            v_range = np.linspace(0, 25, 1000)
            f_v_dict = {}

            for _v in v_range:
                froude = hull.calc_froude(_v)
                f_v_dict[froude] = _v

            for _f, _v in f_v_dict.items():
                # reduce for wave                
                if wave is not None:
                    current_wave_height = get_wave_height(beaufort)                    
                    delta_v = calc_y(current_wave_height, [V_DETERIO_FUNC_COEFFS['cons'], V_DETERIO_FUNC_COEFFS['lin'], V_DETERIO_FUNC_COEFFS['squ']], V_DETERIO_M)
                    # reduce for bow
                    _v += hull.consider_bow_for_wave(delta_v, load_condition)
                modified_v = consider_bow_for_v(hull, _v, load_condition_num)
                ehp = hull.calc_raw_EHP(modified_v, load_condition)
                ct  = hull.calc_ct(ehp, modified_v, load_condition)
                if ct < 2.0:
                    draw_data[load_condition][hull_id][_f] = ct

    for load_condition, value_array in draw_data.items():
        # draw part
        title     = "bow characteristics".title()
        title     = title if (wave is None) else "%s (%s)" % (title, beaufort)
        x_label   = "Fn"
        y_label   = "Ct (V)"
        graphInitializer(title, x_label, y_label)
        for hull_id, values in value_array.items():
            draw_array = np.array(values.items())
            draw_array = np.sort(draw_array, axis=0)
            x_data     = draw_array.transpose()[0]
            y_data     = draw_array.transpose()[1]
            label      = "hull%d" % (hull_id)
            if Hull(hull_list, hull_id).base_data['with_bow'] == 'TRUE':
                label = "%s (BOW)" % (label)
            plt.plot(x_data, y_data, label=label, color=colors[hull_id - 1])
        plt.legend(shadow=True)
        plt.legend(loc='upper left')
        condition = load_condition if (wave is None) else "%s_wave" % (load_condition)
        file_name = "%s/ct_%s.png" % (GRAPH_DIR_PATH, condition)
 
        plt.xlim(0.05, 0.2)
        plt.ylim(0, 2)
        plt.savefig(file_name)
        plt.clf()
    return

def draw_engine_features():
    engines = []
    engine_list   = load_engine_list()
    for engine_info in engine_list:
        engine_id = engine_info['id']
        engine    = Engine(engine_list, engine_id)
        engines.append(engine)

    # draw graph
    title   = "Engine features".title()
    x_label = "rpm".upper()
    y_label = "BHP [kW]"    
    graphInitializer(title, x_label, y_label)
    line_colors = ['k', 'r', 'b', 'g']
    line_styles = ['-', '--', '-.', ':']
    for index, engine in enumerate(engines):
        if not engine.base_data['specific_name'] == 'Hiekata':
            draw_data = engine.generate_modified_bhp()
            label = "Engine %d (%s)" % (engine.base_data['id'], engine.base_data['specific_name'])
            plt.plot(draw_data['rpm'], draw_data['modified_bhp'], label=label, color=line_colors[index], linestyle=line_styles[index])
    plt.legend(shadow=True)
    plt.legend(loc='upper left')
    output_file_path = "%s/engine_features.png" % (GRAPH_DIR_PATH)
    plt.savefig(output_file_path)
    return

# WIP
def draw_propeller_features():
    propellers     = []
    propeller_list = load_propeller_list()
    for propeller_info in propeller_list:
        propeller_id = propeller_info['id']
        propeller    = Propeller(propeller_list, propeller_id)
        propellers.append(propeller)
    # draw graph
    title   = "Propeller features".title()
    x_label = "rpm".upper()
    y_label = "BHP [kW]"    
    graphInitializer(title, x_label, y_label)
    line_colors = ['k', 'r', 'b', 'g']
    line_styles = ['-', '--', '-.', ':']
    for index, engine in enumerate(engines):
        if not engine.base_data['specific_name'] == 'Hiekata':
            draw_data = engine.generate_modified_bhp()
            label = "Engine %d (%s)" % (engine.base_data['id'], engine.base_data['specific_name'])
            plt.plot(draw_data['rpm'], draw_data['modified_bhp'], label=label, color=line_colors[index], linestyle=line_styles[index])
    plt.legend(shadow=True)
    plt.legend(loc='upper left')
    output_file_path = "%s/engine_features.png" % (GRAPH_DIR_PATH)
    plt.savefig(output_file_path)
    return

def draw_hull_features():
    hulls     = []
    hull_list = load_hull_list()
    v_range   = np.linspace(0, 25, 1000)
    for hull_info in hull_list:
        hull_id = hull_info['id']
        hull    = Hull(hull_list, hull_id)
        hulls.append(hull)
    
    # draw graph
    title       = "Hull features".title()
    x_label     = "Froude number".upper()
    y_label     = "Ct"
    graphInitializer(title, x_label, y_label)
    line_colors = ['k', 'k', 'r', 'r']
    line_styles = ['-', '--', '-.', ':']
    dt          = np.dtype({'names': ('froude', 'ehp'),
                            'formats': (np.float, np.float)})
    for load_condition_num, load_condition in LOAD_CONDITION.items():
        for index, hull in enumerate(hulls):
            draw_data = []
            ## plot style
            sub_label   = "(%s, BOW)" % (load_condition) if hull.bow_exists() else "(%s)" % (load_condition)
            label       = "Hull %s" % (sub_label)
            style_index = (index * 2) + load_condition_num
            f_v_dict = {}
            for _v in v_range:
                modified_v       = hull.consider_bow_for_v(_v, load_condition_num)
                froude           = hull.calc_froude(modified_v)
                f_v_dict[froude] = _v
            for _f, _v in f_v_dict.items():
                ehp = hull.calc_raw_EHP(_v, load_condition)
                ct  = hull.calc_ct(ehp, _v, load_condition)
                draw_data.append( (_f, ct))
            draw_data   = np.array(sorted(draw_data, key=lambda x : x[0]), dtype=dt)
            plt.plot(draw_data['froude'], draw_data['ehp'], label=label, color=line_colors[style_index], linestyle=line_styles[style_index])
    plt.legend(shadow=True)
    plt.legend(loc='upper left')
    plt.xlim(0.05, 0.25)
    plt.ylim(0, 4)    
    output_file_path = "%s/hull_features.png" % (GRAPH_DIR_PATH)
    plt.savefig(output_file_path)
    return

def aggregate_output(result_dir_path):
    initial_design_dirs = ['initial_design_mode' + str(i) for i in range(3)]

    # initialize
    dt   = np.dtype({'names': ('scenario_num','hull_id','engine_id','propeller_id','NPV'),
                     'formats': (np.int64, np.int64, np.int64, np.int64, np.float, np.float)})
    # for csv
    column_names = ['scenario_total_num',
                    'design_key',
                    'hull_id',
                    'engine_id',
                    'propeller_id',
                    'average NPV',
                    'std']
    for initial_design_dir in initial_design_dirs:
        target_dir = "%s/%s" % (result_dir_path,
                                initial_design_dir)
        if os.path.exists(target_dir):
            npv_result = {}
            files = os.listdir(target_dir)
            target_files = [_f for _f in files if _f[-4:] == '.csv' and not _f == 'initial_design.csv']
            target_files = [ "%s/%s" % (target_dir, _f) for _f in target_files]
            for _target_file in target_files:
                data = np.genfromtxt(_target_file,
                                     delimiter=',',
                                     dtype=dt,
                                     skiprows=1)
                for _d in data:
                    s_num, h_id, e_id, p_id, npv = _d
                    combination_key = generate_combination_str_with_id(h_id, e_id, p_id)
                    if not npv_result.has_key(combination_key):
                        npv_result[combination_key] = []
                    npv_result[combination_key].append(npv)    
            # output_csv
            senario_total_num = len(unique(data['scenario_num']))
            output_dir_path = "%s/aggregated_results" % (result_dir_path)
            initializeDirHierarchy(output_dir_path)
            output_file_path = "%s/%s.csv" % (output_dir_path,
                                              initial_design_dir)

            for design_key, npvs in npv_result.items():
                hull_id, engine_id, propeller_id = get_component_ids_from_design_key(design_key)
                write_csv(column_names, [senario_total_num,
                                         design_key,
                                         hull_id,
                                         engine_id,
                                         propeller_id,
                                         np.average(npvs),
                                         np.std(npvs)
                                         ], output_file_path)
    return

def aggregate_significant_output(result_dir_path):
    target_dirs = os.listdir(result_dir_path)
    target_dirs = [ d for d in target_dirs if (d != 'aggregated_results') and ('.' not in d)]

    # initialize
    dt   = np.dtype({'names': ('hull_id','engine_id','propeller_id','NPV', 'fuel_cost'),
                     'formats': (np.int64, np.int64, np.int64, np.float, np.float)})
    # for csv
    column_names = ['design_key',
                    'hull_id',
                    'engine_id',
                    'propeller_id',
                    'average NPV',
                    'fuel cost']
    result_dict = {}
    for target_dir in target_dirs:
        desti_dir = "%s/%s" % (result_dir_path,
                               target_dir)
        if os.path.exists(desti_dir):
            # draw velocity logs
            draw_velocity_logs(desti_dir, target_dir)
            # draw velocity logs

            npv_result       = {}
            fuel_cost_result = {}
            files = os.listdir(desti_dir)
            target_files = [_f for _f in files if _f[-4:] == '.csv' and not _f == 'initial_design.csv']
            target_files = [ "%s/%s" % (desti_dir, _f) for _f in target_files]
            whole_design_nums = count_whole_designs()
            for _target_file in target_files:
                data = np.genfromtxt(_target_file,
                                     delimiter=',',
                                     dtype=dt,
                                     skiprows=0)
                if data.ndim == 0:
                    data = np.atleast_1d(data)
                for _d in data:
                    h_id, e_id, p_id, npv, fuel_cost = _d
                    combination_key = generate_combination_str_with_id(h_id, e_id, p_id)
                    if not npv_result.has_key(combination_key):
                        npv_result[combination_key] = npv
                    if not fuel_cost_result.has_key(combination_key):
                        fuel_cost_result[combination_key] = fuel_cost
            maximum_key = max(npv_result.items(), key=itemgetter(1))[0]
            result_dict[target_dir] = [maximum_key, npv_result[maximum_key], len(npv_result.keys()) / float(whole_design_nums)]
                    
            # output_csv
            output_dir_path = "%s/%s/aggregated_results" % (result_dir_path, target_dir)
            initializeDirHierarchy(output_dir_path)
            output_file_path = "%s/%s.csv" % (output_dir_path,
                                              target_dir)
            if os.path.exists(output_file_path):
                os.remove(output_file_path)

            #draw_npv_histgram(npv_result, target_dir, output_dir_path)
            for design_key, npvs in npv_result.items():
                hull_id, engine_id, propeller_id = get_component_ids_from_design_key(design_key)
                write_csv(column_names, [design_key,
                                         hull_id,
                                         engine_id,
                                         propeller_id,
                                         npvs,
                                         fuel_cost_result[design_key],
                                         ], output_file_path)
    print "%20s %20s %10s %25s" % ('scenario_mode', 'design_key', 'NPV', 'progress')
    print "-" * 90
    for k,v in result_dict.items():
        print "%20s %20s %17.3lf %18.2lf[%%]" % (k, v[0], v[1], v[2]*100)
    return

def draw_retrofit_result(result_dir_path):
    if result_dir_path is None:
        print 'No result_dir_path. abort'
        return
    target_dirs     = os.listdir(result_dir_path)
    base_design_key = None

    for target_dir in target_dirs:
        # initialize
        dt   = np.dtype({'names': ('date','npv'),
                         'formats': ('S10', np.float)})        
        # npv comparison
        index_num     = 18
        retrofit_date = datetime.datetime(2021, 4, 12)
        # draw graph
        title       = "NPV".title()
        x_label     = "date".upper()
        y_label     = "PV [USD]".upper()
        ## for flexible
        desti_dir   = "%s/%s/flexible" % (result_dir_path, target_dir)
        if os.path.exists(desti_dir):
            # calc average npv from initial_designs
            files        = os.listdir(desti_dir)
            target_file = [_f for _f in files if _f[:1] == 'H'][-1]
            target_file_path = "%s/%s/simulate%d/npv.csv" % (desti_dir, target_file, index_num)
            if os.path.exists(target_file_path):
                data                    = np.genfromtxt(target_file_path,
                                                        delimiter=',',
                                                        dtype=dt,
                                                        skiprows=1)
                draw_data = [ [datetime.datetime.strptime(_d['date'], '%Y/%m/%d'), _d['npv']] for _d in data]
                draw_data = np.array(sorted(draw_data, key= lambda x : x[0]))
                #draw_data = np.array([_d for _d in sorted(draw_data, key= lambda x : x[0]) if _d[0] >= retrofit_date])
                # for retrofit date
                plt.axvline(x=retrofit_date, color='k', linewidth=4, linestyle='--')                
                draw_label = "%s (Flexible)" % (target_file)
                plt.plot(draw_data.transpose()[0],
                         draw_data.transpose()[1],
                         label=draw_label,
                         color='r', linestyle='-')
                '''
                print 'flexible'
                print sum(draw_data.transpose()[1])
                '''
        # for no retrofit
        desti_dir   = "%s/%s/no_retrofit" % (result_dir_path, target_dir)
        if os.path.exists(desti_dir):
            # calc average npv from initial_designs
            files        = os.listdir(desti_dir)
            target_file = [_f for _f in files if _f[:1] == 'H'][-1]
            target_file_path = "%s/%s/simulate%d/npv.csv" % (desti_dir, target_file, index_num)
            if os.path.exists(target_file_path):
                data                    = np.genfromtxt(target_file_path,
                                                        delimiter=',',
                                                        dtype=dt,
                                                        skiprows=1)
                draw_data = [ [datetime.datetime.strptime(_d['date'], '%Y/%m/%d'), _d['npv']] for _d in data]
                draw_data = np.array(sorted(draw_data, key= lambda x : x[0]))
                plt.plot(draw_data.transpose()[0],
                         draw_data.transpose()[1],
                         label=target_file,
                         color='g', linestyle='--')
                base_design_key = target_file
                '''
                print 'no_retrofit'
                print sum(draw_data.transpose()[1])
                '''
                # debug
                # for first dock-in
                #plt.axvline(x=draw_data.transpose()[0][0] + datetime.timedelta(days=365*DOCK_IN_PERIOD), color='r', linewidth=4, linestyle='--')
                plt.legend(shadow=True)
                plt.legend(loc='upper left')
        #debug
        #plt.xlim([draw_data.transpose()[0].min(), draw_data.transpose()[0].max()])        
        plt.savefig("%s/%s/simulate%d.png" % (result_dir_path, target_dir, index_num))
        plt.close()

        # initialize
        dt   = np.dtype({'names': ('simulation_time', 'hull_id','engine_id','propeller_id','NPV', 'fuel_cost'),
                         'formats': (np.int64, np.int64, np.int64, np.int64, np.float, np.float)})        
        column_names = ["simulation_time",
                        "hull_id",
                        "engine_id",
                        "propeller_id",
                        "NPV",
                        "fuel_cost"]
        combination_key = None
        # for flexible
        desti_dir = "%s/%s/flexible" % (result_dir_path, target_dir)
        flexible_npv_result = {}
        if os.path.exists(desti_dir):
            # calc average npv from initial_designs
            files               = os.listdir(desti_dir)
            target_files        = [_f for _f in files if _f[-4:] == '.csv' and not _f == 'initial_design.csv']
            target_files        = [ "%s/%s" % (desti_dir, _f) for _f in target_files]
            flexible_npv_result = {}
            for target_file in target_files:
                data = np.genfromtxt(target_file,
                                     delimiter=',',
                                     dtype=dt,
                                     skiprows=1)
                if data.ndim == 0:
                    data = np.atleast_1d(data)
                for _d in data:
                    s_num, h_id, e_id, p_id, npv, fuel_cost = _d
                    combination_key = generate_combination_str_with_id(h_id, e_id, p_id)
                    if not flexible_npv_result.has_key(s_num):
                        flexible_npv_result[s_num] = []
                    flexible_npv_result[s_num] = npv
            base_design_key = base_design_key if base_design_key is not None else 'Base'
            print "%s -> %s (%s, flexible): \n %10s: %20lf\n %10s: %20lf (from %d results)" % (base_design_key,
                                                                                               combination_key,
                                                                                               target_dir,
                                                                                               'ave. NPV',
                                                                                               np.average(flexible_npv_result.values()),
                                                                                               'std.', np.std(flexible_npv_result.values()),
                                                                                               len(flexible_npv_result.values()))
        # for no retrofit
        desti_dir = "%s/%s/no_retrofit" % (result_dir_path, target_dir)
        if os.path.exists(desti_dir):
            # calc average npv from initial_designs
            files        = os.listdir(desti_dir)
            target_files = [_f for _f in files if _f[-4:] == '.csv' and not _f == 'initial_design.csv']
            target_files = [ "%s/%s" % (desti_dir, _f) for _f in target_files]
            npv_result   = {}
            for target_file in target_files:
                data = np.genfromtxt(target_file,
                                     delimiter=',',
                                     dtype=dt,
                                     skiprows=1)
                if data.ndim == 0:
                    data = np.atleast_1d(data)                
                for _d in data:
                    s_num, h_id, e_id, p_id, npv, fuel_cost = _d                    
                    combination_key = generate_combination_str_with_id(h_id, e_id, p_id)
                    if not npv_result.has_key(s_num):
                        npv_result[s_num] = []
                    npv_result[s_num] = npv                    
            print "%s (%s): \n %10s: %20lf\n %10s: %20lf (from %d results)" % (combination_key,
                                                                               target_dir,
                                                                               'ave. NPV',
                                                                               np.average(npv_result.values()),
                                                                               'std.', np.std(npv_result.values())
                                                                               , len(npv_result.values()))

        print "\n\n\n\n%40s" % ("details".upper())
        print "%10s %20s %10s %20s %20s %20s" % ('sim'.upper(), 'flexible'.upper(), 'sim'.upper(), 'no retrofit'.upper(), 'judgement'.upper(), 'delta'.upper())
        iterate_count = max(len(flexible_npv_result), len(npv_result))
        flexibles     = [ [_k, _v] for _k, _v in flexible_npv_result.items()]
        no_retrofits  = [ [_k, _v] for _k, _v in npv_result.items()]
        delta_array   = []
        for index in range(iterate_count):
            if len(flexibles) > index:
                flexible    = [ flexibles[index][0], "%17.3lf" % (flexibles[index][1])]
            else:
                flexible    = ['-', '-']
            
            if len(no_retrofits) > index:
                no_retrofit = no_retrofits[index]
                no_retrofit = [ no_retrofits[index][0], "%17.3lf" % (no_retrofits[index][1])]
            else:
                no_retrofit = ['-', '-']

            if (flexible[1] != '-' and no_retrofit[1] != '-'):
                judgement = 'flexible' if flexible[1] > no_retrofit[1] else 'no_retrofit'
                delta     = "%17.3lf" % (float(flexible[1])-float(no_retrofit[1]))
            else:
                judgement = '-'
                delta     = '-'
            #if judgement == 'flexible':
            delta_array.append(delta)
            print "%10s %20s %10s %20s %20s %20s" % (str(flexible[0]), str(flexible[1]),
                                                     str(no_retrofit[0]), str(no_retrofit[1]), judgement, delta)
        delta_array = map(float, [_d for _d in delta_array if _d != '-'])
        print 'ave. delta %20lf' % (np.average(map(float, delta_array)))
        print 'ave. std %20lf' % (np.std(map(float, delta_array)))
    return

def draw_npv_histgram(npv_result, oilprice_mode, output_dir_path):
    # draw graph
    title    = "%s (at %s)" % ("NPV for each design".upper(), "oilprice_low".replace('_', ' '))
    x_label  = "design id".upper()
    y_label  = "PV [USD]".upper()
    filepath = "%s/%s.png" % (output_dir_path, oilprice_mode)
    dt       = np.dtype({'names': ('design_id','npv'),
                     'formats': ('S10', np.float)})    
    graphInitializer(title, x_label, y_label)
    draw_data = np.array(sorted([(k, v[0]) for k,v in npv_result.items()], key=lambda x : x[1], reverse=True), dtype=dt)
    ticks     = { i:_d for i, _d in enumerate(draw_data['design_id'])}
    plt.bar( [ _i + 1 for _i in ticks.keys()], draw_data['npv'])
    plt.xticks( [_i + 1 for _i in ticks.keys()], ticks.values(), rotation=50, fontsize=12)
    plt.savefig(filepath)
    plt.close()    
    
    return

def draw_velocity_logs(result_dir, oilprice_mode):
    velocity_logs_dir = "%s/velocity_logs" % (result_dir)
    if os.path.exists(velocity_logs_dir):
        files        = os.listdir(velocity_logs_dir)
        target_files = [_f for _f in files if _f[-4:] == '.csv' and not _f == 'initial_design.csv']
        for load_condition in ['full', 'ballast']:
            # initialize graph
            title    = "%s (at %s)" % ("rpm and velocity for each design".upper(), oilprice_mode.replace('_', ' '))
            x_label  = "design id".upper()
            y0_label = "rpm".upper()
            y1_label = "velocity".upper() + '[knot]'
            filepath = "%s/velocity_%s_%s.png" % (velocity_logs_dir, oilprice_mode, load_condition)
            dt       = np.dtype({'names': ('date','rpm', 'velocity', 'load_condition'),
                                 'formats': ('S20', np.float, np.float, 'S10')})
            draw_data = []
            for target_file in target_files:
                design_str       = re.compile(r'(.+).csv').search(target_file).groups()[0]
                target_file_path = "%s/%s" % (velocity_logs_dir, target_file)
                data = np.genfromtxt(target_file_path,
                                     delimiter=',',
                                     dtype=dt,
                                     skiprows=1)
                target_data = data[np.where(data['load_condition']==load_condition)]
                average_rpm      = np.average(target_data['rpm'])
                average_velocity = np.average(target_data['velocity'])
                draw_data.append([design_str, average_rpm, average_velocity])
            draw_data = np.array(sorted(draw_data, key=lambda x : x[0]))
            draw_twin_graph(draw_data, title, x_label, y0_label, y1_label)
            plt.savefig(filepath)
    return
    

# authorize exeucation as main script
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-P", "--result-path", dest="result_dir_path",
                      help="results dir path", default=None)    
    parser.add_option("-A", "--aggregate", dest="aggregate",
                      help="aggregate mode", default=False)
    parser.add_option("-S", "--significant", dest="significant",
                      help="aggregate significant mode", default=False)    
    parser.add_option("-F", "--fuel-cost", dest="fuel_cost",
                      help="draw fuel cost mode", default=False)
    parser.add_option("-T", "--retrofit", dest="retrofit",
                      help="draw retrofit result", default=False)        
    (options, args) = parser.parse_args()
    run(options)    
