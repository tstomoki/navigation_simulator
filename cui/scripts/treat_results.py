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
    dt   = np.dtype({'names': ('hull_id','engine_id','propeller_id','NPV', 'fuel_cost','avg_round_num','round_num'),
                     'formats': (np.int64, np.int64, np.int64, np.float, np.float, np.float, np.float)})
    # for csv
    column_names = ['design_key',
                    'hull_id',
                    'engine_id',
                    'propeller_id',
                    'average NPV',
                    'fuel cost',
                    'avg_round_num',
                    'round_num']
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
            round_result     = {}
            files = os.listdir(desti_dir)
            target_files = [_f for _f in files if _f[-4:] == '.csv' and not _f == 'initial_design.csv']
            target_files = [ "%s/%s" % (desti_dir, _f) for _f in target_files]
            whole_design_nums = count_whole_designs()
            for _target_file in target_files:
                data = np.genfromtxt(_target_file,
                                     delimiter=',',
                                     dtype=dt,
                                     skiprows=1)
                if data.ndim == 0:
                    data = np.atleast_1d(data)
                for _d in data:
                    h_id, e_id, p_id, npv, fuel_cost, avg_round_num, round_num = _d
                    combination_key = generate_combination_str_with_id(h_id, e_id, p_id)
                    if not npv_result.has_key(combination_key):
                        npv_result[combination_key] = npv
                    if not fuel_cost_result.has_key(combination_key):
                        fuel_cost_result[combination_key] = fuel_cost
                    if not round_result.has_key(combination_key):
                        round_result[combination_key] = round_num
            try:
                maximum_key             = max(npv_result.items(), key=itemgetter(1))[0]
                maximum_val             = npv_result[maximum_key]
                maximum_elements        = dict(sorted(npv_result.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])
                delta_array             = sorted([ (k, maximum_val - v) for k,v in maximum_elements.items() if not (maximum_val - v) == 0 ], key=lambda x : x[1])
                result_dict[target_dir] = [maximum_key, maximum_val, fuel_cost_result[maximum_key], round_result[maximum_key], len(npv_result.keys()) / float(whole_design_nums), [': '.join([v[0], str(v[1])]) for v in delta_array]]
            except:
                result_dict[target_dir] = ['--------', 0, 0, 0.0, 0.0, ['-'*40]]
                    
            # output_csv
            output_dir_path = "%s/%s/aggregated_results" % (result_dir_path, target_dir)
            initializeDirHierarchy(output_dir_path)
            output_file_path = "%s/%s.csv" % (output_dir_path,
                                              target_dir)
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
            draw_npv_histgram(npv_result, target_dir, output_dir_path)
            draw_fuel_cost_histgram(fuel_cost_result, target_dir, output_dir_path)
            draw_npv_fuel_twingraph(npv_result, fuel_cost_result, target_dir, output_dir_path)
            for design_key, npvs in npv_result.items():
                hull_id, engine_id, propeller_id = get_component_ids_from_design_key(design_key)
                write_csv(column_names, [design_key,
                                         hull_id,
                                         engine_id,
                                         propeller_id,
                                         npvs,
                                         round_result[design_key],
                                         fuel_cost_result[design_key],
                                         ], output_file_path)
    print "%20s %20s %10s %22s %15s %22s %15s %15s %15s" % ('scenario_mode', 'design_key', 'NPV','NPV(sig)', 'Fuel Cost', 'Fuel Cost(sig)', 'Round Num', 'progress', 'delta')
    print "-" * 90
    for k,v in result_dict.items():
        try:
            print "%20s %20s %17.3lf %15.3e %17.3lf %15.3e %15.0lf %18.2lf[%%] %30s" % (k, v[0], v[1], v[1], v[2], v[2], v[3], v[4]*100, "(%s)" % ','.join(v[5]))
        except:
            raise
            set_trace()
    return

def draw_retrofit_result(result_dir_path):
    if result_dir_path is None:
        print 'No result_dir_path. abort'
        return

    # get bf_mode
    bf_modes = os.listdir(result_dir_path)

    for bf_mode in bf_modes:
        target_dir_path = "%s/%s" % (result_dir_path, bf_mode)
        target_dirs     = os.listdir(target_dir_path)
        base_design_key = None
        print "%30s %s %30s\n" % ('-'*30, ("%s sea condition" % bf_mode).upper(), '-'*30)
        for target_dir in target_dirs:
            # initialize
            dt   = np.dtype({'names': ('simulation_time', 'hull_id','engine_id','propeller_id','NPV', 'fuel_cost', 'base_design', 'retrofit_design', 'retrofit_date'),
                             'formats': (np.int64, np.int64, np.int64, np.int64, np.float, np.float, 'S20', 'S20', 'S20')})        
            column_names = ["simulation_time",
                            "hull_id",
                            "engine_id",
                            "propeller_id",
                            "NPV",
                            "fuel_cost",
                            "retrofit_date"]
            print "%20s\n%15s\n%20s" % ('*'*20, target_dir.upper(), '*'*20)
            print "%20s" % ('-^-'*50)
            print "%20s\n%15s\n%20s" % ('-'*20, "statistical result".upper(), '-'*20)
            print "%15s %20s %20s %10s %21s %20s %20s" % ('design_type'.upper(), 
                                                               'avg. npv'.upper(),
                                                               'avg. npv (sig)'.upper(),
                                                               'std'.upper(),
                                                               'std (sig)'.upper(),
                                                               'simulation count'.upper(),
                                                               'retrofit occurs'.upper())
            combination_key = None
            # for flexible
            desti_dir = "%s/%s/flexible" % (target_dir_path, target_dir)
            flexible_result = []
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
                        flexible_result.append(_d)
                flexible_result = np.array(sorted(flexible_result, key=lambda x : x[0]))
                retrofit_count = len([_d for _d in flexible_result['retrofit_date'] if _d != '--'])
                print "%15s %24.3lf %13.3e %20.3lf %14.3e %15d %15d" % ('flexible', 
                                                                        np.average(flexible_result['NPV']),
                                                                        np.average(flexible_result['NPV']),
                                                                        np.std(flexible_result['NPV']),
                                                                        np.std(flexible_result['NPV']),
                                                                        len(flexible_result), retrofit_count)
            # for no retrofit
            desti_dir = "%s/%s/no_retrofit" % (target_dir_path, target_dir)
            if os.path.exists(desti_dir):
                # calc average npv from initial_designs
                files        = os.listdir(desti_dir)
                target_files = [_f for _f in files if _f[-4:] == '.csv' and not _f == 'initial_design.csv']
                target_files = [ "%s/%s" % (desti_dir, _f) for _f in target_files]
                result       = []
                for target_file in target_files:
                    data = np.genfromtxt(target_file,
                                         delimiter=',',
                                         dtype=dt,
                                         skiprows=1)
                    if data.ndim == 0:
                        data = np.atleast_1d(data)                
                    for _d in data:
                        result.append(_d)
                result = np.array(sorted(result, key=lambda x : x[0]))
                print "%15s %24.3lf %13.3e %20.3lf %14.3e %15d %15d" % ('no_retrofit', 
                                                                   np.average(result['NPV']),
                                                                   np.average(result['NPV']),
                                                                   np.std(result['NPV']),
                                                                   np.std(result['NPV']),
                                                                   len(result), 0)
            print "%20s" % ('-^-'*50)
            print "%20s" % ('-#-'*50)
            print "%20s\n%15s\n%20s" % ('-'*20, "simulation details".upper(), '-'*20)
            print "%10s %17s %20s %23s %18s %45s %27s" % ('sim'.upper(),
                                                          'flexible'.upper(), 
                                                          'no retrofit'.upper(), 
                                                          'judgement'.upper(),
                                                          'delta'.upper(),
                                                          'design transition'.upper(),
                                                          'retrofit date'.upper() )
            simulate_induces = np.unique(np.r_[result['simulation_time'],flexible_result['simulation_time']])
            effective_count  = 0
            delta_dict       = {}
            for simulate_index in simulate_induces:
                nr_result = result[np.where(result['simulation_time']==simulate_index)]
                f_result  = flexible_result[np.where(flexible_result['simulation_time']==simulate_index)]
                if len(nr_result) == 1 and len(f_result) == 1:
                    no_retrofit_npv = ("%17.3lf" % nr_result['NPV'])
                    flexible_npv    = ("%17.3lf" % f_result['NPV'])
                    if (flexible_npv > no_retrofit_npv):
                        judgement = 'flexible'
                        effective_count += 1
                    else:
                        judgement = 'no_retrofit'
                    
                    delta           = "%17.3lf" % (float(flexible_npv)-float(no_retrofit_npv))
                    delta_dict[simulate_index] = float(flexible_npv)-float(no_retrofit_npv)
                    base_design_mode     = [key for key, value in RETROFIT_DESIGNS[bf_mode].iteritems() if value == nr_result['base_design']][0]
                    retrofit_design_mode = [key for key, value in RETROFIT_DESIGNS[bf_mode].iteritems() if value == f_result['retrofit_design'][0]]
                    retrofit_design_mode = retrofit_design_mode[0] if len(retrofit_design_mode) > 0 else '--'
                    transition_str  = "%s (%s) -> %s (%s)" % (nr_result['base_design'][0], base_design_mode, f_result['retrofit_design'][0], retrofit_design_mode)
                    retrofit_date   = f_result['retrofit_date'][0]
                    if not retrofit_date == '--':
                        draw_comparison_graph(simulate_index, retrofit_date, target_dir, target_dir_path)
                        
                else:
                    no_retrofit_npv = ("%17.3lf" % nr_result['NPV']) if len(nr_result) == 1 else '--------'
                    if len(f_result) == 1:
                        flexible_npv    = f_result['NPV']
                        retrofit_date   = f_result['retrofit_date'][0] 
                    else:
                        flexible_npv = retrofit_date = '--------'                 
                    transition_str  = "%s -> --" % (nr_result['base_design'][0])
                    judgement = delta = '--------'

                print "%10s %20s %20s %20s %20s %50s %20s" % (simulate_index, 
                                                              flexible_npv,
                                                              no_retrofit_npv,
                                                              judgement.upper(),
                                                              delta,
                                                              transition_str,
                                                              retrofit_date)
            print "%20s" % ('-^-'*50)
            maximum_delta_index = max(delta_dict.iteritems(), key=operator.itemgetter(1))[0]
            print "%29s: %10d\n%29s: %17.3lf\n%20s(at %4d): %17.3lf" % ('effective count'.upper(), effective_count,
                                                                        'average delta'.upper(), np.average(delta_dict.values()),
                                                                        'maximum delta'.upper(), maximum_delta_index, delta_dict[maximum_delta_index])
            print "%20s" % ('-^-'*50)
    return

def draw_comparison_graph(index_num, retrofit_date, target_dir, target_dir_path):
    # initialize
    dt               = np.dtype({'names': ('date','npv'),
                     'formats': ('S10', np.float)})
    output_dir       = "%s/%s/comparison_graphs" % (target_dir_path, target_dir)
    initializeDirHierarchy(output_dir)
    output_file_path = "%s/%d_comparison.png" % (output_dir, index_num)

    # draw graph
    title    = "NPV and oilprice\n".upper()
    x_label  = "date".upper()
    y0_label = "PV [USD]".upper()
    y1_label = "oil price [USD/barrel]".upper()
    
    ## for flexible
    desti_dir   = "%s/%s/flexible" % (target_dir_path, target_dir)
    if os.path.exists(desti_dir):
        # calc average npv from initial_designs
        files        = os.listdir(desti_dir)
        target_file = [_f for _f in files if _f[:1] == 'H'][-1]
        target_file_path = "%s/%s/simulate%d/npv.csv" % (desti_dir, target_file, index_num)
        if os.path.exists(target_file_path):
            data = np.genfromtxt(target_file_path,
                                 delimiter=',',
                                 dtype=dt,
                                 skiprows=1)
            flexible_draw_data = [ [datetime.datetime.strptime(_d['date'], '%Y/%m/%d'), _d['npv']] for _d in data]
            flexible_draw_data = np.array(sorted(flexible_draw_data, key= lambda x : x[0]))
            flexible_draw_label = "%s (Flexible)" % (target_file)
            # for no retrofit
            desti_dir   = "%s/%s/no_retrofit" % (target_dir_path, target_dir)
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
                    nr_draw_data = [ [datetime.datetime.strptime(_d['date'], '%Y/%m/%d'), _d['npv']] for _d in data]
                    nr_draw_data = np.array(sorted(nr_draw_data, key= lambda x : x[0]))
                    '''
                    plt.plot(nr_draw_data.transpose()[0],
                             nr_draw_data.transpose()[1],
                             label=target_file,
                             color='g', linestyle='--')
                    '''
                    # debug
                    # for first dock-in
                    #plt.axvline(x=nr_draw_data.transpose()[0][0] + datetime.timedelta(days=365*DOCK_IN_PERIOD), color='r', linewidth=4, linestyle='--')
                    simulation_duration_years = max(nr_draw_data.transpose()[0]).year - min(nr_draw_data.transpose()[0]).year
                    scenario = generate_sinario_with_seed(COMMON_SEED_NUM * index_num, simulation_duration_years)

                    # draw twin graphs
                    fig = plt.figure()
                    ax1 = fig.add_subplot(111)
                    ax2 = ax1.twinx()
                    
                    # no_retrofit
                    y0_data   = np.array([[_d[0], _d[1]] for _d in nr_draw_data])
                    y0_x_data = y0_data.transpose()[0]
                    p1        = ax1.plot(y0_x_data, [ float(_d[1]) for _d in y0_data], color='r')
                    # flexible
                    y0_data   = np.array([[_d[0], _d[1]] for _d in flexible_draw_data])
                    y0_x_data = y0_data.transpose()[0]
                    p3        = ax1.plot(y0_x_data, [ float(_d[1]) for _d in y0_data], color='g')
                    ax1.set_ylabel(y0_label)
                    # oil price                    
                    y1_data   = scenario.predicted_data['price']
                    y1_x_data = [str_to_date(_d) for _d in scenario.predicted_data['date']]
                    p2        = ax2.plot(y1_x_data, [ float(_d) for _d in y1_data], color='b', linestyle='--')
                    ax2.set_ylabel(y1_label)
                    plt.legend([p1[0], p3[0], p2[0]], [target_file, flexible_draw_label, 'oil price'], loc='lower left')

                    # draw origin line
                    plt.title(title)    
                    ax1.axhline(linewidth=1.5, color='k')
                    # for retrofit date
                    ax1.axvline(x=str_to_datetime(retrofit_date), color='k', linewidth=4, linestyle='--')                

                    ax1.set_xlabel(x_label)
                    ax1.grid(True)
                    plt.savefig(output_file_path)
                    plt.close()
    return

def draw_npv_histgram(npv_result, oilprice_mode, output_dir_path):
    # draw graph
    title    = "%s (at %s)" % ("NPV for each design".upper(), oilprice_mode.replace('_', ' '))
    x_label  = "design id".upper()
    y_label  = "PV [USD]".upper()
    filepath = "%s/%s_npv.png" % (output_dir_path, oilprice_mode)
    dt       = np.dtype({'names': ('design_id','npv'),
                     'formats': ('S10', np.float)})    
    graphInitializer(title, x_label, y_label)
    draw_data = np.array(sorted([(k, v) for k,v in npv_result.items()], key=lambda x : x[1], reverse=True), dtype=dt)
    ticks     = { i:_d for i, _d in enumerate(draw_data['design_id'])}
    plt.bar( [ _i + 1 for _i in ticks.keys()], draw_data['npv'])
    plt.xticks( [_i + 1 for _i in ticks.keys()], ticks.values(), rotation=60, fontsize=7)
    plt.savefig(filepath)
    plt.close()    
    
    return

def draw_fuel_cost_histgram(fuel_cost_result, oilprice_mode, output_dir_path):
    # draw graph
    title    = "%s (at %s)" % ("fuel cost for each design".upper(), oilprice_mode.replace('_', ' '))
    x_label  = "design id".upper()
    y_label  = "fuel cost [USD]".upper()
    filepath = "%s/%s_fuel_cost.png" % (output_dir_path, oilprice_mode)
    dt       = np.dtype({'names': ('design_id','fuel_cost'),
                         'formats': ('S10', np.float)})    
    graphInitializer(title, x_label, y_label)
    draw_data = np.array(sorted([(k, v) for k,v in fuel_cost_result.items()], key=lambda x : x[1], reverse=True), dtype=dt)
    ticks     = { i:_d for i, _d in enumerate(draw_data['design_id'])}
    plt.bar( [ _i + 1 for _i in ticks.keys()], draw_data['fuel_cost'])
    plt.xticks( [_i + 1 for _i in ticks.keys()], ticks.values(), rotation=60, fontsize=7)
    plt.savefig(filepath)
    plt.close()    
    
    return

def draw_npv_fuel_twingraph(npv_result, fuel_cost_result, oilprice_mode, output_dir_path):
    # initialize graph
    title    = "%s (at %s)\n" % ("npv and fuel cost for each design".upper(), oilprice_mode.replace('_', ' '))
    x_label  = "design id".upper()
    y0_label = "npv".upper() + '[USD]'
    y1_label = "fuel cost".upper() + '[USD]'
    filepath = "%s/npv_fuel_twin_%s.png" % (output_dir_path, oilprice_mode)
    dt       = np.dtype({'names': ('design_id','npv', 'fuel_cost'),
                         'formats': ('S10', np.float, np.float)})
    draw_data = np.array(sorted([(k, v, fuel_cost_result[k]) for k,v in npv_result.items()], key=lambda x : x[1], reverse=True), dtype=dt)
    draw_twin_graph(draw_data, title, x_label, y0_label, y1_label, [2.7e9, 3.5e9], [0.6e8, 2.4e8], 'lower left')
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
            title    = "%s (at %s)\n" % ("rpm and velocity for each design".upper(), oilprice_mode.replace('_', ' '))
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
            draw_twin_graph(draw_data, title, x_label, y0_label, y1_label, [40, 100], [12, 20])
            plt.savefig(filepath)
            plt.close()
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
