# import common modules #
import time
import sys
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
        sys.exit()

    draw_hull_features()
    #draw_propeller_features()
    draw_engine_features()
    #aggregate_results(result_dir_path)
    draw_ct_fn()
    draw_ct_fn(True, 'BF6')
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
    column_names = 'scenario_total_num','hull_id','engine_id','propeller_id','average NPV', 'std'
    for initial_design_dir in initial_design_dirs:
        target_dir = "%s/%s" % (result_dir_path, initial_design_dir)
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
            output_file_path = "%s/%s.csv" % (output_dir_path, initial_design_dir)

            for design_key, npvs in npv_result.items():
                hull_id, engine_id, propeller_id = get_component_ids_from_design_key(design_key)
                write_csv(column_names, [senario_total_num,
                                         hull_id,
                                         engine_id,
                                         propeller_id,
                                         np.average(npvs),
                                         np.std(npvs)
                                         ], output_file_path)
    return

# authorize exeucation as main script
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-P", "--result-path", dest="result_dir_path",
                      help="results dir path", default=None)    
    parser.add_option("-A", "--aggregate", dest="aggregate",
                      help="aggregate mode", default=False)    
    (options, args) = parser.parse_args()
    run(options)    
