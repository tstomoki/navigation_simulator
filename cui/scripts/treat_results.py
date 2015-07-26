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
    aggregate_results(result_dir_path)
    draw_ct_fn()
    draw_ct_fn(True, 'BF6')
    return

def aggregate_results(result_dir_path):
    whole_result = {}

    for filename in os.listdir(result_dir_path):
        mode_str  = re.compile(r'(mode.+).json').search(filename).groups()[0]
        result_file_path = "%s/%s" % (result_dir_path, filename)
        whole_result[mode_str] = load_json_file(result_file_path)

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

# authorize exeucation as main script
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-P", "--result-path", dest="result_dir_path",
                      help="results dir path", default=None)    
    (options, args) = parser.parse_args()
    run(options)    
