# import common modules #
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
# import own modules #

# import models #
sys.path.append('../models')
from hull        import Hull
from sinario     import Sinario
from engine      import Engine
from propeller   import Propeller
from world_scale import WorldScale
from flat_rate   import FlatRate
from agent       import Agent
# import models #

def run(options):
    print_with_notice("Program started at %s" % (detailed_datetime_to_human(datetime.datetime.now())))

    # get option variables #
    initial_hull_id                      = options.hull_id
    initial_engine_id                    = options.engine_id
    initial_propeller_id                 = options.propeller_id
    initial_design                       = options.initial_design
    create_combination                   = options.create_combination
    result_visualize_mode                = options.result_visualize_mode
    result_dir_path                      = options.result_dir_path
    no_retrofit_ignore                   = options.no_retrofit_ignore
    propeller_retrofit_ignore            = options.propeller_retrofit_ignore
    propeller_and_engine_retrofit_ignore = options.propeller_and_engine_retrofit_ignore

    # load history data
    from_date = '2004/01/01'
    to_date = '2015/01/01'
    oil_price_history_data   = load_monthly_history_data(from_date, to_date)
    world_scale_history_data = load_world_scale_history_data()
    flat_rate_history_data   = load_flat_rate_history_data()

    # generate sinario
    base_sinario = Sinario(oil_price_history_data)
    # generate world scale
    world_scale  = WorldScale(world_scale_history_data)
    # generate flat rate
    #flat_rate    = FlatRate(flat_rate_history_data)
    flat_rate    = FlatRate(flat_rate_history_data)
    #flat_rate.draw_multiple_flat_rates()
    
    # draw multiple scenario part #
    # base_sinario.draw_multiple_scenarios(world_scale)
    
    # correlation analysis #
    analyze_correlation(oil_price_history_data, world_scale_history_data,
                        {'start': datetime.datetime(2009, 1, 1), 'end': datetime.datetime.now()})
    # initialize directory 
    output_dir_path = "%s/%s" % (AGNET_LOG_DIR_PATH, generate_timestamp())
    initializeDirHierarchy(output_dir_path)

    if result_visualize_mode:
        # for narrowed result
        files = os.listdir(result_dir_path)
        target_files = [_f for _f in files if _f[-4:] == '.csv' and not _f == 'initial_design.csv']
        target_files = [ "%s/%s" % (result_dir_path, _f) for _f in target_files]

        dt   = np.dtype({'names': ('scenario_num','hull_id','engine_id','propeller_id','NPV'),
                         'formats': (np.int64, np.int64, np.int64, np.int64, np.float)})
        npv_result = {}
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
        if (initial_engine_id is None) or (initial_propeller_id is None):
            print "Error: please input engine and propeller ids"
        else:
            compare_hull_design(npv_result, initial_engine_id, initial_propeller_id)
            
        output_result = {}
        for c_key, npv_array in npv_result.items():
            if not output_result.has_key(c_key):
                output_result[c_key] = {}
            output_result[c_key]['npv'] = np.average(npv_array)
            output_result[c_key]['std'] = np.std(npv_array)

        display_sorted_result(output_result, 3)
        sys.exit()

        # output result
        output_mode_str  = re.compile(r'(mode.+)').search(result_dir_path).groups()[0]
        output_json_dir  = "%s/json" % (RESULT_DIR_PATH)
        initializeDirHierarchy(output_json_dir)
        output_json_path = "%s/%s.json" % (output_json_dir, output_mode_str)
        write_file_as_json(output_result, output_json_path)
        
        column_names = ['c_key', 'ave_npv', 'std']
        write_data = [ [c_key, val['npv'], val['std']] for c_key, val in output_result.items()]
        write_simple_array_csv(column_names, write_data, './test.csv')
        sys.exit()

        # display maximum_designs
        draw_NPV_for_each3D(results_data, output_dir_path, [0, 900], [190000000, 206000000])
        display_maximum_designs(results_data, 10)
        draw_whole_NPV(results_data, output_dir_path)
        draw_each_NPV_distribution(results_data, output_dir_path)

        # for narrow_down result
        output_dir_path = "%s/visualization/narrow_down" % (RESULT_DIR_PATH)
        initializeDirHierarchy(output_dir_path)
        json_dirpath    = NARROW_DOWN_RESULT_PATH
        if not os.path.exists(json_dirpath):
            print "abort: there is no such directory, %s" % (json_dirpath)
            sys.exit()
        files = os.listdir(json_dirpath)

        #draw_NPV_histogram_m(json_filepath, output_filepath)

        results_data = {}
        for target_filename in files:
            try:
                combination_key, = re.compile(r'(H.+)\.json').search(target_filename).groups()
            except:
                continue
            target_filepath               = "%s/%s" % (json_dirpath, target_filename)
            results_data[combination_key] = load_json_file(target_filepath)

        draw_NPV_for_each3D(results_data, output_dir_path, [0, 900], [35000000.0, 80000000.0])
        #    output_filepath = "%s/%s_NPV_result.png" % (output_dir_path, combination_key)
        #    draw_NPV_histogram(target_filepath, output_filepath)
        print_with_notice("Program (visualization) finished at %s" % (detailed_datetime_to_human(datetime.datetime.now())))
        sys.exit()
    
    # only creating velocity combinations
    if create_combination:
        retrofit_mode = RETROFIT_MODE['none']
        sinario_mode  = DERIVE_SINARIO_MODE['maintain']
        agent         = Agent(base_sinario, world_scale, flat_rate, retrofit_mode, sinario_mode)        
        agent.only_create_velocity_combinations()
        print_with_notice("Program (calc velocity combinations) finished at %s" % (detailed_datetime_to_human(datetime.datetime.now())))
        sys.exit()
    
    if (initial_hull_id is None) or (initial_engine_id is None) or (initial_propeller_id is None):
        # get initial design #
        # initialize
        retrofit_mode = RETROFIT_MODE['none']
        # fix seed #
        common_seed_num                 = 19901129        
        ## market: maintain, weather: calm
        agent         = Agent(base_sinario,
                              world_scale,
                              flat_rate,
                              retrofit_mode,
                              DERIVE_SINARIO_MODE['maintain'],
                              BF_MODE['calm'])
        initial_design_dir = "%s/initial_design_mode0" % (output_dir_path)
        initializeDirHierarchy(initial_design_dir)
        averaged_NPV, initial_hull, initial_engine, initial_propeller, std = agent.get_initial_design_m(initial_design_dir)        
        ## market: binomial, weather: calm
        agent         = Agent(base_sinario,
                              world_scale,
                              flat_rate,
                              retrofit_mode,
                              DERIVE_SINARIO_MODE['binomial'],
                              BF_MODE['calm'])
        initial_design_dir = "%s/initial_design_mode1" % (output_dir_path)
        initializeDirHierarchy(initial_design_dir)
        averaged_NPV, initial_hull, initial_engine, initial_propeller, std = agent.get_initial_design_m(initial_design_dir)                
        ## market: binomial, weather: rough 
        agent         = Agent(base_sinario,
                              world_scale,
                              flat_rate,
                              retrofit_mode,
                              DERIVE_SINARIO_MODE['binomial'],
                              BF_MODE['rough'])
        initial_design_dir = "%s/initial_design_mode2" % (output_dir_path)
        initializeDirHierarchy(initial_design_dir)
        averaged_NPV, initial_hull, initial_engine, initial_propeller, std = agent.get_initial_design_m(initial_design_dir)                
        if initial_design:
            print_with_notice("Program (search initial design) finished at %s" % (detailed_datetime_to_human(datetime.datetime.now())))
    else:
        # load components list
        hull_list      = load_hull_list()
        engine_list    = load_engine_list()
        propeller_list = load_propeller_list()
        # get components
        initial_hull      = Hull(hull_list, initial_hull_id)
        initial_engine    = Engine(engine_list, initial_engine_id)
        initial_propeller = Propeller(propeller_list, initial_propeller_id)        

    if not initial_design:
        # to compare retrofitting mode #
        # fix seed #
        common_seed_num                 = 19901129
        sinario_mode                    = DERIVE_SINARIO_MODE['binomial']
        vessel_life_time_for_simulation = VESSEL_LIFE_TIME
        retrofit_senario_mode           = RETROFIT_SCENARIO_MODE['significant']
        ## no retrofit ##
        if no_retrofit_ignore:
            print "%30s" % ("ignoring the no retrofit simulation")
        else:
            np.random.seed(common_seed_num)
            generate_market_scenarios(base_sinario, world_scale, flat_rate, sinario_mode, vessel_life_time_for_simulation)
            each_output_path           = "%s/no_retrofit" % (output_dir_path)
            initializeDirHierarchy(each_output_path)
            retrofit_mode              = RETROFIT_MODE['none']
            for weather_mode, weather_mode_val in BF_MODE.items():
                output_path = "%s/%s" % (each_output_path, weather_mode)
                initializeDirHierarchy(output_path)
                agent         = Agent(base_sinario,
                                      world_scale,
                                      flat_rate,
                                      retrofit_mode,
                                      DERIVE_SINARIO_MODE['binomial'],
                                      weather_mode_val,
                                      initial_hull, initial_engine, initial_propeller)
                agent.operation_date_array  = generate_operation_date(base_sinario.predicted_data['date'][0], str_to_date(base_sinario.predicted_data['date'][-1]))
                agent.output_dir_path       = each_output_path
                agent.retrofit_senario_mode = retrofit_senario_mode
                # simmulate with multi flag and log
                agent.simmulate(None, None, None, True, True)

        ## whole retrofit ##
        np.random.seed(common_seed_num)
        generate_market_scenarios(base_sinario, world_scale, flat_rate, sinario_mode, vessel_life_time_for_simulation)        
        each_output_path           = "%s/whole_retrofits" % (output_dir_path)
        initializeDirHierarchy(each_output_path)
        retrofit_mode              = RETROFIT_MODE['whole']
        for weather_mode, weather_mode_val in BF_MODE.items():
            output_path = "%s/%s" % (each_output_path, weather_mode)
            initializeDirHierarchy(output_path)
            agent         = Agent(base_sinario,
                                  world_scale,
                                  flat_rate,
                                  retrofit_mode,
                                  DERIVE_SINARIO_MODE['binomial'],
                                  weather_mode_val,
                                  initial_hull, initial_engine, initial_propeller)
            agent.operation_date_array  = generate_operation_date(base_sinario.predicted_data['date'][0], str_to_date(base_sinario.predicted_data['date'][-1]))
            agent.output_dir_path       = each_output_path
            agent.retrofit_senario_mode = retrofit_senario_mode
            # simmulate with multi flag and log
            agent.simmulate(None, None, None, True, True)
        
        '''
        ## propeller retrofit ##
        if propeller_retrofit_ignore:
            print "%30s" % ("ignoring the propeller retrofit simulation")
        else:
            np.random.seed(common_seed_num)
            generate_market_scenarios(base_sinario, world_scale, flat_rate, sinario_mode, vessel_life_time_for_simulation)
            each_output_path           = "%s/propeller" % (output_dir_path)
            initializeDirHierarchy(each_output_path)
            retrofit_mode              = RETROFIT_MODE['propeller']
            agent                      = Agent(base_sinario, world_scale, flat_rate, retrofit_mode, sinario_mode, initial_hull, initial_engine, initial_propeller)
            agent.operation_date_array = generate_operation_date(base_sinario.predicted_data['date'][0], str_to_date(base_sinario.predicted_data['date'][-1]))                
            agent.output_dir_path      = each_output_path
            # simmulate with multi flag and log
            agent.simmulate(None, None, None, True, True)

        ## propeller and engine retrofit ##
        if propeller_and_engine_retrofit_ignore:
            print "%30s" % ("ignoring the propeller retrofit simulation")
        else:
            np.random.seed(common_seed_num)
            generate_market_scenarios(base_sinario, world_scale, flat_rate, sinario_mode, vessel_life_time_for_simulation)        
            each_output_path           = "%s/propeller_and_engine" % (output_dir_path)
            initializeDirHierarchy(each_output_path)
            retrofit_mode              = RETROFIT_MODE['propeller_and_engine']
            agent                      = Agent(base_sinario, world_scale, flat_rate, retrofit_mode, sinario_mode, initial_hull, initial_engine, initial_propeller)
            agent.operation_date_array = generate_operation_date(base_sinario.predicted_data['date'][0], str_to_date(base_sinario.predicted_data['date'][-1]))
            agent.output_dir_path      = each_output_path
            # simmulate with multi flag and log
            agent.simmulate(None, None, None, True, True)
        '''
        print_with_notice("Program finished at %s" % (detailed_datetime_to_human(datetime.datetime.now())))        
        
    return 

# authorize exeucation as main script
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-H", "--hull", dest="hull_id",
                      help="designate initial hull", default=None, type="int")
    parser.add_option("-E", "--engine", dest="engine_id",
                      help="designate initial engine", default=None, type="int")
    parser.add_option("-P", "--propeller", dest="propeller_id",
                      help="designate initial propeller", default=None, type="int")
    parser.add_option("-I", "--initial", dest="initial_design",
                      help="only search initial design", default=False)
    parser.add_option("-C", "--combinations", dest="create_combination",
                      help="only create velocity combinations", default=False)
    parser.add_option("-R", "--result-mode", dest="result_visualize_mode",
                      help="results visualize mode", default=False)
    parser.add_option("-A", "--result-path", dest="result_dir_path",
                      help="results dir path", default=None)
    parser.add_option("-N", "--no-retrofit", dest="no_retrofit_ignore",
                      help="ignore no retrofit simulation if True", default=False)
    parser.add_option("-B", "--propeller-retrofit", dest="propeller_retrofit_ignore",
                      help="ignore propeller retrofit simulation if True", default=False)
    parser.add_option("-D", "--propeller-and-engine-retrofit", dest="propeller_and_engine_retrofit_ignore",
                      help="ignore propeller and engine retrofit simulation if True", default=False)

    (options, args) = parser.parse_args()
    
    #output_ratio_dict_sub_for_sig_kst()
    run(options)

