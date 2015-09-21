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
# for multi processing
import multiprocessing as mp
# for multi processing
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
    retrofit_mode                        = options.retrofit_mode

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
    
    # retrofit mode
    # load components list
    hull_list           = load_hull_list()
    engine_list         = load_engine_list()
    propeller_list      = load_propeller_list()
    if retrofit_mode == '2':
        high_design_key           = "H1E3P514"
        low_design_key            = "H2E1P514"
        case_modes                = ['high', 'low']
        common_seed_num           = 19901129
        simulation_duration_years = VESSEL_LIFE_TIME
        simulation_times          = 30
        ## DEBUG ##
        simulation_duration_years = 1
        simulation_times          = 2
        ## DEBUG ##
        devided_simulation_times  = np.array_split(range(simulation_times), PROC_NUM)
        # initialize
        pool                      = mp.Pool(PROC_NUM)
        for case_mode in case_modes:
            # integrated case
            base_design_key            = eval(case_mode + "_design_key")
            retrofit_case_mode         = [_e for _e in case_modes if not _e == case_mode][-1]
            retrofit_design_key        = eval(retrofit_case_mode + "_design_key")
            retrofit_mode              = RETROFIT_MODE['significant']
            output_integrated_dir_path = "%s/%s_design/integrated" % (output_dir_path, case_mode)
            initializeDirHierarchy(output_integrated_dir_path)
            agent                      = Agent(base_sinario,
                                               world_scale,
                                               flat_rate,
                                               retrofit_mode,
                                               DERIVE_SINARIO_MODE['binomial'],
                                               BF_MODE['calm'])
            agent.output_dir_path = output_integrated_dir_path
            

            # multi processing #
            callback              = [pool.apply_async(agent.calc_integrated_design_m, args=(index, common_seed_num, hull_list, engine_list, propeller_list, simulation_duration_years, devided_simulation_times, base_design_key, retrofit_design_key, output_integrated_dir_path)) for index in xrange(PROC_NUM)]
            callback_combinations = [p.get() for p in callback]
            ret_combinations      = flatten_3d_to_2d(callback_combinations)
            pool.close()
            pool.join()
            # multi processing #            

            # significant case
            design_key           = base_design_key
            # initialize
            retrofit_mode        = RETROFIT_MODE['none']
            # fix seed #
            agent                = Agent(base_sinario,
                                         world_scale,
                                         flat_rate,
                                         retrofit_mode,
                                         DERIVE_SINARIO_MODE['binomial'],
                                         BF_MODE['calm'])
            output_case_dir_path = "%s/%s_design" % (output_dir_path, case_mode)
            # multi processing #
            callback              = [pool.apply_async(agent.calc_design_m, args=(index, common_seed_num, hull_list, engine_list, propeller_list, simulation_duration_years, devided_simulation_times, design_key, output_case_dir_path)) for index in xrange(PROC_NUM)]
            callback_combinations = [p.get() for p in callback]
            ret_combinations      = flatten_3d_to_2d(callback_combinations)
            pool.close()
            pool.join()
            # multi processing #
        print_with_notice("Program finished at %s" % (detailed_datetime_to_human(datetime.datetime.now())))        
        sys.exit()
    
    if retrofit_mode:
        oilprice_modes = [ 'oilprice_' + s for s in ['high', 'low']]
        
        # search initial_design
        for oilprice_mode in oilprice_modes:
            # generate sinario
            sinario, world_scale, flat_rate = generate_significant_modes(oilprice_mode, 
                                                                         oil_price_history_data, 
                                                                         world_scale_history_data, 
                                                                         flat_rate_history_data)
            # initialize
            retrofit_mode = RETROFIT_MODE['none']
            agent         = Agent(sinario,
                                  world_scale,
                                  flat_rate,
                                  retrofit_mode,
                                  'significant',
                                  BF_MODE['calm'])

            output_path = "%s/%s" % (output_dir_path, oilprice_mode)
            initializeDirHierarchy(output_path)

            devided_component_ids = []
            for hull_info in hull_list:
                for engine_info in engine_list:
                    for propeller_info in propeller_list:
                        devided_component_ids.append([hull_info['id'], engine_info['id'], propeller_info['id']])
            devided_component_ids = np.array_split(devided_component_ids, PROC_NUM)

            # debug
            simulation_duration_years = VESSEL_LIFE_TIME
            #agent.calc_significant_design_m(0, hull_list, engine_list, propeller_list, simulation_duration_years, devided_component_ids, output_path)

            # initialize
            pool                      = mp.Pool(PROC_NUM)
            # multi processing #
            callback              = [pool.apply_async(agent.calc_significant_design_m, args=(index, hull_list, engine_list, propeller_list, simulation_duration_years, devided_component_ids, output_path)) for index in xrange(PROC_NUM)]

            callback_combinations = [p.get() for p in callback]
            ret_combinations      = flatten_3d_to_2d(callback_combinations)
            pool.close()
            pool.join()
            # multi processing #
        print_with_notice("Program finished at %s" % (detailed_datetime_to_human(datetime.datetime.now())))        
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
    parser.add_option("-T", "--retrofit-mode", dest="retrofit_mode",
                      help="conduct retrofit case studies if True", default=False)

    (options, args) = parser.parse_args()
    
    #output_ratio_dict_sub_for_sig_kst()
    run(options)

