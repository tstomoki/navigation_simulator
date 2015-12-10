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
    # get option variables #
    final_mode = options.final_mode
    combi_mode = options.combination_mode

    # load history data
    from_date                = '2004/01/01'
    to_date                  = '2015/01/01'
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

    # initialize directory 
    output_dir_path = "%s/%s" % (AGNET_LOG_DIR_PATH, generate_timestamp())
    initializeDirHierarchy(output_dir_path)

    # retrofit mode
    # load components list
    hull_list           = load_hull_list()
    engine_list         = load_engine_list()
    propeller_list      = load_propeller_list()

    # bow test
    #bow_test()
    
    if final_mode == '2':
        # init retrofit designs
        retrofit_designs          = RETROFIT_DESIGNS

        # initialize parameters
        simulation_duration_years = VESSEL_LIFE_TIME
        simulation_times          = 100
        ## debug
        devided_simulation_times  = np.array_split(range(simulation_times), PROC_NUM)
        sinario_mode              = DERIVE_SINARIO_MODE['binomial']
        retrofit_mode             = RETROFIT_MODE['significant_rule']

        # create variables
        ## 10 cases
        case_num = 10
        trends   = np.linspace(BASE_TREND['origin'], BASE_TREND['end'], case_num)
        ## 10 cases
        deltas   = np.linspace(BASE_DELTA['origin'], BASE_DELTA['end'], case_num)

        # for single case
        #trends = [0.05, 0.50, 0.10, 0.20]
        trends = [0.05]
        deltas = [0.30]
        for trend in trends:
            for delta in deltas:
                dir_name = "trend_%0.2lf_delta%0.2lf" % (trend, delta)
                #for bf_mode in sorted(BF_MODE.keys()):
                for bf_mode in ['rough']:
                    designs    = retrofit_designs[bf_mode]
                    case_modes = designs.keys()
                    case_modes = TARGET_DESIGNS[bf_mode]
                    for case_mode in case_modes:
                        base_design_key      = designs[case_mode]
                        retrofit_design_keys = { k:v for k,v in designs.items() if not k == case_mode}
                        agent                = Agent(base_sinario,
                                                     world_scale,
                                                     flat_rate,
                                                     retrofit_mode,
                                                     sinario_mode, BF_MODE[bf_mode])
                        agent.rules          = {'trend': trend, 'delta': delta}
                        agent.output_dir_path = "%s/%s/%s/%s_design" % (output_dir_path, dir_name, bf_mode, case_mode)

                        # multi processing #
                        # initialize
                        pool                  = mp.Pool(PROC_NUM)
                        callback              = [pool.apply_async(agent.calc_flexible_design_m, args=(index, hull_list, engine_list, propeller_list, simulation_duration_years, devided_simulation_times, base_design_key, retrofit_design_keys, retrofit_mode)) for index in xrange(PROC_NUM)]
                        pool.close()
                        pool.join()
                        # multi processing #
                        
    if final_mode:
        significant_modes = ['high', 'low', 'dec', 'inc']
        oilprice_modes    = [ 'oilprice_' + s for s in significant_modes]
        # search initial_design
        for bf_mode in BF_MODE.keys():
            for oilprice_mode in oilprice_modes:
                # generate sinario
                sinario, world_scale, flat_rate = generate_final_significant_modes(oilprice_mode, 
                                                                                   oil_price_history_data, 
                                                                                   world_scale_history_data, 
                                                                                   flat_rate_history_data)
                # visualize for debug
                #sinario.draw_significant_oilprice_modes(oilprice_modes)
                #world_scale.draw_significant_oilprice_modes(oilprice_modes, sinario)
        
                # initialize
                retrofit_mode = RETROFIT_MODE['none']
                agent         = Agent(sinario,
                                      world_scale,
                                      flat_rate,
                                      retrofit_mode,
                                      'significant',
                                      BF_MODE[bf_mode])

                output_path = "%s/%s/%s" % (output_dir_path, bf_mode, oilprice_mode)
                initializeDirHierarchy(output_path)
                
                devided_component_ids = []
                for hull_info in hull_list:
                    for engine_info in engine_list:
                        for propeller_info in propeller_list:
                            devided_component_ids.append([hull_info['id'], engine_info['id'], propeller_info['id']])
                devided_component_ids = np.array_split(devided_component_ids, PROC_NUM)
                            
                simulation_duration_years = VESSEL_LIFE_TIME
                '''
                simulation_duration_years = 1
                devided_component_ids[0] = np.array([[1, 1, 0], [1, 2, 0]], dtype=np.int16)
                agent.calc_significant_design_m(0, hull_list, engine_list, propeller_list, simulation_duration_years, devided_component_ids, output_path)
                sys.exit()
                '''
                
                # initialize
                pool                      = mp.Pool(PROC_NUM)
                # multi processing #
                callback              = [pool.apply_async(agent.calc_significant_design_m, args=(index, hull_list, engine_list, propeller_list, simulation_duration_years, devided_component_ids, output_path)) for index in xrange(PROC_NUM)]
                
                callback_combinations = [p.get() for p in callback]
                ret_combinations      = flatten_3d_to_2d(callback_combinations)
                pool.close()
                pool.join()
        # multi processing #
    if combi_mode:
        Agent(None,
              None,
              None,
              None,
              None,
              None).only_create_velocity_combinations()
    return 

# authorize exeucation as main script
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-C", "--combi-mode", dest="combination_mode",
                      help="create combination if True", default=False)        
    parser.add_option("-F", "--final-mode", dest="final_mode",
                      help="conduct final case studies if True", default=False)    
    (options, args) = parser.parse_args()
    start = time.time()
    run(options)
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"
