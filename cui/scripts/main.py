# import common modules #
import sys
import pdb
import matplotlib
matplotlib.use('Agg')
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
from agent       import Agent
# import models #

def run(options):
    print_with_notice("Program started at %s" % (detailed_datetime_to_human(datetime.datetime.now())))

    # get option variables #
    initial_hull_id      = options.hull_id
    initial_engine_id    = options.engine_id
    initial_propeller_id = options.propeller_id
    design_num           = options.design_num
    create_combination   = options.create_combination

    # load history data
    from_date = '2004/01/01'
    to_date = '2014/01/01'
    history_data = load_monthly_history_data(from_date, to_date)

    # generate sinario
    base_sinario = Sinario(history_data)
    # generate world scale
    world_scale = WorldScale(load_world_scale_history_data())
    base_sinario.draw_multiple_scenarios(world_scale)
    sys.exit()

    # initialize directory 
    output_dir_path = "%s/%s" % (AGNET_LOG_DIR_PATH, generate_timestamp())
    initializeDirHierarchy(output_dir_path)

    # only creating velocity combinations
    if create_combination:
        retrofit_mode = RETROFIT_MODE['none']
        sinario_mode  = DERIVE_SINARIO_MODE['maintain']
        agent         = Agent(base_sinario, world_scale, retrofit_mode, sinario_mode)        
        agent.only_create_velocity_combinations()
        print_with_notice("Program (calc velocity combinations) finished at %s" % (detailed_datetime_to_human(datetime.datetime.now())))
        sys.exit()
    
    if (initial_hull_id is None) or (initial_engine_id is None) or (initial_propeller_id is None):
        # for design 0 #
        # get initial design #
        retrofit_mode = RETROFIT_MODE['none']
        sinario_mode  = DERIVE_SINARIO_MODE['maintain']
        agent         = Agent(base_sinario, world_scale, retrofit_mode, sinario_mode)
        NPV, initial_hull, initial_engine, initial_propeller = agent.get_initial_design_m(output_dir_path)
        # get initial design #
        # for design 0 #
    else:
        # load components list
        hull_list      = load_hull_list()
        engine_list    = load_engine_list()
        propeller_list = load_propeller_list()
        # get components
        initial_hull      = Hull(hull_list, initial_hull_id)
        initial_engine    = Engine(engine_list, initial_engine_id)
        initial_propeller = Propeller(propeller_list, initial_propeller_id)        

    if (design_num is None) or (design_num == 1):
        # for design 1
        retrofit_mode = RETROFIT_MODE['propeller']
        sinario_mode  = DERIVE_SINARIO_MODE['binomial']
        tmp_output_path = "%s/propeller" % (output_dir_path)
        initializeDirHierarchy(tmp_output_path)
        agent         = Agent(base_sinario, world_scale, retrofit_mode, sinario_mode, initial_hull, initial_engine, initial_propeller)
        agent.output_dir_path = tmp_output_path
        # simmulate with multi flag
        agent.simmulate(None, None, None, True)
        print_with_notice("Program (retrofit propeller_and_engine) finished at %s" % (detailed_datetime_to_human(datetime.datetime.now())))
    elif (design_num is None) or (design_num == 2):
        # for design 2
        retrofit_mode   = RETROFIT_MODE['propeller_and_engine']
        sinario_mode    = DERIVE_SINARIO_MODE['binomial']
        tmp_output_path = "%s/propeller_and_engine" % (output_dir_path)
        initializeDirHierarchy(tmp_output_path)
        agent           = Agent(base_sinario, world_scale, retrofit_mode, sinario_mode)
        agent.output_dir_path = tmp_output_path
        # simmulate with multi flag
        agent.simmulate(None, None, None, True)
        print_with_notice("Program (retrofit propeller_and_engine) finished at %s" % (detailed_datetime_to_human(datetime.datetime.now())))
        
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
    parser.add_option("-D", "--design", dest="design_num",
                      help="designate execute simmulate num", default=None, type="int")

    parser.add_option("-C", "--combinations", dest="create_combination",
                      help="only create velocity combinations", default=False)

    (options, args) = parser.parse_args()
    run(options)

