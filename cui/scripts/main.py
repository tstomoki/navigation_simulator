# import common modules #
import sys
import pdb
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

def run():
    # load history data
    from_date = '2004/01/01'
    to_date = '2014/01/01'
    history_data = load_monthly_history_data(from_date, to_date)

    # generate sinario
    base_sinario = Sinario(history_data)
    # generate world scale
    world_scale = WorldScale(load_world_scale_history_data())

    # initiate a simmulation
    # for design 0
    retrofit_mode = RETROFIT_MODE['none']
    sinario_mode  = DERIVE_SINARIO_MODE['maintain']
    agent         = Agent(base_sinario, world_scale, retrofit_mode, sinario_mode)


    # get initial design #
    output_dir_path = "%s/%s" % (AGNET_LOG_DIR_PATH, generate_timestamp())
    initializeDirHierarchy(output_dir_path)
    NPV, initial_hull, initial_engine, initial_propeller = agent.get_initial_design_m(output_dir_path)
    # get initial design #
    '''
    # for design 1
    retrofit_mode = RETROFIT_MODE['high']
    sinario_mode  = DERIVE_SINARIO_MODE['propeller']
    agent         = Agent(base_sinario, world_scale, retrofit_mode, sinario_mode)

    # for design 2
    retrofit_mode = RETROFIT_MODE['high']
    sinario_mode  = DERIVE_SINARIO_MODE['propeller_and_engine']
    agent         = Agent(base_sinario, world_scale, retrofit_mode, sinario_mode)
    '''
    return 



# authorize exeucation as main script
if __name__ == '__main__':
    #nohup_dir_path = "%s/%s" % (NOHUP_LOG_DIR_PATH, generate_timestamp())
    #initializeDirHierarchy(nohup_dir_path)
    run()

