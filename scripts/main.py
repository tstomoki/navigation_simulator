# import common modules #
import sys
import pdb
# import common modules #

# import own modules #
sys.path.append('../public')
from my_modules import *
# import own modules #

# import models #
sys.path.append('../models')
from hull        import Hull
from sinario     import Sinario
from engine      import Engine
from propeller   import Propeller
from world_scale import WorldScale
# import models #

def simmulate():
    # load history data
    from_date = '2004/01/01'
    to_date = '2014/01/01'
    history_data = load_monthly_history_data(from_date, to_date)

    # generate sinario
    base_sinario = Sinario(history_data)
    base_sinario.generate_sinario()
    
    # generate world scale
    world_scale = WorldScale(load_world_scale_history_data())
    world_scale.draw_history_data()

# authorize exeucation as main script
if __name__ == '__main__':
    simmulate()

