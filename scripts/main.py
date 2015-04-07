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
from hull      import Hull
from sinario   import Sinario
from engine    import Engine
from propeller import Propeller
# import models #

def simmulate():
    # load history data
    history_data = load_history_data()

    # generate sinario
    base_sinario = Sinario(history_data)
    base_sinario.show_history_data()

# authorize exeucation as main script
if __name__ == '__main__':
    simmulate()

