# import common modules #
import sys
import math
import pdb
import matplotlib.pyplot as plt
import numpy as np
# import common modules #

# import own modules #
sys.path.append('../public')
from my_modules import *
from constants  import *
# import own modules #

# constants #
RESULTSDIR = '../results/'
# constants #

# import models #
sys.path.append('../models')
from hull        import Hull
from sinario     import Sinario
from engine      import Engine
from propeller   import Propeller
from world_scale import WorldScale
# import models #

class Agent:
    def __init__(self, sinario, world_scale, retrofit_mode, sinario_mode, hull=None, engine=None, propeller=None):
        
        self.sinario       = sinario
        self.world_scale   = world_scale
        self.retrofit_mode = retrofit_mode
        self.sinario_mode  = sinario_mode

        if (hull is None or engine is None or propeller is None):
            self.hull, self.engine, self.propeller = self.get_initial_design(retrofit_mode, sinario_mode, sinario, world_scale)
        else:
            self.hull, self.engine, self.propeller = hull, engine, propeller
            
    ### full search with hull, engine and propeller
    def get_initial_design(self, retrofit_mode, sinario_mode, sinario, world_scale):
        # load components list
        hull_list           = load_hull_list()
        engine_list         = load_engine_list()
        propeller_list      = load_propeller_list()

        ## hull
        ### list has only 1 hull
        ret_hull = Hull(1, hull_list)

        ## engine and propeller
        ### full search with sinario and world_scale
        sinario.generate_sinario(sinario_mode)
        ### default flat_rate is 50 [%]
        world_scale.set_flat_rate(50)

        design = {}
        for engine in engine_list:
            for propeller in propeller_list:
                # conduct simmulation
                NPV = self.simmulate(ret_hull, engine, propeller, sinario, world_scale, retrofit_mode)
                # update design #
                if len(design) == 0:
                    design['hull']      = ret_hull
                    design['engine']    = engine
                    design['propeller'] = propeller
                    design['NPV']       = NPV
                elif design['NPV'] < NPV:
                    design['hull']      = ret_hull
                    design['engine']    = engine
                    design['propeller'] = propeller
                    design['NPV']       = NPV
                # update design #

        return design['hull'], design['engine'], design['propeller'], design['NPV']

    def simmulate(self, hull, engine, propeller, sinario, world_scale, retrofit_mode):
        # initialize retrofit_count
        if retrofit_mode == RETROFIT_MODE['none']:
            retrofit_count = 0

        # start navigation
        for current_date in sinario.predicted_data['date']:
            # calculate optimized speed
            v_knot = self.calc_velosity()
            pdb.set_trace()
