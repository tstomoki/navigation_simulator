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
# import own modules #

# constants #
RESULTSDIR = '../results/'
# constants #

class Agent:
    def __init__(self, sinario, world_scale, mode, hull=None, engine=None, propeller=None):
        self.sinario     = sinario
        self.world_scale = world_scale
        self.mode = mode

        if (hull is None or engine is None or propeller is None):
            self.hull, self.engine, self.propeller = self.get_initial_design(mode)
        else:
            self.hull, self.engine, self.propeller = hull, engine, propeller

    def get_initial_design(self, mode):
        # full search with hull, engine and propeller
        pdb.set_trace()
