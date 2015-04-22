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
        # initialize the range of velosity and rps
        self.velocity_array = np.arange(VELOCITY_RANGE['from'], VELOCITY_RANGE['to'], VELOCITY_RANGE['stride'])
        self.rpm_array      = np.arange(RPM_RANGE['from'], RPM_RANGE['to'], RPM_RANGE['stride'])            
        
        if (hull is None or engine is None or propeller is None):
            self.hull, self.engine, self.propeller = self.get_initial_design()
        else:
            self.hull, self.engine, self.propeller = hull, engine, propeller
       
           
    ### full search with hull, engine and propeller
    def get_initial_design(self):
        # load components list
        hull_list           = load_hull_list()
        engine_list         = load_engine_list()
        propeller_list      = load_propeller_list()

        ## hull
        ### list has only 1 hull
        ret_hull = Hull(1, hull_list)

        ## engine and propeller
        ### full search with sinario and world_scale
        self.sinario.generate_sinario(self.sinario_mode)
        ### default flat_rate is 50 [%]
        self.world_scale.set_flat_rate(50)

        design = {}
        for engine in engine_list:
            for propeller in propeller_list:
                # conduct simmulation
                NPV = self.simmulate(ret_hull, engine, propeller)
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

    def simmulate(self, hull, engine, propeller):
        # initialize retrofit_count
        if self.retrofit_mode == RETROFIT_MODE['none']:
            retrofit_count = 0

        # define velocity and rps for given [hull, engine, propeller]
        velocity_combination = self.create_velocity_combination(hull, engine, propeller)
            
        # load condition [ballast, full]
        load_condition = INITIAL_LOAD_CONDITION
        for current_date in self.sinario.predicted_data['date']:
            # calculate optimized speed and rps
            v_knot, rps = self.calc_velosity(load_condition)
            pdb.set_trace()

    # define velocity and rps for given [hull, engine, propeller]
    def create_velocity_combination(self, hull, engine, propeller):
        ret_combinations = {}
        
        for load_condition in LOAD_CONDITION.keys():
            combinations = np.array([])
            for rpm in self.rpm_array:
                # reject rps over the engine Max rps
                if rpm >= engine['N_max']:
                    next
                rps = round(rpm / 60.0, 4)
                tmp_combinations = np.array([])
                for velocity in self.velocity_array:
                    velocity = round(velocity, 4)
    
                    # calc error of fitness bhp values
                    error = self.rps_velocity_fitness(hull, engine, propeller, velocity, rps, load_condition)
                    tmp_combinations = np.append(tmp_combinations, [rpm, velocity, error]) if len(tmp_combinations) == 0 else np.vstack((tmp_combinations, [rpm, velocity, error]))
                # remove None error value
                remove_induces = np.array([])
                for index, element in enumerate(tmp_combinations):
                    if element[2] is None:
                        remove_induces = np.append(remove_induces, index)
                tmp_combinations = np.delete(tmp_combinations, remove_induces, 0)
                min_combination = tmp_combinations[np.argmin(tmp_combinations[:, 2])]

                combinations = np.append(combinations, [min_combination[0],min_combination[1]]) if len(combinations) == 0 else np.vstack((combinations, [min_combination[0],min_combination[1]]))
            ret_combinations[load_condition] = combinations

        return ret_combinations

    def rps_velocity_fitness(self, hull, engine, propeller, velocity, rps, load_condition):
        # calc bhp [WW]
        fitness_bhp0 = self.calc_bhp_with_rps(          rps, hull, engine, propeller)
        fitness_bhp1 = self.calc_bhp_with_ehp(velocity, rps, hull, engine, propeller, load_condition)
        '''
        if velocity == 8.8:
            pdb.set_trace()                
        '''
        # reject bhp over the engine Max load
        if fitness_bhp0 is None or fitness_bhp1 is None or fitness_bhp0 > engine['max_load'] or fitness_bhp1 > engine['max_load']:
            return None
                
        error = math.pow(fitness_bhp0 - fitness_bhp1, 2)
        error = math.sqrt(error)

        return error

    # return bhp [kW]
    def calc_bhp_with_rps(self, rps, hull, engine, propeller):
        # read coefficients
        bhp_coefficients = dict.fromkeys(['bhp0', 'bhp1', 'bhp2'], 0)
        for bhp_coefficients_key in bhp_coefficients.keys():
            bhp_coefficients[bhp_coefficients_key] = engine[bhp_coefficients_key]

        max_rps = round(engine['N_max'] / 60.0, 4)
        bhp  = bhp_coefficients['bhp0'] + bhp_coefficients['bhp1'] * (rps / max_rps) + bhp_coefficients['bhp2'] * math.pow(rps / max_rps, 2)

        return bhp

    # return bhp [kW]    
    def calc_bhp_with_ehp(self, velocity, rps, hull, engine, propeller, load_condition):
        # reject if the condition (KT > 0 and eta > 0) fulfilled
        J   = self.calc_advance_constant(velocity, rps, propeller['D'])
        KT  = propeller['KT0'] + propeller['KT1'] * J + propeller['KT2'] * math.pow(J,2)
        eta = THRUST_COEFFICIENT * ( velocity / (2 * math.pi) ) * (1.0 / (rps * propeller['D']) ) * ( (KT) / (propeller['KQ0'] + propeller['KQ1'] * J + propeller['KQ2'] * math.pow(J,2)) )

        if KT < 0 or eta < 0:
            return None
        
        # read coefficients
        ehp_coefficients = dict.fromkeys(['ehp0', 'ehp1', 'ehp2', 'ehp3', 'ehp4'], 0)
        for ehp_coefficients_key in ehp_coefficients.keys():
            data_key = "%s_%s" % (ehp_coefficients_key, load_condition)
            ehp_coefficients[ehp_coefficients_key] = hull.base_data[data_key][0]

        # advance constants
        J = self.calc_advance_constant(velocity, rps, propeller['D'])

        # calc numerator
        numerator =  ehp_coefficients['ehp0'] + ehp_coefficients['ehp1'] * velocity
        numerator += ehp_coefficients['ehp2'] * math.pow(velocity, 2) + ehp_coefficients['ehp3'] * math.pow(velocity, 3) + ehp_coefficients['ehp4'] * math.pow(velocity, 4)

        # calc denominator
        denominator = THRUST_COEFFICIENT * (velocity / (2 * math.pi) ) * (1 / (rps * propeller['D']) ) * ( (propeller['KT0'] + propeller['KT1'] * J + propeller['KT2'] * math.pow(J,2)) / (propeller['KQ0'] + propeller['KQ1'] * J + propeller['KQ2'] * math.pow(J,2)) ) * ETA_S

        bhp = numerator / denominator
        # return bhp [kW]

        return bhp    
    
    def calc_velosity(self, load_condition):
        pdb.set_trace()

    # calc advance constant
    def calc_advance_constant(self, velocity, rps, diameter):
        return (WAKE_COEFFICIENT * velocity) / (rps * diameter)
        
        
