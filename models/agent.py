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
        self.icr           = DEFAULT_ICR
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
        self.velocity_combination = self.create_velocity_combination(hull, engine, propeller)
            
        # load condition [ballast, full]
        self.load_condition = INITIAL_LOAD_CONDITION

        # voyage date
        self.operation_date_array = self.generate_operation_date(self.sinario.predicted_data['date'][0])
        self.origin_date          = self.operation_date_array[0]
        self.current_distance     = 0
        self.voyage_date          = self.origin_date
        self.previous_oil_price   = self.sinario.history_data[-2]['price']
        self.oil_price            = self.sinario.history_data[-1]['price']
        self.current_fare         = self.world_scale.calc_fare(self.previous_oil_price)
        self.log                  = init_dict_from_keys_with_array(LOAD_CONDITION.values())
        for current_date in self.operation_date_array:
            # define voyage_date
            if self.voyage_date is None:
                self.voyage_date = current_date
                
            # calculate optimized speed and rps
            self.elapsed_days = 0
            v_knot, rpm  = self.calc_optimal_velosity(current_date, hull, engine)

            # update
            current_distance += 0
            if len(self.log[load_condition_to_human(self.load_condition)]) == 0:
                self.log[load_condition_to_human(self.load_condition)] = np.append(self.log[load_condition_to_human(self.load_condition)], [v_knot, rpm])
            else:
                self.log[load_condition_to_human(self.load_condition)] = np.vstack((self.log[load_condition_to_human(self.load_condition)], [v_knot, rpm]))

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
            ret_combinations[load_condition_to_human(load_condition)] = combinations

        return ret_combinations

    def rps_velocity_fitness(self, hull, engine, propeller, velocity, rps, load_condition):
        # calc bhp [WW]
        fitness_bhp0 = self.calc_bhp_with_rps(          rps, hull, engine, propeller)
        fitness_bhp1 = self.calc_bhp_with_ehp(velocity, rps, hull, engine, propeller, load_condition)

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
            data_key = "%s_%s" % (ehp_coefficients_key, load_condition_to_human(load_condition))
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
    
    def calc_optimal_velosity(self, current_date, hull, engine):
        combinations        = np.array([])
        round_trip_distance = NAVIGATION_DISTANCE * 2.0
        left_distance       = round_trip_distance - self.current_distance
        for rpm_first, velocity_first in self.velocity_combination[load_condition_to_human(self.load_condition)]:
            # ignore second parameter when the navigation is return trip
            if is_ballast(self.load_condition):
                ## when the ship is ballast
                tmp_combinations = np.array([])
                for rpm_second, velocity_second in self.velocity_combination[load_condition_to_human(get_another_condition(self.load_condition))]:
                    ND     = self.calc_ND(left_distance, velocity_first, velocity_second, False, hull)
                    CF_day = self.calc_cash_flow(current_date, velocity_first, velocity_second, hull, engine, ND)
                    pdb.set_trace()
            else:
                ## when the ship is full (return trip)
                ND     = self.calc_ND(left_distance, velocity_first, 0, True, hull)
                CF_day = self.calc_cash_flow(current_date, velocity_first, 0, hull, engine, ND)
                pdb.set_trace()
                
        return v_knot, rpm            

    # calc advance constant
    def calc_advance_constant(self, velocity, rps, diameter):
        return (WAKE_COEFFICIENT * velocity) / (rps * diameter)
        
    def generate_operation_date(self, start_month):
        operation_date_array = np.array([])
        operation_start_date = first_day_of_month(datetime.datetime.strptime(start_month, "%Y/%m/%d"))
        operation_end_date   = add_year(operation_start_date, OPERATION_DURATION_YEARS)

        current_date = operation_start_date
        while True:
            operation_date_array = np.append(operation_date_array, current_date)
            current_date += datetime.timedelta(days=1)
            if current_date >= operation_end_date:
                break
        return operation_date_array
    
    # return ND [days]
    def calc_ND(self, left_distance, velocity_first, velocity_second, ignore_flag, hull):
        # ignore second clause when ignore_flag is True
        if ignore_flag:
            first_clause  = left_distance / knot2mileday(velocity_first)
            second_clause = 0
        else:
            first_clause  = (left_distance - NAVIGATION_DISTANCE) / knot2mileday(velocity_first)
            second_clause = NAVIGATION_DISTANCE / knot2mileday(velocity_second)

        return self.elapsed_days + first_clause + second_clause
            
    def calc_cash_flow(self, current_date, velocity_first, velocity_second, hull, engine, ND):
        # Income_day
        I      = self.current_fare * hull.base_data['DWT']
        I_day  = I / float(ND)
        # Fuel Consumption_day
        C_fuel = self.calc_fuel_cost(current_date, engine, ND, velocity_first, velocity_second)
        # Cost for fix_day
        C_fix  = self.calc_fix_cost()
        # Cost for port_day
        C_port = self.calc_port_cost(ND)
        
        return (1 - DEFAULT_ICR_RATE) * I_day - C_fuel - C_fix - C_port

    # calc fuel cost per day    
    def calc_fuel_cost(self, current_date, engine, ND, velocity_first, velocity_second):
        ret_fuel_cost = 0
        if is_ballast(self.load_condition):
            # ballast
            fuel_cost_ballast = 0
            fuel_cost_full    = 0
            
        else:
            # full
        
        return fuel_cost
    
    # calc fix cost per day
    def calc_fix_cost(self):
        total_fix_cost = 0
        # dry dock maintenance [USD/year] #
        total_fix_cost += DRY_DOCK_MAINTENANCE
        # maintenance [USD/year] #
        total_fix_cost += MAINTENANCE
        # crew labor cost [USD/year] #
        total_fix_cost += CREW_LABOR_COST
        # Insurance [USD/year] #
        total_fix_cost += INSURANCE

        # convert unit to [USD/day]
        total_fix_cost /= 365.0
        
        return total_fix_cost

    # calc port cost per day
    def calc_port_cost(self, ND):
        # Cport * PORT_DWELL_DAYS * 2 / ND
        total_port_cost = PORT_CHARGES * PORT_DWELL_DAYS * 2 / ND
        return total_port_cost    
