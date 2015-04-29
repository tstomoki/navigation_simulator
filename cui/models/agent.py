# import common modules #
import sys
import math
import copy
import pdb
import os
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
        self.icr           = DEFAULT_ICR_RATE
        # initialize the range of velosity and rps
        self.velocity_array = np.arange(VELOCITY_RANGE['from'], VELOCITY_RANGE['to'], VELOCITY_RANGE['stride'])
        self.rpm_array      = np.arange(RPM_RANGE['from'], RPM_RANGE['to'], RPM_RANGE['stride'])            
        
        if (hull is None or engine is None or propeller is None):
            output_dir_path = "%s/%s" % (AGNET_LOG_DIR_PATH, generate_timestamp())
            initializeDirHierarchy(output_dir_path)
            NPV, self.hull, self.engine, self.propeller = self.get_initial_design(output_dir_path)
        else:
            self.hull, self.engine, self.propeller = hull, engine, propeller
       
           
    ### full search with hull, engine and propeller
    def get_initial_design(self, output_dir_path):
        # load components list
        hull_list           = load_hull_list()
        engine_list         = load_engine_list()
        propeller_list      = load_propeller_list()

        ## hull
        ### list has only 1 hull
        ret_hull = Hull(hull_list, 1)

        ## engine and propeller
        ### full search with sinario and world_scale
        self.sinario.generate_sinario(self.sinario_mode)
        ### default flat_rate is 50 [%]
        self.world_scale.set_flat_rate(50)

        dtype  = np.dtype({'names': ('hull_id', 'engine_id', 'propeller_id', 'NPV'),'formats': (np.int , np.int, np.int, np.float)})
        design_array = np.array([], dtype=dtype)
        for engine_info in engine_list:
            engine = Engine(engine_list, engine_info['id'])
            for propeller_info in propeller_list:
                propeller = Propeller(propeller_list, propeller_info['id'])
                # conduct simmulation
                NPV = self.simmulate(ret_hull, engine, propeller)
                # update design #
                add_design   = np.array([(NPV,
                                        ret_hull.base_data['id'],
                                        engine.base_data['id'],
                                        propeller.base_data['id'])],
                                        dtype=dtype)
                design_array = append_for_np_array(design, add_design)
                # update design #
                # write simmulation result
                output_file_path = "%s/%s" % (output_dir_path, 'initial_design.csv')
                write_csv(ret_hull, engine, propeller, NPV, output_file_path)
        # get design whose NPV is the maximum
        NPV, hull_id, engine_id, propeller_id = design_array[np.argmax(design_array, axis=0)[0]]
        hull      = Hull(hull_list, 1)
        engine    = Engine(engine_list, engine_id) 
        propeller = Propeller(propeller_list, propeller_id)
        return NPV, hull, engine, propeller

    def simmulate(self, hull, engine, propeller):
        # initialize retrofit_count
        if self.retrofit_mode == RETROFIT_MODE['none']:
            retrofit_count = 0

        # define velocity and rps for given [hull, engine, propeller]
        self.velocity_combination = self.create_velocity_combination(hull, engine, propeller)
            
        # load condition [ballast, full]
        self.load_condition = INITIAL_LOAD_CONDITION

        # static variables 
        self.operation_date_array  = self.generate_operation_date(self.sinario.predicted_data['date'][0])
        self.origin_date           = self.operation_date_array[0]
        self.retire_date           = self.operation_date_array[-1]
        self.round_trip_distance   = NAVIGATION_DISTANCE * 2.0
        self.NPV                   = np.array([],np.dtype({'names': ('navigation_finished_date', 'NPV_in_navigation'),
                                                           'formats': ('S20' , np.float)}))        
        self.log                   = init_dict_from_keys_with_array(LOG_COLUMNS,
                                                                    np.dtype({'names': ('rpm', 'velocity'),
                                                                              'formats': (np.float , np.float)}))
        self.total_cash_flow       = 0
        self.total_NPV             = 0
        self.total_distance        = 0
        self.total_elapsed_days    = 0

        # dynamic variables
        self.current_distance      = 0
        self.left_distance_to_port = NAVIGATION_DISTANCE
        self.voyage_date           = self.origin_date
        self.previous_oilprice     = self.sinario.history_data[-2]['price']
        self.oilprice_ballast      = self.sinario.history_data[-1]['price']
        self.oilprice_full         = self.sinario.history_data[-1]['price']
        self.current_fare          = self.world_scale.calc_fare(self.previous_oilprice)
        self.cash_flow             = 0
        self.loading_flag          = False
        self.loading_days          = 0
        self.elapsed_days          = 0
        self.latest_dockin_date    = self.origin_date
        self.dockin_flag           = False
        self.ballast_trip_days     = 0
        self.return_trip_days      = 0
        
        for current_date in self.operation_date_array:
            # update current_date
            self.current_date = current_date

            # for loading duration
            if self.is_loading():
                if self.is_ballast():
                    print_with_notice("unloading on %s" % (self.current_date))
                else:
                    print_with_notice("loading on %s" % (self.current_date))
                # update total elapsed days
                self.total_elapsed_days += 1
                continue

            # for dock-in
            if self.is_dockin():
                print_with_notice("docking in on %s" % (self.current_date))
                # update total elapsed days
                self.total_elapsed_days += 1
                continue

            # define voyage_date
            if self.voyage_date is None:
                self.voyage_date = self.current_date            
                
            # calculate optimized speed and rps
            CF_day, rpm, v_knot  = self.calc_optimal_velosity(hull, engine, propeller)
            ## update velocity log
            self.update_velocity_log(rpm, v_knot)
            
            # update variables
            ## update with the distance on a day
            navigated_distance = knot2mileday(v_knot)
            updated_distance   = self.current_distance + knot2mileday(v_knot)
            if (self.current_distance < NAVIGATION_DISTANCE) and (updated_distance >= NAVIGATION_DISTANCE):
                # ballast -> full
                print_with_notice("ballast trip finished on %s" % (self.current_date))
                # calc distance to the port
                navigated_distance = NAVIGATION_DISTANCE - self.current_distance                                
                # subtract unnavigated cash flow which depends on the distance
                discounted_distance = updated_distance - NAVIGATION_DISTANCE
                CF_day -= self.calc_fuel_cost_with_distance(discounted_distance, rpm, v_knot, hull, engine, propeller)
                
                self.current_distance      = NAVIGATION_DISTANCE
                self.left_distance_to_port = NAVIGATION_DISTANCE
                # update oil price
                self.update_oilprice_and_fare()
                
                # loading flags
                self.initiate_loading()
                self.change_load_condition()
        
            elif updated_distance >= self.round_trip_distance:
                # full -> ballast
                print_with_notice("Navigation finished on %s" % (self.current_date))
                # calc distance to the port
                navigated_distance = self.round_trip_distance - self.current_distance                
                # subtract unnavigated cash flow which depends on the distance
                discounted_distance = updated_distance - self.round_trip_distance
                CF_day -= self.calc_fuel_cost_with_distance(discounted_distance, rpm, v_knot, hull, engine, propeller)
                # loading flags (unloading)
                self.initiate_loading()
                self.change_load_condition()
                # update oil price'
                self.update_oilprice_and_fare()
                # reset current_distance
                self.current_distance = 0
                self.left_distance_to_port = NAVIGATION_DISTANCE

                # calc Net Present Value
                self.update_NPV_in_navigation()
                # NPV
                self.display_latest_NPV()
                # initialize the vairables
                self.current_distance = 0
                self.cash_flow     = 0
                self.elapsed_days  = 0
                self.voyage_date   = None
                self.left_distance_to_port = NAVIGATION_DISTANCE

                '''
                # for dev
                if self.current_date > datetime.date(2014, 1,30):
                    break
                '''
                # dock-in flag
                if self.update_dockin_flag():
                    self.initiate_dockin()
                continue
            else:
                # full -> full or ballast -> ballast
                self.current_distance      += navigated_distance
                self.left_distance_to_port -= navigated_distance
            
            # update total distance
            self.total_distance += navigated_distance
            
            # update days
            self.elapsed_days       += 1
            self.total_elapsed_days += 1            
            self.update_trip_days()

            # update cash flow
            self.cash_flow += CF_day
            self.total_cash_flow += CF_day

            # display current infomation
            print "--------------Finished Date: %s--------------" % (self.current_date)
            print "%25s: %10d [days]"     % ('ballast trip days'    , self.ballast_trip_days)
            print "%25s: %10d [days]"     % ('return trip days'     , self.return_trip_days)
            print "%25s: %10d [days]"     % ('elapsed_days'         , self.elapsed_days)
            print "%25s: %10d [days]"     % ('total_elapsed_days'   , self.total_elapsed_days)
            print "%25s: %10s"            % ('load condition'       , self.load_condition_to_human())
            print "%25s: %10.3lf [mile]"  % ('navigated_distance'   , navigated_distance)
            print "%25s: %10.3lf [mile]"  % ('current_distance'     , self.current_distance)
            print "%25s: %10.3lf [mile]"  % ('left_distance_to_port', self.left_distance_to_port)
            print "%25s: %10.3lf [mile]"  % ('total_distance'       , self.total_distance)
            print "%25s: %10.3lf [rpm]"   % ('rpm'                  , rpm)
            print "%25s: %10.3lf [knot]"  % ('velocity'             , v_knot)
            print "%25s: %10s [$/day]"    % ('Cash flow'            , number_with_delimiter(CF_day))
            print "%25s: %10s [$]"        % ('Total Cash flow'      , number_with_delimiter(self.total_cash_flow))

        # update whole NPV in vessel life time
        return self.update_whole_NPV()

    # define velocity and rps for given [hull, engine, propeller]
    def create_velocity_combination(self, hull, engine, propeller):
        ret_combinations = {}
        
        for load_condition in LOAD_CONDITION.keys():
            combinations = np.array([])
            for rpm in self.rpm_array:
                # reject rps over the engine Max rps
                if rpm >= engine.base_data['N_max']:
                    next
                rps = round(rpm / 60.0, 4)
                tmp_combinations = np.array([])
                for velocity in self.velocity_array:
                    velocity = round(velocity, 4)
    
                    # calc error of fitness bhp values
                    error = self.rps_velocity_fitness(hull, engine, propeller, velocity, rps, load_condition)
                    tmp_combinations = append_for_np_array(tmp_combinations, [rpm, velocity, error])
                # remove None error value
                remove_induces = np.array([])
                for index, element in enumerate(tmp_combinations):
                    if element[2] is None:
                        remove_induces = np.append(remove_induces, index)
                tmp_combinations = np.delete(tmp_combinations, remove_induces, 0)
                min_combination = tmp_combinations[np.argmin(tmp_combinations[:, 2])]

                combinations = append_for_np_array(combinations, [min_combination[0],min_combination[1]])
            ret_combinations[load_condition_to_human(load_condition)] = combinations

        return ret_combinations

    def rps_velocity_fitness(self, hull, engine, propeller, velocity, rps, load_condition):
        # calc bhp [WW]
        fitness_bhp0 = self.calc_bhp_with_rps(          rps, hull, engine, propeller)
        fitness_bhp1 = self.calc_bhp_with_ehp(velocity, rps, hull, engine, propeller, load_condition)

        # reject bhp over the engine Max load
        if fitness_bhp0 is None or fitness_bhp1 is None or fitness_bhp0 > engine.base_data['max_load'] or fitness_bhp1 > engine.base_data['max_load']:
            return None
                
        error = math.pow(fitness_bhp0 - fitness_bhp1, 2)
        error = math.sqrt(error)

        return error

    # return bhp [kW]
    def calc_bhp_with_rps(self, rps, hull, engine, propeller):
        # read coefficients
        bhp_coefficients = dict.fromkeys(['bhp0', 'bhp1', 'bhp2'], 0)
        for bhp_coefficients_key in bhp_coefficients.keys():
            bhp_coefficients[bhp_coefficients_key] = engine.base_data[bhp_coefficients_key]

        max_rps = round(rpm2rps(engine.base_data['N_max']), 4)
        bhp  = bhp_coefficients['bhp0'] + bhp_coefficients['bhp1'] * (rps / max_rps) + bhp_coefficients['bhp2'] * math.pow(rps / max_rps, 2)

        return bhp

    # return bhp [kW]    
    def calc_bhp_with_ehp(self, velocity, rps, hull, engine, propeller, load_condition):
        # reject if the condition (KT > 0 and eta > 0) fulfilled
        J   = self.calc_advance_constant(velocity, rps, propeller.base_data['D'])
        KT  = propeller.base_data['KT0'] + propeller.base_data['KT1'] * J + propeller.base_data['KT2'] * math.pow(J,2)
        eta = THRUST_COEFFICIENT * ( velocity / (2 * math.pi) ) * (1.0 / (rps * propeller.base_data['D']) ) * ( (KT) / (propeller.base_data['KQ0'] + propeller.base_data['KQ1'] * J + propeller.base_data['KQ2'] * math.pow(J,2)) )
        if KT < 0 or eta < 0:
            return None
        
        # read coefficients
        ehp_coefficients = dict.fromkeys(['ehp0', 'ehp1', 'ehp2', 'ehp3', 'ehp4'], 0)
        for ehp_coefficients_key in ehp_coefficients.keys():
            data_key = "%s_%s" % (ehp_coefficients_key, load_condition_to_human(load_condition))
            ehp_coefficients[ehp_coefficients_key] = hull.base_data[data_key]

        # advance constants
        J = self.calc_advance_constant(velocity, rps, propeller.base_data['D'])

        # calc numerator
        numerator =  ehp_coefficients['ehp0'] + ehp_coefficients['ehp1'] * velocity
        numerator += ehp_coefficients['ehp2'] * math.pow(velocity, 2) + ehp_coefficients['ehp3'] * math.pow(velocity, 3) + ehp_coefficients['ehp4'] * math.pow(velocity, 4)

        # calc denominator
        denominator = THRUST_COEFFICIENT * (velocity / (2 * math.pi) ) * (1 / (rps * propeller.base_data['D']) ) * ( (propeller.base_data['KT0'] + propeller.base_data['KT1'] * J + propeller.base_data['KT2'] * math.pow(J,2)) / (propeller.base_data['KQ0'] + propeller.base_data['KQ1'] * J + propeller.base_data['KQ2'] * math.pow(J,2)) ) * ETA_S

        bhp = numerator / denominator
        # return bhp [kW]

        return bhp    
    
    def calc_optimal_velosity(self, hull, engine, propeller):
        combinations        = np.array([])
        for rpm_first, velocity_first in self.velocity_combination[load_condition_to_human(self.load_condition)]:
            # ignore second parameter when the navigation is return trip
            if self.is_ballast():
                ## when the ship is ballast
                tmp_combinations = np.array([])
                # decide velocity of full                
                for rpm_second, velocity_second in self.velocity_combination[load_condition_to_human(get_another_condition(self.load_condition))]:
                    ND     = self.calc_ND(velocity_first, velocity_second, hull)
                    CF_day = self.calc_cash_flow(rpm_first, velocity_first, rpm_second, velocity_second, hull, engine, propeller, ND)
                    tmp_combinations = append_for_np_array(tmp_combinations, [CF_day, rpm_second, velocity_second])
                CF_day, optimal_rpm_full, optimal_velocity_full = tmp_combinations[np.argmax(tmp_combinations, axis=0)[0]]
            else:
                ## when the ship is full (return trip)
                ND     = self.calc_ND(velocity_first, 0, hull)
                CF_day = self.calc_cash_flow(rpm_first, velocity_first, 0, 0, hull, engine, propeller, ND)
            combinations = append_for_np_array(combinations, [CF_day, rpm_first, velocity_first])

        # decide the velocity
        CF_day, optimal_rpm, optimal_velocity = combinations[np.argmax(combinations, axis=0)[0]]
        return CF_day, optimal_rpm, optimal_velocity

    # calc advance constant
    def calc_advance_constant(self, velocity, rps, diameter):
        return (WAKE_COEFFICIENT * velocity) / (rps * diameter)
        
    def generate_operation_date(self, start_month):
        operation_date_array = np.array([])
        operation_start_date = first_day_of_month(str_to_datetime(start_month))
        operation_end_date   = add_year(operation_start_date, OPERATION_DURATION_YEARS)

        current_date = operation_start_date
        while True:
            operation_date_array = np.append(operation_date_array, current_date)
            current_date += datetime.timedelta(days=1)
            if current_date >= operation_end_date:
                break
        return operation_date_array
    
    # return ND [days]
    # ND is whole number of days in navigation
    def calc_ND(self, velocity_first, velocity_second, hull):
        # ignore second clause when 'full'
        if self.is_full():
            first_clause  = self.calc_left_distance() / knot2mileday(velocity_first)
            second_clause = 0
        else:
            first_clause  = (self.calc_left_distance() - NAVIGATION_DISTANCE) / knot2mileday(velocity_first)
            second_clause = NAVIGATION_DISTANCE / knot2mileday(velocity_second)

        ret_ND = self.elapsed_days + first_clause + second_clause
        return ret_ND
            
    def calc_cash_flow(self, rpm_first, velocity_first, rpm_second, velocity_second, hull, engine, propeller, ND):
        # Income_day
        I      = self.current_fare * hull.base_data['DWT']
        I_day  = I / float(ND)
        # Cost for fix_day
        C_fix  = self.calc_fix_cost()
        # Cost for port_day
        C_port = self.calc_port_cost(ND)
        
        # Fuel Consumption_day
        C_fuel = self.calc_fuel_cost(hull, engine, propeller, ND, rpm_first, velocity_first, rpm_second, velocity_second)
        cash_flow = (1 - self.icr) * I_day - C_fuel - C_fix - C_port

        return cash_flow

    # calc fuel cost per day    
    def calc_fuel_cost(self, hull, engine, propeller, ND, rpm_first, velocity_first, rpm_second, velocity_second):
        ret_fuel_cost = 0
        if self.is_ballast():
            # ballast
            bhp_ballast       = self.calc_bhp_with_rps(rpm2rps(rpm_first),  hull, engine, propeller)
            sfoc_ballast      = engine.calc_sfoc(bhp_ballast)
            bhp_full          = self.calc_bhp_with_rps(rpm2rps(rpm_second), hull, engine, propeller)
            sfoc_full         = engine.calc_sfoc(bhp_full)
            # calc fuel_cost per day
            fuel_cost_ballast = (1000 * self.oilprice_ballast) / 159.0 * bhp_ballast * sfoc_ballast * (24.0 / 1000000.0)
            fuel_cost_full    = (1000 * self.oilprice_full)    / 159.0 * bhp_full    * sfoc_full    * (24.0 / 1000000.0)

            # for the navigated distance
            if len(self.log[self.load_condition_to_human()]) == 0:
                first_clause = 0
            else:
                # use averaged rps for navigated distance
                averaged_v    = np.average(self.log[self.load_condition_to_human()]['velocity'])
                averaged_rpm  = np.average(self.log[self.load_condition_to_human()]['rpm'])
                averaged_bhp  = self.calc_bhp_with_rps(rpm2rps(averaged_rpm),  hull, engine, propeller)
                averaged_sfoc = engine.calc_sfoc(averaged_bhp)
                averaged_fuel_cost = (1000 * self.oilprice_ballast) / 159.0 * averaged_bhp * averaged_sfoc * (24.0 / 1000000.0)
                first_clause = averaged_fuel_cost * self.elapsed_days

            fuel_cost  = first_clause
            # go navigation
            fuel_cost += fuel_cost_ballast * ( (self.calc_left_distance() - NAVIGATION_DISTANCE) / knot2mileday(velocity_first) )
            # return navigation
            fuel_cost += fuel_cost_full * ( NAVIGATION_DISTANCE / knot2mileday(velocity_second) )
            # consider navigated 
            fuel_cost /= float(ND)
        else:
            # ballast
            # use averaged rps for navigated distance
            another_condition          = self.get_another_load_condition_to_human()
            averaged_v_ballast         = np.average(self.log[another_condition]['velocity'])
            averaged_rpm_ballast       = np.average(self.log[another_condition]['rpm'])
            averaged_bhp_ballast       = self.calc_bhp_with_rps(rpm2rps(averaged_rpm_ballast), hull, engine, propeller)
            averaged_sfoc_ballast      = engine.calc_sfoc(averaged_bhp_ballast)
            averaged_fuel_cost_ballast = (1000 * self.oilprice_ballast) / 159.0 * averaged_bhp_ballast * averaged_sfoc_ballast * (24.0 / 1000000.0)
            fuel_cost_ballast = averaged_fuel_cost_ballast * self.ballast_trip_days            
            
            # full
            bhp_full          = self.calc_bhp_with_rps(rpm2rps(rpm_first), hull, engine, propeller)
            sfoc_full         = engine.calc_sfoc(bhp_full)
            # calc fuel_cost per day
            fuel_cost_full    = (1000 * self.oilprice_full) / 159.0 * bhp_full * sfoc_full * (24.0 / 1000000.0)

            # for the navigated distance
            if len(self.log[self.load_condition_to_human()]) == 0:
                first_clause = 0
            else:
                # use averaged rps for navigated distance
                averaged_v    = np.average(self.log[self.load_condition_to_human()]['velocity'])
                averaged_rpm  = np.average(self.log[self.load_condition_to_human()]['rpm'])
                averaged_bhp  = self.calc_bhp_with_rps(rpm2rps(averaged_rpm), hull, engine, propeller)
                averaged_sfoc = engine.calc_sfoc(averaged_bhp)
                averaged_fuel_cost = (1000 * self.oilprice_full) / 159.0 * averaged_bhp * averaged_sfoc * (24.0 / 1000000.0)
                first_clause = averaged_fuel_cost * self.elapsed_days

            # return navigation                
            fuel_cost  = first_clause
            fuel_cost += fuel_cost_full * ( (self.calc_left_distance()) / knot2mileday(velocity_first) )

            # add ballast fuel cost
            fuel_cost += fuel_cost_ballast
            # consider navigated 
            fuel_cost /= float(ND)
        
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

    def is_ballast(self):
        return is_ballast(self.load_condition)

    def is_full(self):
        return is_full(self.load_condition)

    def calc_left_distance(self):
        return self.round_trip_distance - self.current_distance

    def load_condition_to_human(self):
        return load_condition_to_human(self.load_condition)

    def get_another_load_condition(self):
        return get_another_condition(self.load_condition)

    def get_another_load_condition_to_human(self):
        return load_condition_to_human(self.get_another_load_condition())

    def update_velocity_log(self, rpm, v_knot):
        add_array       = np.array([rpm, v_knot])
        add_array.dtype = np.dtype({'names': ('rpm', 'velocity'), 'formats': (np.float , np.float)})
        self.log[self.load_condition_to_human()] = append_for_np_array(self.log[self.load_condition_to_human()], add_array)
        return

    def update_NPV_log(self, NPV_in_navigation):
        dtype     = np.dtype({'names': ('navigation_finished_date', 'NPV_in_navigation'),'formats': ('S20' , np.float)})
        add_array = np.array([(datetime_to_human(self.current_date), NPV_in_navigation)], dtype=dtype)
        self.NPV = append_for_np_array(self.NPV, add_array)
        return

    def change_load_condition(self):
        self.load_condition = self.get_another_load_condition()
        return

    def is_loading(self):
        if (self.loading_flag and self.loading_days < LOAD_DAYS):
            self.loading_days += 1
            return True
        else:
            self.loading_flag = False
        
        return 

    def initiate_loading(self):
        self.loading_flag = True
        self.loading_days = 0
        return
    
    # calc cash flow which depends on the distance
    def calc_fuel_cost_with_distance(self, discounted_distance, rpm, v_knot, hull, engine, propeller):
        ret_fuel_cost = 0
        
        # load oil price based on the load_condition
        oilprice = self.oilprice_ballast if self.is_ballast() else self.oilprice_full
        # calc fuel_cost per day
        bhp           = self.calc_bhp_with_rps(rpm2rps(rpm), hull, engine, propeller)
        sfoc          = engine.calc_sfoc(bhp)
        fuel_cost     = (1000 * oilprice) / 159.0 * bhp * sfoc * (24.0 / 1000000.0)
        ret_fuel_cost = fuel_cost * ( (discounted_distance) / knot2mileday(v_knot) )
        
        return ret_fuel_cost

    def update_trip_days(self):
        if self.is_ballast():
            self.ballast_trip_days += 1
        else:
            self.return_trip_days += 1            
        return

    def calc_NPV(self, CF_day, v_knot):
        left_days = self.calc_left_days_in_navigation(v_knot)
        PV        = self.calc_present_value_in_navigation(left_days, CF_day)
        whole_PV  = self.calc_present_value(PV)
        pass

    def update_oilprice_and_fare(self):
        # update oilprice
        current_date_index        = search_near_index(self.current_date, self.sinario.predicted_data['date'])
        self.previous_oilprice    = self.get_previous_oilprice(current_date_index)
        current_index,            = np.where(self.sinario.predicted_data['date']==current_date_index)
        if self.is_ballast():
            self.oilprice_full    = self.sinario.predicted_data[current_index[0]]['price']
        else:
            self.oilprice_ballast = self.sinario.predicted_data[current_index[0]]['price']

        # update fare
        self.current_fare         = self.world_scale.calc_fare(self.previous_oilprice)

        # for once debug (remove when confirm)#
        if not self.oilprice_ballast == self.oilprice_full:
            print self.oilprice_ballast
            print self.oilprice_full
            pdb.set_trace()
        
        return

    def get_previous_oilprice(self, date_index):
        index, = np.where(self.sinario.predicted_data['date']==date_index)
        if len(index) == 0:
            raise
        elif index[0] == 0:
            ret_oilprice = self.sinario.history_data[-1]['price']
        else:
            ret_oilprice = self.sinario.predicted_data[index[0]]['price']
            
        return ret_oilprice

    def calc_left_days_in_navigation(self, v_knot):
        left_distance_in_navigation = self.calc_left_distance()
        left_days = left_distance_in_navigation / knot2mileday(v_knot)
        return left_days
    
    def calc_left_days(self):
        pdb.set_trace()
        return    

    # PV = CF_in_navigation / (1-Discount_rate)^elapsed_month
    def update_NPV_in_navigation(self):
        denominator       = math.pow( (1 - DISCOUNT_RATE), self.calc_elapsed_month())
        numerator         = self.cash_flow
        NPV_in_navigation = numerator / denominator
        self.update_NPV_log(NPV_in_navigation)
        return
                
    def calc_present_value_in_navigation(self, left_days, CF_day):
        ret_present_value  = 0
        ret_present_value += self.cash_flow
        ret_present_value += CF_day * left_days
        return ret_present_value    


    def calc_present_value(self, PV):
        ret_present_value  = 0
        ret_present_value += self.cash_flow
        ret_present_value += CF_day * left_days
        return ret_present_value            

    # including a potential error
    def calc_elapsed_month(self):
        days_delta_from_voyage = (self.current_date - self.voyage_date).days
        days_num               = get_days_num_in_the_month(self.current_date)
        ret_elapsed_month      = days_delta_from_voyage / float(days_num)
        return ret_elapsed_month

    def display_latest_NPV(self):
        try:
            date, npv = self.NPV[-1]
        except:
            date, npv = self.NPV[-1][0]
        print_with_notice("Present Value on %15s: %20s" % (date, number_with_delimiter(npv)))
        return 
        
    def update_whole_NPV(self):
        self.total_NPV = np.sum(self.NPV['NPV_in_navigation'])
        return self.total_NPV

    def update_dockin_flag(self):
        latest_dockin_date = copy.deepcopy(self.latest_dockin_date)
        next_dockin_date   = add_year(latest_dockin_date, DOCK_IN_PERIOD)
        return self.current_date >= next_dockin_date

    def initiate_dockin(self):
        left_dock_date          = add_month(copy.deepcopy(self.current_date), DOCK_IN_DURATION)
        self.latest_dockin_date = left_dock_date
        self.dockin_flag = True
        return

    def is_dockin(self):
        return self.dockin_flag and (self.current_date <= self.latest_dockin_date)
