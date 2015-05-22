# import common modules #
import sys
import math
import copy
import pdb
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from types import *
import json
# import common modules #

# append the load path
sys.path.append('../public')
sys.path.append('../models')
# append the load path

# for multi processing
import multiprocessing as mp
import multiprocess_with_instance_methods
# for multi processing

# import own modules #
from my_modules import *
from constants  import *
# import own modules #

# constants #
RESULTSDIR = '../results/'
# constants #

# import models #
from hull        import Hull
from sinario     import Sinario
from engine      import Engine
from propeller   import Propeller
from world_scale import WorldScale
# import models #

class Agent(object):
    def __init__(self, sinario, world_scale, retrofit_mode, sinario_mode, hull=None, engine=None, propeller=None, rpm_array=None, velocity_array=None):
        self.sinario       = sinario
        self.world_scale   = world_scale
        self.retrofit_mode = retrofit_mode
        self.sinario_mode  = sinario_mode
        self.icr           = DEFAULT_ICR_RATE
        self.operation_date_array = None
        # initialize the range of velosity and rps
        
        # for velocity and rps array #
        self.velocity_array = np.arange(DEFAULT_VELOCITY_RANGE['from'], DEFAULT_VELOCITY_RANGE['to'], DEFAULT_VELOCITY_RANGE['stride']) if velocity_array is None else velocity_array
        self.rpm_array      = np.arange(DEFAULT_RPM_RANGE['from'], DEFAULT_RPM_RANGE['to'], DEFAULT_RPM_RANGE['stride']) if rpm_array is None else rpm_array
        # for velocity and rps array #

        ### full search with sinario and world_scale
        self.sinario.generate_sinario(self.sinario_mode)
        self.world_scale.generate_sinario(self.sinario_mode)

        if not (hull is None or engine is None or propeller is None):
            self.hull, self.engine, self.propeller = hull, engine, propeller

    # display base data
    def display_variables(self):
        for variable_key in self.__dict__.keys():
            instance_variable_key = "self.%s" % (variable_key)
            instance_variable     = eval(instance_variable_key)
            if isinstance(instance_variable, NoneType):
                print "%25s: %20s" % (instance_variable_key, 'NoneType')
            elif isinstance(instance_variable, np.ndarray):
                print "%25s: %20s" % (instance_variable_key, 'Numpy with length (%d)' % (len(instance_variable)))                
            elif isinstance(instance_variable, DictType):
                key_str = ', '.join([_k for _k in instance_variable.keys()])
                print "%25s: %20s" % (instance_variable_key, 'DictType with keys % 10s' % (key_str))
            else:
                print "%25s: %20s" % (instance_variable_key, str(instance_variable))                
        return
           
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

        dtype  = np.dtype({'names': ('NPV', 'hull_id', 'engine_id', 'propeller_id'),'formats': (np.float, np.int , np.int, np.int)})
        design_array    = np.array([], dtype=dtype)
        simmulate_count = 0
        column_names    = ['simmulate_count',
                           'ret_hull',
                           'engine',
                           'propeller',
                           'NPV',
                           'processing_time']

        start_time = time.clock()
        for engine_info in engine_list:
            engine = Engine(engine_list, engine_info['id'])
            for propeller_info in propeller_list:
                propeller = Propeller(propeller_list, propeller_info['id'])
                print_with_notice("conducting the %10d th combination" % (simmulate_count + 1))
                # conduct simmulation
                NPV = self.simmulate(ret_hull, engine, propeller)
                # update design #
                add_design   = np.array([(NPV,
                                        ret_hull.base_data['id'],
                                        engine.base_data['id'],
                                        propeller.base_data['id'])],
                                        dtype=dtype)
                design_array = append_for_np_array(design_array, add_design)
                # update design #
                # write simmulation result
                output_file_path = "%s/%s" % (output_dir_path, 'initial_design.csv')
                lap_time         = convert_second(time.clock() - start_time)
                write_csv(column_names, [simmulate_count, ret_hull.base_data['id'], engine.base_data['id'], propeller.base_data['id'], NPV, lap_time], output_file_path)
                simmulate_count += 1
                    
        # get design whose NPV is the maximum
        NPV, hull_id, engine_id, propeller_id = design_array[np.argmax(design_array, axis=0)[0]]
        hull      = Hull(hull_list, 1)
        engine    = Engine(engine_list, engine_id) 
        propeller = Propeller(propeller_list, propeller_id)
        return NPV, hull, engine, propeller

    ### full search with hull, engine and propeller
    # multi processing method #
    def get_initial_design_m(self, output_dir_path):
        # load components list
        hull_list           = load_hull_list()
        engine_list         = load_engine_list()
        propeller_list      = load_propeller_list()

        ## hull
        ### list has only 1 hull
        ret_hull = Hull(hull_list, 1)

        ### default flat_rate is 50 [%]
        self.flat_rate = 50
        
        # devide the range of propeller list
        propeller_combinations = np.array_split(propeller_list, PROC_NUM)
        
        # initialize
        pool = mp.Pool(PROC_NUM)

        # multi processing #
        callback              = [pool.apply_async(self.calc_initial_design_m, args=(index, propeller_combinations, ret_hull, engine_list, propeller_list, output_dir_path)) for index in xrange(PROC_NUM)]
        callback_combinations = [p.get() for p in callback]
        ret_combinations      = flatten_3d_to_2d(callback_combinations)
        pool.close()
        pool.join()
        # multi processing #
        
        # get design whose NPV is the maximum
        if len(ret_combinations) == 0:
            print_with_notice("ERROR: Couldn't found initial design, abort")
            sys.exit()

        # write whole simmulation result
        output_file_path = "%s/%s" % (output_dir_path, 'initial_design.csv')
        column_names    = ['hull_id',
                           'engine_id',
                           'propeller_id',
                           'NPV']
        write_array_to_csv(column_names, ret_combinations, output_file_path)

        NPV, hull_id, engine_id, propeller_id = ret_combinations[np.argmax(ret_combinations['NPV'], axis=0)[0]][0]
        hull                                  = Hull(hull_list, 1)
        engine                                = Engine(engine_list, engine_id) 
        propeller                             = Propeller(propeller_list, propeller_id)

        return NPV, hull, engine, propeller

    def simmulate(self, hull=None, engine=None, propeller=None, multi_flag=None):
        # use instance variables if hull or engine or propeller are not given
        if (hull is None) or (engine is None) or (propeller is None):
            hull      = self.hull
            engine    = self.engine
            propeller = self.propeller

        # initialize retrofit_count
        if self.retrofit_mode == RETROFIT_MODE['none']:
            self.retrofit_count = 0
        else:
            self.retrofit_count = 1

        # define velocity and rps for given [hull, engine, propeller]
        self.velocity_combination = self.create_velocity_combination(hull, engine, propeller)
        # abort if proper velocity combination is calculated #
        if self.check_abort_simmulation():
            # return None PV if the simmulation is aborted
            notice_str = "Simmulation aborted for [hull: %d, engine: %d, propeller: %d]" % (hull.base_data['id'], propeller.base_data['id'], engine.base_data['id'])
            print_with_notice(notice_str)
            return None
            
        # load condition [ballast, full]
        self.load_condition = INITIAL_LOAD_CONDITION

        # static variables
        if self.operation_date_array is None:
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
        self.flat_rate             = self.world_scale.history_data[-1]['ws']
        self.current_fare          = self.world_scale.calc_fare(self.previous_oilprice, self.flat_rate)
        self.cash_flow             = 0
        self.loading_flag          = False
        self.loading_days          = 0
        self.elapsed_days          = 0
        self.latest_dockin_date    = self.origin_date
        self.dockin_flag           = False
        self.ballast_trip_days     = 0
        self.return_trip_days      = 0
        
        # initialize the temporal variables
        CF_day = rpm = v_knot = None                        
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
                
            # calculate optimized speed and rps during 
            if (CF_day is None) and (rpm is None) and (v_knot is None):
                if multi_flag:
                    CF_day, rpm, v_knot  = self.calc_optimal_velocity_m(hull, engine, propeller)
                else:
                    CF_day, rpm, v_knot  = self.calc_optimal_velocity(hull, engine, propeller)


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

                # update cash flow
                self.cash_flow       += CF_day
                self.total_cash_flow += CF_day
                
                # initialize the temporal variables
                CF_day = rpm = v_knot = None                
        
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

                # dock-in flag
                if self.update_dockin_flag():
                    self.initiate_dockin()
                    self.check_retrofit()

                # update cash flow
                self.cash_flow       += CF_day
                self.total_cash_flow += CF_day
                
                # initialize the temporal variables
                CF_day = rpm = v_knot = None
                self.ballast_trip_days = 0
                self.return_trip_days  = 0
                continue
            else:
                # full -> full or ballast -> ballast
                self.current_distance      += navigated_distance
                self.left_distance_to_port -= navigated_distance
                # update cash flow
                self.cash_flow       += CF_day
                self.total_cash_flow += CF_day                
            
            # update total distance
            self.total_distance += navigated_distance
            
            # update days
            self.elapsed_days       += 1
            self.total_elapsed_days += 1            
            self.update_trip_days()

            # display current infomation
            print "--------------Finished Date: %s--------------" % (self.current_date)
            print "%25s: %10d"            % ('Hull ID'              , hull.base_data['id'])
            print "%25s: %10d"            % ('Engine ID'            , engine.base_data['id'])
            print "%25s: %10d"            % ('Propeller ID'         , propeller.base_data['id'])
            print "%25s: %10d [days]"     % ('ballast trip days'    , self.ballast_trip_days)
            print "%25s: %10d [days]"     % ('return trip days'     , self.return_trip_days)
            print "%25s: %10d [days]"     % ('elapsed_days'         , self.elapsed_days)
            print "%25s: %10d [days]"     % ('total_elapsed_days'   , self.total_elapsed_days)
            print "%25s: %10s"            % ('load condition'       , self.load_condition_to_human())
            print "%25s: %10.3lf [mile]"  % ('navigated_distance'   , navigated_distance)
            print "%25s: %10.3lf [mile]"  % ('current_distance'     , self.current_distance)
            print "%25s: %10.3lf [mile]"  % ('left_distance_to_port', self.left_distance_to_port)
            print "%25s: %10.3lf [mile]"  % ('total_distance'       , self.total_distance)
            print "%25s: %10s [rpm]"      % ('rpm'                  , ("%10.3lf" % rpm)    if not rpm is None else '----')
            print "%25s: %10s [knot]"     % ('velocity'             , ("%10.3lf" % v_knot) if not v_knot is None else '----')
            print "%25s: %10s [$/day]"    % ('Cash flow'            , number_with_delimiter(CF_day) if not CF_day is None else '----')
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
                    velocity    = round(velocity, 4)
                    velocity_ms = knot2ms(velocity)
    
                    # calc error of fitness bhp values
                    error = self.rps_velocity_fitness(hull, engine, propeller, velocity_ms, rps, load_condition)
                    tmp_combinations = append_for_np_array(tmp_combinations, [rpm, velocity, error])
                # remove None error value
                remove_induces = np.array([])
                for index, element in enumerate(tmp_combinations):
                    if element[2] is None:
                        remove_induces = np.append(remove_induces, index)
                tmp_combinations = np.delete(tmp_combinations, remove_induces, 0)
                # for no combinations case
                if len(tmp_combinations) > 0:
                    min_combination = tmp_combinations[np.argmin(tmp_combinations[:, 2])]
                    combinations = append_for_np_array(combinations, [min_combination[0],min_combination[1]])
                    ret_combinations[load_condition_to_human(load_condition)] = combinations
                else:
                    # beyond the potential of the components (e.g. max_load or so on)
                    '''
                    print 'beyond the potential of components:'
                    print "%s: %s, %s: %s, %s: %s" % ('Hull'     , self.hull.base_data['id'],
                                                      'Engine'   , self.engine.base_data['id'],
                                                      'Propeller', self.propeller.base_data['id'])
                    '''
                    pass
        # draw RPS-velocity combinations
        self.draw_combinations(hull, engine, propeller, ret_combinations)
        # output json as file
        self.write_combinations_as_json(hull, engine, propeller, ret_combinations)
        return ret_combinations

    def rps_velocity_fitness(self, hull, engine, propeller, velocity_ms, rps, load_condition):
        # calc bhp [WW]
        fitness_bhp0 = self.calc_bhp_with_rps(          rps, hull, engine, propeller)
        fitness_bhp1 = self.calc_bhp_with_ehp(velocity_ms, rps, hull, engine, propeller, load_condition)
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
    def calc_bhp_with_ehp(self, velocity_ms, rps, hull, engine, propeller, load_condition):
        # reject if the condition (KT > 0 and eta > 0) fulfilled
        # advance constants
        J   = self.calc_advance_constant(velocity_ms, rps, propeller.base_data['D'])

        KT  = propeller.base_data['KT0'] + propeller.base_data['KT1'] * J + propeller.base_data['KT2'] * math.pow(J,2)
        eta = THRUST_COEFFICIENT * ( velocity_ms / (2 * math.pi) ) * (1.0 / (rps * propeller.base_data['D']) ) * ( (KT) / (propeller.base_data['KQ0'] + propeller.base_data['KQ1'] * J + propeller.base_data['KQ2'] * math.pow(J,2)) )
        if KT < 0 or eta < 0:
            return None
        
        # read coefficients
        ehp_coefficients = dict.fromkeys(['ehp0', 'ehp1', 'ehp2', 'ehp3', 'ehp4'], 0)
        for ehp_coefficients_key in ehp_coefficients.keys():
            data_key = "%s_%s" % (ehp_coefficients_key, load_condition_to_human(load_condition))
            ehp_coefficients[ehp_coefficients_key] = hull.base_data[data_key]

        # calc numerator
        numerator =  ehp_coefficients['ehp0'] + ehp_coefficients['ehp1'] * velocity_ms
        numerator += ehp_coefficients['ehp2'] * math.pow(velocity_ms, 2) + ehp_coefficients['ehp3'] * math.pow(velocity_ms, 3) + ehp_coefficients['ehp4'] * math.pow(velocity_ms, 4)

        # calc denominator
        denominator = THRUST_COEFFICIENT * (velocity_ms / (2 * math.pi) ) * (1 / (rps * propeller.base_data['D']) ) * ( (propeller.base_data['KT0'] + propeller.base_data['KT1'] * J + propeller.base_data['KT2'] * math.pow(J,2)) / (propeller.base_data['KQ0'] + propeller.base_data['KQ1'] * J + propeller.base_data['KQ2'] * math.pow(J,2)) ) * ETA_S

        bhp = numerator / denominator
        # return bhp [kW]

        return bhp    
    
    def calc_optimal_velocity(self, hull, engine, propeller):
        combinations        = np.array([])
        # cull the combination for the fast cunduct #
        target_combination = self.cull_combination()
        for rpm_first, velocity_first in target_combination[load_condition_to_human(self.load_condition)]:
            # ignore second parameter when the navigation is return trip
            if self.is_ballast():
                ## when the ship is ballast
                tmp_combinations = np.array([])
                # decide velocity of full                
                for rpm_second, velocity_second in target_combination[load_condition_to_human(get_another_condition(self.load_condition))]:
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

    # multi processing method #
    def calc_optimal_velocity_m(self, hull, engine, propeller):
        # devide the range
        combinations         = self.velocity_combination[load_condition_to_human(self.load_condition)]
        devided_combinations = np.array_split(combinations, PROC_NUM)

        # initialize
        pool = mp.Pool(PROC_NUM)

        # multi processing #
        callback = [pool.apply_async(self.calc_combinations, args=(index, devided_combinations, hull, engine, propeller)) for index in xrange(PROC_NUM)]
        callback_combinations = [p.get() for p in callback]
        ret_combinations      = flatten_3d_to_2d(callback_combinations)
        pool.close()
        pool.join()
        # multi processing #
        
        # decide the velocity
        CF_day, optimal_rpm, optimal_velocity = ret_combinations[np.argmax(ret_combinations, axis=0)[0]]
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
                first_clause = averaged_fuel_cost * self.ballast_trip_days

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
            fuel_cost_ballast = averaged_fuel_cost_ballast * self.return_trip_days
            
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

        # update world_scale (flate_rate)
        current_index,            = np.where(self.world_scale.predicted_data['date']==current_date_index)
        try:
            self.flat_rate = round(self.world_scale.predicted_data[current_index]['ws'][0], 4)
        except:
            print "debug: %s is not in %s" % (current_index, "self.world_scale.predicted_data")
            
        # update fare
        self.current_fare         = self.world_scale.calc_fare(self.previous_oilprice, self.flat_rate)

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

    # return True if velocity combination is not proper
    def check_abort_simmulation(self):
        # abort if 'ballast' and 'full' conditions had been calculated
        if len(self.velocity_combination.keys()) != len(LOAD_CONDITION.keys()):
            return True

        # abort if array length is not sufficient
        for load_condition_key in LOAD_CONDITION.values():
            if len(self.velocity_combination[load_condition_key]) < (len(self.rpm_array) / MINIMUM_ARRAY_REQUIRE_RATE):
                return True

        return False

    ## multi processing method ##
    def calc_combinations(self, index, devided_combinations, hull, engine, propeller):
        combinations = np.array([])
        for rpm_first, velocity_first in devided_combinations[index]:
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
        return combinations
    
    def calc_initial_design_m(self, index, propeller_combinations, ret_hull, engine_list, propeller_list, output_dir_path):
        column_names    = ['simmulate_count',
                           'hull_id',
                           'engine_id',
                           'propeller_id',
                           'NPV']
        dtype  = np.dtype({'names': ('NPV', 'hull_id', 'engine_id', 'propeller_id'),'formats': (np.float, np.int , np.int, np.int)})
        design_array = np.array([], dtype=dtype)
        start_time = time.clock()
        simmulate_count = 0
        for propeller_info in propeller_combinations[index]:
            propeller = Propeller(propeller_list, propeller_info['id'])
            for engine_info in engine_list:
                engine = Engine(engine_list, engine_info['id'])
                # create each arrays #
                rpm_array = np.arange(DEFAULT_RPM_RANGE['from'], engine.base_data['N_max'], RPM_RANGE_STRIDE)
                # conduct simmulation
                agent = Agent(self.sinario, self.world_scale, self.retrofit_mode, self.sinario_mode, ret_hull, engine, propeller, rpm_array)
                NPV   = agent.simmulate()
                # ignore aborted simmulation
                if NPV is None:
                    simmulate_count += 1
                    continue
                # update design #
                add_design   = np.array([(NPV,
                                        ret_hull.base_data['id'],
                                        engine.base_data['id'],
                                        propeller.base_data['id'])],
                                        dtype=dtype)
                design_array = append_for_np_array(design_array, add_design)
                # update design #
                # write simmulation result
                output_file_path = "%s/%s_core%d.csv" % (output_dir_path, 'initial_design', index)
                lap_time         = convert_second(time.clock() - start_time)
                write_csv(column_names, [simmulate_count, ret_hull.base_data['id'], engine.base_data['id'], propeller.base_data['id'], NPV, lap_time], output_file_path)
                simmulate_count += 1
        return design_array    
    ## multi processing method ##

    # draw rps - velocity combinations
    def draw_combinations(self, hull, engine, propeller, combinations):
        combination_str = self.generate_combination_str(hull, engine, propeller)        
        dir_name        = "%s/%s"     % (COMBINATIONS_DIR_PATH, combination_str)
        filename        = "%s/%s.png" % (dir_name, combination_str)
        title           = "%s of %s"  % ("revolution and velocity combination".title(),
                                         combination_str)

        x_label = "rps".upper()
        y_label = "%s %s" % ('velocity'.upper(), '[knot]')
        # initialize directory
        initializeDirHierarchy(dir_name)
        graphInitializer(title,
                         x_label,
                         y_label)
        colors = ['b', 'r']
        for key, load_condition in LOAD_CONDITION.items():
            draw_data = combinations[load_condition]
            x_data    = draw_data.transpose()[0]
            y_data    = draw_data.transpose()[1]
            plt.scatter(x_data, y_data, label=load_condition, color=colors[key])
        plt.legend(shadow=True)
        plt.legend(loc='upper left')
        plt.ylim([0, 30])
        plt.savefig(filename)
        plt.close()
        return

    def write_combinations_as_json(self, hull, engine, propeller, combinations):
        combination_str = self.generate_combination_str(hull, engine, propeller)
        dir_name        = "%s/%s"     % (COMBINATIONS_DIR_PATH, combination_str)
        output_path     = "%s/%s_combinations.json" % (dir_name, combination_str)

        # to serialize numpy ndarray #
        json_serialized_combinations = {}
        for condition in combinations.keys():
            json_serialized_combinations[condition] = combinations[condition].tolist()

        f = open(output_path, 'w')
        json_data = json.dumps(json_serialized_combinations, indent=4)
        f.write(json_data)
        f.close()        
        return

    def generate_combination_str(self, hull, engine, propeller):
        return "H%dE%dP%d" % (hull.base_data['id'], engine.base_data['id'], propeller.base_data['id'])

    def generate_combination_str_with_id(self, hull_id, engine_id, propeller_id):
        return "H%dE%dP%d" % (hull_id, engine_id, propeller_id)


    def cull_combination(self):
        velocity_log        = self.get_log_of_current_condition()
        target_combinations = self.velocity_combination
        if not len(velocity_log) == 0:
            target_combinations = {}
            latest_velocity_log = velocity_log[-1]['velocity']
            for load_condition in LOAD_CONDITION.values():
                base    = self.velocity_combination[load_condition]
                start_v = max( latest_velocity_log - CULL_THRESHOLD, base[:,1].min())
                end_v   = min( latest_velocity_log + CULL_THRESHOLD, base[:,1].max())
                target_combinations[load_condition] = base[np.where( (base[:,1] > start_v) & ( base[:,1] < end_v) )]
        return target_combinations
    
    def get_log_of_current_condition(self):
        return self.log[self.load_condition_to_human()]

    # check whether the vessel needs retrofit or not
    def check_retrofit(self):
        # load components list
        hull_list           = load_hull_list()
        engine_list         = load_engine_list()
        propeller_list      = load_propeller_list()        
        
        # return if the retrofit is not allowed
        if (self.retrofit_mode == RETROFIT_MODE['none']) or (self.retrofit_count == 0):
            return 

        # retrive days from current_day to X[WIP]
        simmulation_days = self.operation_date_array[np.where(self.operation_date_array > self.current_date)]

        # simmulation to check the vessel retrofits
        retrofit_designs = np.array([])
        # multi processing #
        target_combinations_array   = self.get_target_combinations()
        temp_npv = {}
        devided_target_combinations = np.array_split(target_combinations_array, PROC_NUM)
        for i in range(SIMMULATION_TIMES_FOR_RETROFITS):
            scenario = Sinario(self.sinario.history_data)
            scenario.generate_sinario(self.sinario_mode)
            current_combinations       = (self.hull, self.engine, self.propeller)
            start_time = time.clock()
            # initialize
            pool = mp.Pool(PROC_NUM)

            callback = [pool.apply_async(self.multi_simmulations, args=(index, devided_target_combinations, hull_list, engine_list, propeller_list, simmulation_days)) for index in xrange(PROC_NUM)]
            callback_combinations = [p.get() for p in callback]
            ret_combinations      = flatten_3d_to_2d(callback_combinations)
            pool.close()
            pool.join()
            # multi processing #

            potential_retrofit_designs = self.get_potential_retrofit_designs(ret_combinations)
            NPV, hull_id, engine_id, propeller_id = ret_combinations[np.argmax(ret_combinations['NPV'], axis=0)[0]][0]  
            combination_key = self.generate_combination_str_with_id(hull_id, engine_id, propeller_id)
            temp_npv[combination_key] = NPV
            lap_time = convert_second(time.clock() - start_time)
            print_with_notice("took %s for a scenario" % (lap_time))

        # average NPV for each design
        design_npv = {}
        for combination_key, npv_array in temp_npv.items():
            design_npv[combination_key] = np.average(npv_array)

        output_path = "%s/dock-in-log" % (self.output_dir_path)
        initializeDirHierarchy(output_path)
        output_filename = "%s/%s-%s.json" % (output_path, self.current_date, self.latest_dockin_date)
        write_file_as_json(design_npv, output_path)
        
        return 

    def get_target_combinations(self):
        dtype  = np.dtype({'names': ('hull_id', 'engine_id', 'propeller_id'),
                           'formats': (np.int , np.int, np.int)})
        
        target_design_array = np.array([], dtype=dtype)
        if self.retrofit_mode == RETROFIT_MODE['propeller']:
            # load components list
            hull_id        = self.hull.base_data['id']
            engine_id      = self.engine.base_data['id']
            propeller_list = load_propeller_list()
            for propeller_info in propeller_list:
                propeller_id = propeller_info['id']
                '''
                if propeller_id == self.propeller.base_data['id']:
                    continue
                '''
                add_design          = np.array([(hull_id,
                                                 engine_id,
                                                 propeller_id)],
                                               dtype=dtype)            
                target_design_array = append_for_np_array(target_design_array, add_design)
        elif self.retrofit_mode == RETROFIT_MODE['propeller_and_engine']:
            # load components list            
            engine_list         = load_engine_list()
            propeller_list      = load_propeller_list()
            pdb.set_trace()
            pass
            
        return target_design_array

    def multi_simmulations(self, index, devided_target_combinations, hull_list, engine_list, propeller_list, simmulation_days):
        dtype  = np.dtype({'names': ('hull_id', 'engine_id', 'propeller_id', 'NPV'),
                           'formats': (np.int , np.int, np.int, np.float)})
        combinations = np.array([], dtype=dtype)
        for target_combination in devided_target_combinations[index]:
            hull_id, engine_id, propeller_id = target_combination[0]
            hull      = Hull(hull_list, hull_id)
            engine    = Engine(engine_list, engine_id)
            propeller = Propeller(propeller_list, propeller_id)
            # create each arrays #
            rpm_array = np.arange(DEFAULT_RPM_RANGE['from'], engine.base_data['N_max'], RPM_RANGE_STRIDE)
            # conduct simmulation
            # prohibit retrofits here #
            retrofit_mode = RETROFIT_MODE['none']
            agent         = Agent(self.sinario, self.world_scale, retrofit_mode, self.sinario_mode, hull, engine, propeller, rpm_array)
            agent.operation_date_array = simmulation_days
            NPV           = agent.simmulate()
            # ignore aborted simmulation
            if NPV is None:
                continue
            
            # subtract retrofits cost
            if self.retrofit_mode == RETROFIT_MODE['propeller']:
                NPV -= RETROFIT_COST['propeller']
            elif self.retrofit_mode == RETROFIT_MODE['propeller_and_engine']:
                NPV -= RETROFIT_COST['propeller_and_engine']

            add_design = np.array([(hull_id,
                                    engine_id,
                                    propeller_id,
                                    NPV)],
                                  dtype=dtype)
            combinations = append_for_np_array(combinations, add_design)
            
        return combinations

    def get_potential_retrofit_designs(self, combinations):
        potential_retrofit_designs = np.array([], dtype=combinations.dtype)
        
        # get NPV of current design
        hull_id, engine_id, propeller_id = (self.hull.base_data['id'],
                                            self.engine.base_data['id'],
                                            self.propeller.base_data['id'])
        current_design_index, _dummy = np.where( (combinations['hull_id']==hull_id) &
                                                 (combinations['engine_id']==engine_id) &
                                                 (combinations['propeller_id']==propeller_id))
        current_design     = combinations[current_design_index]
        current_design_NPV = current_design['NPV'][0]
        potential_retrofit_design_induces = np.where(combinations['NPV'] > current_design_NPV)
        potential_retrofit_designs = combinations[potential_retrofit_design_induces]
        return potential_retrofit_designs
