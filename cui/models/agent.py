# import common modules #
import sys
import math
import copy
from pdb import *
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
from flat_rate   import FlatRate
from world_scale import WorldScale
# import models #

class Agent(object):
    def __init__(self, sinario, world_scale, flat_rate, retrofit_mode, sinario_mode, bf_mode, hull=None, engine=None, propeller=None, rpm_array=None, velocity_array=None):
        self.sinario       = sinario
        self.world_scale   = world_scale
        self.flat_rate     = flat_rate
        self.retrofit_mode = retrofit_mode
        self.sinario_mode  = sinario_mode
        self.bf_mode       = bf_mode
        self.icr           = DEFAULT_ICR_RATE
        self.operation_date_array = None


        if not hasattr(self, 'bf_info'):
            self.bf_info = load_bf_info()

        # initialize the range of velosity and rps
        # for velocity and rps array #
        self.velocity_array = np.arange(DEFAULT_VELOCITY_RANGE['from'], DEFAULT_VELOCITY_RANGE['to'], DEFAULT_VELOCITY_RANGE['stride']) if velocity_array is None else velocity_array
        self.rpm_array      = np.arange(DEFAULT_RPM_RANGE['from'], DEFAULT_RPM_RANGE['to'], DEFAULT_RPM_RANGE['stride']) if rpm_array is None else rpm_array
        # for velocity and rps array #

        # beaufort mode
        self.bf_prob          = self.load_bf_prob()
        self.change_sea_flag  = False

        # set route
        self.current_navigation_distance = NAVIGATION_DISTANCE_A

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
    # multi processing method #
    def get_initial_design_m(self, output_dir_path, initial_design_result_path=None):
        # load components list
        hull_list           = load_hull_list()
        engine_list         = load_engine_list()
        propeller_list      = load_propeller_list()

        simulation_duration_years = SIMMULATION_DURATION_YEARS_FOR_INITIAL_DESIGN
        simulate_count            = DEFAULT_SIMULATE_COUNT

        devided_component_ids = []
        for hull_info in hull_list:
            for engine_info in engine_list:
                for propeller_info in propeller_list:
                    devided_component_ids.append([hull_info['id'], engine_info['id'], propeller_info['id']])
        devided_component_ids = np.array_split(devided_component_ids, PROC_NUM)

        # initialize
        pool                      = mp.Pool(PROC_NUM)
        # multi processing #
        callback              = [pool.apply_async(self.calc_initial_design_m, args=(index, hull_list, engine_list, propeller_list, simulation_duration_years, simulate_count, devided_component_ids, output_dir_path)) for index in xrange(PROC_NUM)]

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
        aggregated_combi = aggregate_combinations(ret_combinations, output_dir_path)
        column_names     = ['hull_id',
                            'engine_id',
                            'propeller_id',
                            'averaged_NPV',
                            'std']
        write_array_to_csv(column_names, aggregated_combi, output_file_path)
        max_index, _dummy = np.where(aggregated_combi['averaged_NPV']==np.max(aggregated_combi['averaged_NPV']))
        hull_id, engine_id, propeller_id, averaged_NPV, std = aggregated_combi[max_index][0][0]
        hull                                                = Hull(hull_list, 1)
        engine                                              = Engine(engine_list, engine_id) 
        propeller                                           = Propeller(propeller_list, propeller_id)
        return averaged_NPV, hull, engine, propeller, std

    def simmulate(self, hull=None, engine=None, propeller=None, multi_flag=None, log_mode=None, retrofit_design_keys=None, validation_flg=None):
        # use instance variables if hull or engine or propeller are not given
        if (hull is None) or (engine is None) or (propeller is None):
            hull      = self.hull
            engine    = self.engine
            propeller = self.propeller

        # initialize retrofit_count
        self.retrofit_count_limit = 0 if self.retrofit_mode == RETROFIT_MODE['none'] else 1
        self.retrofit_count       = 0
        self.retrofit_design_keys = retrofit_design_keys

        # define velocity and rps for given [hull, engine, propeller]
        ## load combinations if combination file exists 
        self.velocity_combination = check_combinations_exists(hull, engine, propeller)
        if (not hasattr(self, 'velocity_combination')) or self.velocity_combination is None:
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
            self.operation_date_array  = generate_operation_date(self.sinario.predicted_data['date'][0])
        self.origin_date         = self.operation_date_array[0]
        self.retire_date         = self.operation_date_array[-1]
        self.round_trip_distance = self.current_navigation_distance * 2.0
        self.NPV                 = np.array([],np.dtype({'names': ('navigation_finished_date', 'NPV_in_navigation'),
                                                         'formats': ('S20' , np.float)}))        
        self.fuel_cost           = np.array([],np.dtype({'names': ('navigation_finished_date', 'fuel_cost_in_navigation'),
                                                           'formats': ('S20' , np.float)}))
        self.log                 = init_dict_from_keys_with_array(LOG_COLUMNS,
                                                                  np.dtype({'names': ('date', 'rpm', 'velocity'),
                                                                            'formats': ('S20', np.float , np.float)}))
        self.retrofit_date       = None
        self.base_design         = generate_combination_str(self.hull, self.engine, self.propeller)
        self.retrofit_design     = None
        self.elapsed_days_log    = np.array([])
        self.total_cash_flow     = 0
        self.total_NPV           = 0
        self.total_distance      = 0
        self.total_elapsed_days  = 0

        # dynamic variables
        self.current_date          = self.sinario.history_data['date'][-1]
        self.current_distance      = 0
        self.left_distance_to_port = self.current_navigation_distance
        self.voyage_date           = self.origin_date
        self.previous_oilprice     = self.sinario.history_data[-2]['price']
        self.oilprice_ballast      = self.sinario.history_data[-1]['price']
        self.oilprice_full         = self.sinario.history_data[-1]['price']

        # market factors
        self.current_oilprice    = self.oilprice_ballast
        self.current_flat_rate   = self.flat_rate.history_data[-1]['fr']
        self.current_world_scale = self.world_scale.history_data[-1]['ws']
        
        self.current_flat_rate  = self.flat_rate.history_data[-1]['fr']
        self.current_fare       = self.calc_fare()
        self.cash_flow          = 0
        self.loading_flag       = False
        self.loading_days       = 0
        self.elapsed_days       = 0
        self.latest_dockin_date = self.origin_date
        self.dockin_flag        = False
        self.dock_in_count      = 0
        self.ballast_trip_days  = 0
        self.return_trip_days   = 0

        # acutal sea fluctuation
        self.change_sea_count  = 1
        self.route_change_date = None

        # deteriorate
        self.d2d_det = {'v_knot': 0, 'rpm': 0, 'ehp': 0}
        self.age_eff = {'v_knot': 0, 'rpm': 0, 'ehp': 0}
        # initialize the temporal variables
        CF_day = rpm = v_knot = None
        raw_v  = raw_rpm = None
        for current_date in self.operation_date_array:
            # update current_date
            self.current_date = current_date

            # for loading duration
            if self.is_loading():
                # update total elapsed days
                self.total_elapsed_days += 1
                continue

            # for dock-in
            if self.is_dockin():
                # update total elapsed days
                self.total_elapsed_days += 1
                continue

            # define voyage_date
            if self.voyage_date is None:
                self.voyage_date = self.current_date

            # calculate optimized speed and rps during 
            if (CF_day is None) and (rpm is None) and (v_knot is None):
                if hasattr(self, 'constant_rpm'):
                    C_fuel, CF_day, rpm, v_knot = self.calc_constant_velocity(hull, engine, propeller)
                else:
                    C_fuel, CF_day, rpm, v_knot = self.calc_optimal_velocity(hull, engine, propeller)
            else:
                # conserve calm sea velocity
                v_knot = raw_v
                rpm    = raw_rpm
                
            # modification of the performance #
            raw_v   = v_knot
            raw_rpm = rpm            
            ## consider deterioration
            rpm, v_knot = self.modify_by_deterioration(rpm, v_knot)
            ## consider beaufort for velocity
            v_knot = self.modify_by_external(v_knot)
            ## update velocity log
            self.update_velocity_log(rpm, v_knot)

            '''
            ## consider real v_knot for fuel_cost and CF_day
            delta_distance  = knot2mileday(raw_v) - knot2mileday(v_knot)
            delta_fuel_cost = self.calc_fuel_cost_with_distance(delta_distance, rpm, v_knot, hull, engine, propeller)
            C_fuel -= delta_fuel_cost
            CF_day -= delta_fuel_cost
            '''
            
            # update variables
            ## update with the distance on a day
            navigated_distance = knot2mileday(v_knot)
            updated_distance   = self.current_distance + knot2mileday(v_knot)
            if (self.current_distance < self.current_navigation_distance) and (updated_distance >= self.current_navigation_distance):
                # ballast -> full
                # calc distance to the port
                navigated_distance = self.current_navigation_distance - self.current_distance                                
                # subtract unnavigated cash flow which depends on the distance
                discounted_distance = updated_distance - self.current_navigation_distance
                extra_fuel_cost = self.calc_fuel_cost_with_distance(discounted_distance, rpm, v_knot, hull, engine, propeller)
                CF_day -= extra_fuel_cost
                C_fuel -= extra_fuel_cost
                
                self.current_distance      = self.current_navigation_distance
                self.left_distance_to_port = self.current_navigation_distance
                # update oil price
                self.update_oilprice_and_fare()
                
                # loading flags
                self.initiate_loading()
                self.change_load_condition()

                # update cash flow
                self.cash_flow       += CF_day
                self.total_cash_flow += CF_day

                self.update_fuel_cost(C_fuel)

                # log elapsed days
                self.elapsed_days_log = np.append(self.elapsed_days_log, self.elapsed_days)
                
                # initialize the temporal variables
                C_fuel = CF_day = rpm = v_knot = None
        
            elif updated_distance >= self.round_trip_distance:
                # full -> ballast
                # calc distance to the port
                navigated_distance  = self.round_trip_distance - self.current_distance                
                # subtract unnavigated cash flow which depends on the distance
                discounted_distance = updated_distance - self.round_trip_distance
                extra_fuel_cost     = self.calc_fuel_cost_with_distance(discounted_distance, rpm, v_knot, hull, engine, propeller)
                CF_day -= extra_fuel_cost
                C_fuel -= extra_fuel_cost
                # loading flags (unloading)
                self.initiate_loading()
                self.change_load_condition()
                # update oil price'
                self.update_oilprice_and_fare()
                # reset current_distance
                self.current_distance = 0
                self.left_distance_to_port = self.current_navigation_distance

                self.update_fuel_cost(C_fuel)

                # calc Net Present Value and fuel_cost
                self.update_NPV_in_navigation()
                # initialize the vairables
                self.current_distance = 0
                self.cash_flow     = 0
                self.elapsed_days  = 0
                self.voyage_date   = None
                self.left_distance_to_port = self.current_navigation_distance

                # dock-in flag
                if self.update_dockin_flag():
                    # clear dock-to-dock deterioration
                    self.clear_d2d()
                    # initiate dock-in
                    self.initiate_dockin()
                    # consider age effect
                    self.update_age_effect()
                    # change route based on prob
                    self.change_route()
                    if self.retrofit_mode == RETROFIT_MODE['significant']:
                        if self.check_significant_retrofit():
                            self.conduct_retrofit()
                            self.retrofit_count_limit = 0
                            self.clear_age_effect()
                        else:
                            # consider dock-to-dock deterioration
                            self.update_d2d()
                    elif self.retrofit_mode == RETROFIT_MODE['significant_rule']:
                        retrofit_flag, mode = self.check_significant_rule_retrofit()
                        if retrofit_flag:
                            retorfit_design = self.retrofit_design_keys[mode]
                            self.conduct_retrofit(retorfit_design)
                            self.retrofit_count_limit = 0
                            self.clear_age_effect()
                        else:
                            # consider dock-to-dock deterioration
                            self.update_d2d()
                    elif self.retrofit_mode == RETROFIT_MODE['route_change']:
                        retrofit_flag, mode = self.check_route_change_retrofit()
                        if retrofit_flag:
                            retorfit_design = self.retrofit_design_keys[mode]
                            self.conduct_retrofit(retorfit_design)
                            self.retrofit_count_limit = 0
                            self.clear_age_effect()
                        else:
                            # consider dock-to-dock deterioration
                            self.update_d2d()
                    elif self.retrofit_mode == RETROFIT_MODE['route_change_merged']:
                        retrofit_flag, mode = self.check_route_change_retrofit_merged()
                        if retrofit_flag:
                            retorfit_design = self.retrofit_design_keys[mode]
                            self.conduct_retrofit(retorfit_design)
                            self.retrofit_count_limit = 0
                            self.clear_age_effect()
                        else:
                            # consider dock-to-dock deterioration
                            self.update_d2d()
                    else:
                        # consider dock-to-dock deterioration
                        self.update_d2d()                        
                    # for the same random seed
                    if hasattr(self, 'simulate_log_index'):
                        np.random.seed(calc_seed(self.simulate_log_index + self.dock_in_count))
                            
                # update cash flow
                self.cash_flow       += CF_day
                self.total_cash_flow += CF_day

                # initialize the temporal variables
                C_fuel = CF_day = rpm = v_knot = None
                self.ballast_trip_days = 0
                self.return_trip_days  = 0

                # validation flg
                if validation_flg:
                    # update whole NPV in vessel life time
                    whole_NPV       = round(self.calc_whole_NPV(), 3)
                    whole_fuel_cost = round(np.sum(self.fuel_cost['fuel_cost_in_navigation']), 3)
                    return whole_NPV, whole_fuel_cost
                continue
            else:
                # full -> full or ballast -> ballast
                self.current_distance      += navigated_distance
                self.left_distance_to_port -= navigated_distance
                # update cash flow
                self.cash_flow       += CF_day
                self.total_cash_flow += CF_day                
                self.update_fuel_cost(C_fuel)
            
            # update total distance
            self.total_distance += navigated_distance
            
            # update days
            self.elapsed_days       += 1
            self.total_elapsed_days += 1            
            self.update_trip_days()

            # display current infomation
            # self.dispaly_current_infomation(navigated_distance, rpm, v_knot, CF_day, C_fuel)

            if log_mode:
                self.update_CF_log(CF_day)

        # update whole NPV in vessel life time
        whole_NPV       = round(self.calc_whole_NPV(), 3)
        whole_fuel_cost = round(np.sum(self.fuel_cost['fuel_cost_in_navigation']), 3)
        return whole_NPV, whole_fuel_cost

    def dispaly_current_infomation(self, navigated_distance, rpm, v_knot, CF_day, C_fuel):
        print "--------------Finished Date: %s--------------" % (self.current_date)
        print "%25s: %10d"            % ('Hull ID'              , self.hull.base_data['id'])
        print "%25s: %10d"            % ('Engine ID'            , self.engine.base_data['id'])
        print "%25s: %10d"            % ('Propeller ID'         , self.propeller.base_data['id'])
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
        print "%25s: %10s [$/day]"    % ('Fuel Cost'            , number_with_delimiter(C_fuel) if not C_fuel is None else '----')
        print "%25s: %10s [$]"        % ('Total Cash flow'      , number_with_delimiter(self.total_cash_flow))
        print "%25s: %10s"            % ('Retrofit Mode'        , self.retrofit_mode_to_human())
        print "--------------Retire Date: %s--------------" % (self.retire_date)
        return
    
    # define velocity and rps for given [hull, engine, propeller]
    def create_velocity_combination(self, hull, engine, propeller):
        ret_combinations = {}
        for load_condition in LOAD_CONDITION.keys():
            combinations = []
            for rpm in engine.rpm_array:
                tmp_combinations = []
                for velocity in self.velocity_array:
                    velocity    = round(velocity, 4)
                    velocity_ms = knot2ms(velocity)
                    # calc error of fitness bhp values
                    error = self.rpm_velocity_fitness(hull, engine, propeller, velocity_ms, rpm, load_condition)
                    if error is not None:
                        tmp_combinations.append([rpm, velocity, error])
                # for no combinations case
                if len(tmp_combinations) > 0:
                    min_combination = tmp_combinations[np.argmin(map(lambda x: x[2], tmp_combinations))]
                    add_velocity = hull.consider_bow_for_v(min_combination[1], load_condition)
                    combinations.append([min_combination[0],add_velocity])
            ret_combinations[load_condition_to_human(load_condition)] = combinations

        dir_name        = "%s/designs" % (COMBINATIONS_DIR_PATH)
        initializeDirHierarchy(dir_name)
        # draw RPS-velocity combinations
        self.draw_combinations(hull, engine, propeller, ret_combinations, dir_name)
        # draw BHP-rpm
        self.draw_BHP_rpm_graph(hull, engine, propeller, ret_combinations, dir_name)
        # draw EHP-knot
        self.draw_EHP_Knot_graph(hull, engine, propeller, ret_combinations, dir_name)
        # output json as file
        self.write_combinations_as_json(hull, engine, propeller, ret_combinations, dir_name)

        return ret_combinations

    def rpm_velocity_fitness(self, hull, engine, propeller, velocity_ms, rpm, load_condition):
        # calc bhp [WW]
        fitness_bhp0 = self.get_modified_bhp(rpm, engine)
        fitness_bhp1 = self.calc_bhp_with_ehp(velocity_ms, rpm, hull, engine, propeller, load_condition)
        # reject bhp over the engine Max load
        if fitness_bhp0 is None or fitness_bhp1 is None or fitness_bhp0 > engine.base_data['max_load'] or fitness_bhp1 > engine.base_data['max_load']:
            return None

        error = math.pow(fitness_bhp0 - fitness_bhp1, 2)
        error = math.sqrt(error)
        return error

    # return modified bhp[kW] by efficiency
    def get_modified_bhp(self, rpm, engine):
        return engine.calc_bhp(rpm)

    # return bhp [kW]    
    def calc_bhp_with_ehp(self, velocity_ms, rpm, hull, engine, propeller, load_condition):
        rps = rpm2rps(rpm)
        # reject if the condition (KT > 0 and eta > 0) fulfilled
        # advance constants
        J   = propeller.calc_advance_constant(velocity_ms, rps)
        KT  = propeller.calc_KT(J)
        KQ  = propeller.calc_KQ(J)
        eta = propeller.calc_eta(rps, velocity_ms, KT, KQ)
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
        denominator = THRUST_COEFFICIENT * (velocity_ms / (2 * math.pi) ) * (1 / (rps * propeller.base_data['D']) ) * ( KT / KQ ) * ETA_S
        bhp = numerator / denominator
        # return bhp [kW]
        
        return engine.consider_efficiency(rpm, bhp)

    def calc_optimal_velocity(self, hull, engine, propeller):
        combinations       = []
        # cull the combination for the fast cunduct #
        target_combination = self.cull_combination()
        for rpm_first, velocity_first in target_combination[self.load_condition_to_human()]:
            # ignore second parameter when the navigation is return trip
            if self.is_ballast():
                ## when the ship is ballast
                tmp_combinations = []
                # decide velocity of full                
                for rpm_second, velocity_second in target_combination[self.get_another_load_condition_to_human()]:
                    ND                        = self.calc_ND(velocity_first, velocity_second)
                    tmp_cash_flow, tmp_C_fuel = self.calc_cash_flow(rpm_first, velocity_first, rpm_second, velocity_second, hull, engine, propeller, ND)
                    tmp_combinations.append((tmp_C_fuel, tmp_cash_flow, rpm_second, velocity_second))
                C_fuel, cash_flow, optimal_rpm_full, optimal_velocity_full = sorted(tmp_combinations, key=lambda x : x[1], reverse=True)[0]
            else:
                ## when the ship is full (return trip)
                ND                = self.calc_ND(velocity_first, 0)
                cash_flow, C_fuel = self.calc_cash_flow(rpm_first, velocity_first, 0, 0, hull, engine, propeller, ND)
            combinations.append((C_fuel, cash_flow, rpm_first, velocity_first))
        # decide the velocity
        C_fuel, CF_day, optimal_rpm, optimal_velocity = sorted(combinations, key=lambda x : x[1], reverse=True)[0]
        return C_fuel, CF_day, optimal_rpm, optimal_velocity

    def calc_constant_velocity(self, hull, engine, propeller):
        combinations       = []
        # cull the combination for the fast cunduct #
        target_combination = self.cull_combination()
        constant_rpm       = self.constant_rpm
        
        if self.is_ballast():
            combinations    = target_combination[self.load_condition_to_human()]
            velocity_first  = [_d[1] for _d in combinations if _d[0] == constant_rpm][0]
            combinations    = target_combination[self.get_another_load_condition_to_human()]
            velocity_second = [_d[1] for _d in combinations if _d[0] == constant_rpm][0]
            ND              = self.calc_ND(velocity_first, velocity_second)
            CF_day, C_fuel  = self.calc_cash_flow(constant_rpm, velocity_first, constant_rpm, velocity_second, hull, engine, propeller, ND)
        else:
            combinations    = target_combination[self.get_another_load_condition_to_human()]
            velocity_first  = [_d[1] for _d in combinations if _d[0] == constant_rpm][0]
            ND              = self.calc_ND(velocity_first, 0)
            CF_day, C_fuel  = self.calc_cash_flow(constant_rpm, velocity_first, 0, 0, hull, engine, propeller, ND)
        return C_fuel, CF_day, constant_rpm, velocity_first
    
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
        C_fuel, CF_day, optimal_rpm, optimal_velocity = ret_combinations[np.argmax(ret_combinations, axis=0)[0]]
        return C_fuel, CF_day, optimal_rpm, optimal_velocity    

    # return ND [days]
    # ND is whole number of days in navigation
    def calc_ND(self, velocity_first, velocity_second):
        # ignore second clause when 'full'
        if self.is_full():
            first_clause  = self.calc_left_distance() / knot2mileday(velocity_first)
            second_clause = 0
        else:
            first_clause  = (self.calc_left_distance() - self.current_navigation_distance) / knot2mileday(velocity_first)
            second_clause = self.current_navigation_distance / knot2mileday(velocity_second)

        ret_ND = self.elapsed_days + first_clause + second_clause
        return ret_ND
            

    def calc_CF_in_navigation(self):
        ND = self.elapsed_days

        # income
        I = (1 - self.icr) * self.current_fare * self.hull.base_data['DWT']

        # Cost for fix_day
        C_fix  = self.calc_fix_cost() * ND

        # Cost for port_day
        C_port = self.calc_port_cost(ND) * ND

        # Fuel Cost
        voyage_start_index = np.where(self.fuel_cost['navigation_finished_date'] == datetime_to_human(self.voyage_date))[0]
        voyage_end_index   = np.where(self.fuel_cost['navigation_finished_date'] == datetime_to_human(self.current_date))[0]
        C_fuel = np.sum(self.fuel_cost[voyage_start_index:voyage_end_index+1]['fuel_cost_in_navigation'])

        # total CF
        CF_in_navigation = I - C_fix - C_port - C_fuel
        return CF_in_navigation

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

        return cash_flow, C_fuel

    # calc fuel cost per day    
    def calc_fuel_cost(self, hull, engine, propeller, ND, rpm_first, velocity_first, rpm_second, velocity_second):
        ret_fuel_cost = 0
        if self.is_ballast():
            # ballast
            bhp_ballast       = self.get_modified_bhp(rpm_first, engine)
            sfoc_ballast      = engine.calc_sfoc(bhp_ballast)
            bhp_full          = self.get_modified_bhp(rpm_second, engine)
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
                averaged_bhp  = self.get_modified_bhp(averaged_rpm, engine)
                averaged_sfoc = engine.calc_sfoc(averaged_bhp)
                averaged_fuel_cost = (1000 * self.oilprice_ballast) / 159.0 * averaged_bhp * averaged_sfoc * (24.0 / 1000000.0)
                first_clause = averaged_fuel_cost * self.ballast_trip_days

            fuel_cost  = first_clause
            # go navigation
            fuel_cost += fuel_cost_ballast * ( (self.calc_left_distance() - self.current_navigation_distance) / knot2mileday(velocity_first) )
            # return navigation
            fuel_cost += fuel_cost_full * ( self.current_navigation_distance / knot2mileday(velocity_second) )
            # consider navigated 
            fuel_cost /= float(ND)
        else:
            # ballast
            # use averaged rps for navigated distance
            another_condition          = self.get_another_load_condition_to_human()
            averaged_v_ballast         = np.average(self.log[another_condition]['velocity'])
            averaged_rpm_ballast       = np.average(self.log[another_condition]['rpm'])
            averaged_bhp_ballast       = self.get_modified_bhp(averaged_rpm_ballast, engine)
            averaged_sfoc_ballast      = engine.calc_sfoc(averaged_bhp_ballast)
            averaged_fuel_cost_ballast = (1000 * self.oilprice_ballast) / 159.0 * averaged_bhp_ballast * averaged_sfoc_ballast * (24.0 / 1000000.0)
            fuel_cost_ballast = averaged_fuel_cost_ballast * self.return_trip_days
            
            # full
            bhp_full          = self.get_modified_bhp(rpm_first, engine)
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
                averaged_bhp  = self.get_modified_bhp(averaged_rpm, engine)
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
        add_array = np.array((datetime_to_human(self.current_date), rpm, v_knot), dtype=self.log['full'].dtype)
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
        bhp           = self.get_modified_bhp(rpm, engine)
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

    def update_oilprice_and_fare(self):
        # get current_date
        if not ( isinstance(self.current_date, datetime.date) or isinstance(self.current_date, datetime.datetime)):
            current_date           = str_to_datetime(self.current_date)
        else:
            current_date = self.current_date
        
        # update oilprice
        current_date_index        = search_near_index(self.current_date, self.sinario.predicted_data['date'])
        self.previous_oilprice    = self.get_previous_oilprice(current_date_index)
        current_index,            = np.where(self.sinario.predicted_data['date']==current_date_index)
        if self.is_ballast():
            self.oilprice_ballast = self.sinario.predicted_data[current_index[0]]['price']
            self.current_oilprice = self.oilprice_ballast
        else:
            self.oilprice_full    = self.sinario.predicted_data[current_index[0]]['price']
            self.current_oilprice = self.oilprice_full            

        # update world_scale
        world_scale_index      = search_near_index(current_date, self.world_scale.predicted_data['date'])
        designated_world_scale = self.world_scale.predicted_data[np.where(self.world_scale.predicted_data['date']==world_scale_index)]        
        self.current_world_scale = designated_world_scale['ws'][0]
        # update flat_rate
        if not ( isinstance(self.current_date, datetime.date) or isinstance(self.current_date, datetime.datetime)):
            current_date           = str_to_datetime(self.current_date)
        else:
            current_date = self.current_date            
        flat_rate_index        = search_near_index(current_date, self.flat_rate.predicted_data['date'])
        designated_flat_rate   = self.flat_rate.predicted_data[np.where(self.flat_rate.predicted_data['date']==flat_rate_index)]
        self.current_flat_rate = designated_flat_rate['fr'][0]        
            
        # update fare
        self.current_fare = self.calc_fare()
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
        raise
        return    

    # PV = CF_in_navigation / (1+Discount_rate)^elapsed_month
    def update_NPV_in_navigation(self):
        denominator       = math.pow( (1 + DISCOUNT_RATE), self.calc_elapsed_month())
        CF_in_navigation  = self.calc_CF_in_navigation()
        NPV_in_navigation = CF_in_navigation / denominator
        self.update_NPV_log(NPV_in_navigation)
        return

    def update_fuel_cost(self, fuel_cost):
        dtype     = np.dtype({'names': ('navigation_finished_date', 'fuel_cost_in_navigation'),'formats': ('S20' , np.float)})
        add_array = np.array([(datetime_to_human(self.current_date), fuel_cost)], dtype=dtype)
        self.fuel_cost = append_for_np_array(self.fuel_cost, add_array)
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
        days_delta_from_voyage = (self.current_date - self.origin_date).days
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
        
    def calc_whole_NPV(self):
        self.total_NPV = np.sum(self.NPV['NPV_in_navigation'])
        return self.total_NPV

    def update_dockin_flag(self):
        next_dockin_date   = add_year(self.latest_dockin_date, DOCK_IN_PERIOD)
        return self.current_date >= next_dockin_date

    def initiate_dockin(self):
        import copy
        left_dock_date          = add_month(copy.deepcopy(self.current_date), DOCK_IN_DURATION)
        self.latest_dockin_date = left_dock_date
        self.dock_in_count      += 1        
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
                    ND     = self.calc_ND(velocity_first, velocity_second)
                    cash_flow, C_fuel = self.calc_cash_flow(rpm_first, velocity_first, rpm_second, velocity_second, hull, engine, propeller, ND)
                    tmp_combinations = append_for_np_array(tmp_combinations, [C_fuel, cash_flow, rpm_second, velocity_second])
                C_fuel, CF_day, optimal_rpm_full, optimal_velocity_full = tmp_combinations[np.argmax(tmp_combinations, axis=0)[0]]
            else:
                ## when the ship is full (return trip)
                ND     = self.calc_ND(velocity_first, 0)
                cash_flow, C_fuel = self.calc_cash_flow(rpm_first, velocity_first, 0, 0, hull, engine, propeller, ND)
            combinations = append_for_np_array(combinations, [C_fuel, cash_flow, rpm_first, velocity_first])
        return combinations
    
    def calc_significant_design_m(self, index, hull_list, engine_list, propeller_list, simulation_duration_years, devided_component_ids, result_path):
        column_names    = ['hull_id',
                           'engine_id',
                           'propeller_id',
                           'NPV',
                           'fuel_cost', 'avg_round_num', 'round_num']
        dtype  = np.dtype({'names': ('hull_id', 'engine_id', 'propeller_id', 'NPV', 'fuel_cost', 'avg_round_num', 'round_num'),
                           'formats': (np.int, np.int , np.int, np.float, np.float, np.float, np.float)})
        design_array = np.array([], dtype=dtype)

        result_data  = load_result(result_path)

        start_time   = time.clock()
        # conduct multiple simmulation for each design
        result_array = {}
        for component_ids in devided_component_ids[index]:
            # fix random seed
            np.random.seed(COMMON_SEED_NUM)
            hull, engine, propeller = get_component_from_id_array(component_ids, hull_list, engine_list, propeller_list)
            # get existing result file
            combination_str  = generate_combination_str(hull, engine, propeller)
            # conduct simulation #
            agent                      = Agent(self.sinario, self.world_scale, self.flat_rate, self.retrofit_mode, self.sinario_mode, self.bf_mode, hull, engine, propeller)
            end_date                   = add_year(str_to_date(self.sinario.predicted_data['date'][0]), simulation_duration_years)
            agent.operation_date_array = generate_operation_date(self.sinario.predicted_data['date'][0], end_date)
            NPV, fuel_cost             = agent.simmulate()
            # draw velocity log
            agent.draw_velocity_log(result_path)            
            # write npv and fuel_cost file
            output_dir_path = "%s/%s" % (result_path, generate_combination_str(hull, engine, propeller))
            initializeDirHierarchy(output_dir_path)
            write_array_to_csv(agent.NPV.dtype.names, agent.NPV, "%s/npv.csv" % (output_dir_path))
            write_array_to_csv(agent.fuel_cost.dtype.names, agent.fuel_cost, "%s/fuel_cost.csv" % (output_dir_path))
     
            # conduct simulation #
            # ignore aborted simmulation
            if NPV is None:
                continue
            # write simmulation result
            output_file_path = "%s/%s_core%d.csv" % (result_path, 'initial_design', index)
            lap_time         = convert_second(time.clock() - start_time)
            write_csv(column_names, [hull.base_data['id'],
                                     engine.base_data['id'],
                                     propeller.base_data['id'],
                                     NPV, fuel_cost, np.average(agent.elapsed_days_log), len(agent.elapsed_days_log), lap_time], output_file_path)
            add_design   = np.array([(hull.base_data['id'],
                                      engine.base_data['id'],
                                      propeller.base_data['id'],
                                      NPV, fuel_cost, np.average(agent.elapsed_days_log), len(agent.elapsed_days_log))],
                                     dtype=dtype)
            design_array = append_for_np_array(design_array, add_design)                    
        return design_array

    def calc_design_m(self, index, common_seed_num, hull_list, engine_list, propeller_list, simulation_duration_years, devided_simulation_times, combination_str, result_path):
        column_names    = ['hull_id',
                           'engine_id',
                           'propeller_id',
                           'NPV',
                           'fuel_cost']
        dtype  = np.dtype({'names': ('hull_id', 'engine_id', 'propeller_id', 'NPV', 'fuel_cost'),
                           'formats': (np.int, np.int , np.int, np.float, np.float)})
        design_array = np.array([], dtype=dtype)        
        simulation_times = devided_simulation_times[index]
        for simulation_time in simulation_times:
            start_time                 = time.clock()
            seed_num                   = common_seed_num * simulation_time
            np.random.seed(seed_num)
            generate_market_scenarios(self.sinario, self.world_scale, self.flat_rate, self.sinario_mode, simulation_duration_years)
            component_ids              = get_component_ids_from_design_key(combination_str)
            hull, engine, propeller    = get_component_from_id_array(map(int, component_ids), hull_list, engine_list, propeller_list)
            # conduct simulation #
            agent                      = Agent(self.sinario, self.world_scale, self.flat_rate, self.retrofit_mode, self.sinario_mode, self.bf_mode, hull, engine, propeller)
            end_date                   = add_year(str_to_date(self.sinario.predicted_data['date'][0]), simulation_duration_years)
            agent.simulate_log_index   = simulation_time
            agent.operation_date_array = generate_operation_date(self.sinario.predicted_data['date'][0], end_date)
            NPV, fuel_cost             = agent.simmulate()
            # write npv and fuel_cost file
            output_dir_path            = "%s/%s/simulate%d" % (result_path, combination_str, agent.simulate_log_index)
            initializeDirHierarchy(output_dir_path)
            write_array_to_csv(agent.NPV.dtype.names, agent.NPV, "%s/npv.csv" % (output_dir_path))
            write_array_to_csv(agent.fuel_cost.dtype.names, agent.fuel_cost, "%s/fuel_cost.csv" % (output_dir_path))
            # conduct simulation #
            # ignore aborted simmulation
            if NPV is None:
                continue
            # write simmulation result
            output_file_path = "%s/%s_core%d.csv" % (result_path, 'initial_design', index)
            lap_time         = convert_second(time.clock() - start_time)
            write_csv(column_names, [hull.base_data['id'],
                                     engine.base_data['id'],
                                     propeller.base_data['id'],
                                     NPV, fuel_cost, lap_time], output_file_path)
            add_design   = np.array([(hull.base_data['id'],
                                      engine.base_data['id'],
                                      propeller.base_data['id'],
                                      NPV, fuel_cost)],
                                    dtype=dtype)
            design_array = append_for_np_array(design_array, add_design)                    
        return design_array        

    def calc_integrated_design_m(self, index, common_seed_num, hull_list, engine_list, propeller_list, simulation_duration_years, devided_simulation_times, base_design_key, retrofit_design_key, output_integrated_dir_path):
        column_names     = ['simulation_time',
                            'hull_id',
                            'engine_id',
                            'propeller_id',
                            'NPV',
                            'fuel_cost']
        dtype            = np.dtype({'names': ('simulation_time', 'hull_id', 'engine_id', 'propeller_id', 'NPV', 'fuel_cost'),
                           'formats': (np.int, np.int, np.int , np.int, np.float, np.float)})
        design_array     = np.array([], dtype=dtype)        
        simulation_times = devided_simulation_times[index]

        for simulation_time in simulation_times:
            start_time                 = time.clock()
            seed_num                   = common_seed_num * simulation_time
            np.random.seed(seed_num)
            component_ids              = get_component_ids_from_design_key(base_design_key)
            hull, engine, propeller    = get_component_from_id_array(map(int, component_ids), hull_list, engine_list, propeller_list)
            # conduct simulation #
            agent                      = Agent(self.sinario, self.world_scale, self.flat_rate, self.retrofit_mode, self.sinario_mode, self.bf_mode, hull, engine, propeller)
            agent.output_dir_path      = self.output_dir_path
            agent.simulate_log_index   = simulation_time
            generate_market_scenarios(agent.sinario, agent.world_scale, agent.flat_rate, self.sinario_mode, simulation_duration_years)
            end_date                   = add_year(str_to_date(self.sinario.predicted_data['date'][0]), simulation_duration_years)
            agent.operation_date_array = generate_operation_date(self.sinario.predicted_data['date'][0], end_date)
            NPV, fuel_cost             = agent.simmulate(hull, engine, propeller, None, None, retrofit_design_key)
            # write npv and fuel_cost file
            output_dir_path            = "%s/%s/simulate%d" % (output_integrated_dir_path, base_design_key, agent.simulate_log_index)
            initializeDirHierarchy(output_dir_path)
            write_array_to_csv(agent.NPV.dtype.names, agent.NPV, "%s/npv.csv" % (output_dir_path))
            write_array_to_csv(agent.fuel_cost.dtype.names, agent.fuel_cost, "%s/fuel_cost.csv" % (output_dir_path))
            # conduct simulation #
            # ignore aborted simmulation
            if NPV is None:
                continue
            # write simmulation result
            output_file_path = "%s/%s_core%d.csv" % (output_integrated_dir_path, 'initial_design', index)
            lap_time         = convert_second(time.clock() - start_time)
            write_csv(column_names, [simulation_time,
                                     hull.base_data['id'],
                                     engine.base_data['id'],
                                     propeller.base_data['id'],
                                     NPV, fuel_cost, lap_time], output_file_path)
            add_design   = np.array([(simulation_time,
                                      hull.base_data['id'],
                                      engine.base_data['id'],
                                      propeller.base_data['id'],
                                      NPV, fuel_cost)],
                                    dtype=dtype)
            design_array = append_for_np_array(design_array, add_design)                    
        return design_array

    def calc_flexible_design_m(self, index, hull_list, engine_list, propeller_list, simulation_duration_years, devided_simulation_times, base_design_key, retrofit_design_keys, retrofit_mode):
        column_names     = ['simulation_time',
                            'hull_id',
                            'engine_id',
                            'propeller_id',
                            'NPV',
                            'fuel_cost',
                            'base_design',
                            'retrofit_design',
                            'retrofit_date',
                            'change_route_date']
        dtype            = np.dtype({'names': ('simulation_time', 'hull_id', 'engine_id', 'propeller_id', 'NPV', 'fuel_cost'),
                           'formats': (np.int, np.int, np.int , np.int, np.float, np.float)})
        design_array     = np.array([], dtype=dtype)        
        simulation_times = devided_simulation_times[index]
        for simulation_time in simulation_times:
            # common process
            component_ids                          = get_component_ids_from_design_key(base_design_key)
            self.hull, self.engine, self.propeller = get_component_from_id_array(map(int, component_ids), hull_list, engine_list, propeller_list)
            self.simulate_log_index                = simulation_time

            # for route fluc
            if hasattr(self, 'change_route_prob'):
                np.random.seed(calc_seed(simulation_time))
                self.set_change_route_period()

            # for no retrofit design
            start_time                 = time.clock()
            np.random.seed(calc_seed(simulation_time))
            generate_market_scenarios(self.sinario, self.world_scale, self.flat_rate, self.sinario_mode, simulation_duration_years)
            end_date                   = add_year(str_to_date(self.sinario.predicted_data['date'][0]), simulation_duration_years)
            self.operation_date_array  = generate_operation_date(self.sinario.predicted_data['date'][0], end_date)            
            self.retrofit_mode         = RETROFIT_MODE['none']
            self.set_route_distance('A')
            self.bf_prob               = self.load_bf_prob()
            NPV, fuel_cost             = self.simmulate(None, None, None, None, None, None)                        
            ## write npv and fuel_cost file
            output_dir_path            = "%s/no_retrofit/%s/simulate%d" % (self.output_dir_path, base_design_key, self.simulate_log_index)
            initializeDirHierarchy(output_dir_path)
            write_array_to_csv(self.NPV.dtype.names, self.NPV, "%s/npv.csv" % (output_dir_path))
            write_array_to_csv(self.fuel_cost.dtype.names, self.fuel_cost, "%s/fuel_cost.csv" % (output_dir_path))
            # write simmulation result
            output_dir_path  = "%s/no_retrofit" % (self.output_dir_path)
            output_file_path = "%s/%s_core%d.csv" % (output_dir_path, 'initial_design', index)
            lap_time         = convert_second(time.clock() - start_time)
            write_csv(column_names, [simulation_time,
                                     self.hull.base_data['id'],
                                     self.engine.base_data['id'],
                                     self.propeller.base_data['id'],
                                     NPV, fuel_cost, self.base_design, '--', '--', self.route_change_date, lap_time], output_file_path)            
            
            # for flexible design
            start_time                 = time.clock()
            np.random.seed(calc_seed(simulation_time))
            generate_market_scenarios(self.sinario, self.world_scale, self.flat_rate, self.sinario_mode, simulation_duration_years)
            self.retrofit_mode         = retrofit_mode
            self.set_route_distance('A')
            self.bf_prob               = self.load_bf_prob()
            NPV, fuel_cost             = self.simmulate(None, None, None, None, None, retrofit_design_keys)        

            ## write npv and fuel_cost file
            output_dir_path            = "%s/flexible/%s/simulate%d" % (self.output_dir_path, base_design_key, self.simulate_log_index)
            initializeDirHierarchy(output_dir_path)
            write_array_to_csv(self.NPV.dtype.names, self.NPV, "%s/npv.csv" % (output_dir_path))
            write_array_to_csv(self.fuel_cost.dtype.names, self.fuel_cost, "%s/fuel_cost.csv" % (output_dir_path))
            # write simmulation result
            output_dir_path  = "%s/flexible" % (self.output_dir_path)
            output_file_path = "%s/%s_core%d.csv" % (output_dir_path, 'initial_design', index)
            lap_time         = convert_second(time.clock() - start_time)
            retrofit_design  = self.retrofit_design_human
            write_csv(column_names, [simulation_time,
                                     self.hull.base_data['id'],
                                     self.engine.base_data['id'],
                                     self.propeller.base_data['id'],
                                     NPV, fuel_cost, self.base_design, self.retrofit_design_human(), self.retrofit_date_human(), self.route_change_date, lap_time], output_file_path)            
        return 

    def set_change_route_period(self):
        self.change_route_period = prob_with_weight(self.change_route_prob)
        return 

    def reset_change_route_period(self):
        self.change_route_period = None
        return 
    
    def calc_flexible_design_m_route_change(self, index, hull_list, engine_list, propeller_list, simulation_duration_years, devided_periods, base_design_key, retrofit_design_key, retrofit_mode):
        column_names     = ['hull_id',
                            'engine_id',
                            'propeller_id',
                            'NPV',
                            'fuel_cost',
                            'base_design',
                            'retrofit_design',
                            'retrofit_date',
                            'change_route_date']
        dtype            = np.dtype({'names': ('simulation_time', 'hull_id', 'engine_id', 'propeller_id', 'NPV', 'fuel_cost'),
                           'formats': (np.int, np.int, np.int , np.int, np.float, np.float)})
        simulation_periods = devided_periods[index]
        for change_route_period in simulation_periods:
            # common process
            component_ids                          = get_component_ids_from_design_key(base_design_key)
            self.hull, self.engine, self.propeller = get_component_from_id_array(map(int, component_ids), hull_list, engine_list, propeller_list)
            self.change_route_period = change_route_period
            self.output_dir_path     = "%s/period_%d" % (self.output_dir_path, change_route_period)
            initializeDirHierarchy(self.output_dir_path)
            self.simulate_log_index  = COMMON_SEED_NUM
            np.random.seed(calc_seed(COMMON_SEED_NUM))

            # for no retrofit design
            start_time                 = time.clock()
            end_date                   = add_year(str_to_date(self.sinario.predicted_data['date'][0]), simulation_duration_years)
            self.operation_date_array  = generate_operation_date(self.sinario.predicted_data['date'][0], end_date)            
            self.retrofit_mode         = RETROFIT_MODE['none']
            self.set_route_distance('A')
            self.bf_prob               = self.load_bf_prob()
            np.random.seed(calc_seed(COMMON_SEED_NUM))
            NPV, fuel_cost             = self.simmulate(None, None, None, None, None, None)                        
            ## write npv and fuel_cost file
            output_dir_path            = "%s/no_retrofit/%s" % (self.output_dir_path, base_design_key)
            initializeDirHierarchy(output_dir_path)
            write_array_to_csv(self.NPV.dtype.names, self.NPV, "%s/npv.csv" % (output_dir_path))
            write_array_to_csv(self.fuel_cost.dtype.names, self.fuel_cost, "%s/fuel_cost.csv" % (output_dir_path))
            # write simmulation result
            output_dir_path  = "%s/no_retrofit" % (self.output_dir_path)
            output_file_path = "%s/%s_core%d.csv" % (output_dir_path, 'initial_design', index)
            lap_time         = convert_second(time.clock() - start_time)
            write_csv(column_names, [self.hull.base_data['id'],
                                     self.engine.base_data['id'],
                                     self.propeller.base_data['id'],
                                     NPV, fuel_cost, self.base_design, '--', '--', self.route_change_date, lap_time], output_file_path)            
            # for flexible design
            start_time                 = time.clock()
            np.random.seed(calc_seed(COMMON_SEED_NUM))
            self.retrofit_mode         = retrofit_mode
            self.set_route_distance('A')
            self.bf_prob               = self.load_bf_prob()
            np.random.seed(calc_seed(COMMON_SEED_NUM))
            NPV, fuel_cost             = self.simmulate(None, None, None, None, None, retrofit_design_key)        
            ## write npv and fuel_cost file
            output_dir_path            = "%s/flexible/%s" % (self.output_dir_path, base_design_key)
            initializeDirHierarchy(output_dir_path)
            write_array_to_csv(self.NPV.dtype.names, self.NPV, "%s/npv.csv" % (output_dir_path))
            write_array_to_csv(self.fuel_cost.dtype.names, self.fuel_cost, "%s/fuel_cost.csv" % (output_dir_path))
            # write simmulation result
            output_dir_path  = "%s/flexible" % (self.output_dir_path)
            output_file_path = "%s/%s_core%d.csv" % (output_dir_path, 'initial_design', index)
            lap_time         = convert_second(time.clock() - start_time)
            retrofit_design  = self.retrofit_design_human
            write_csv(column_names, [self.hull.base_data['id'],
                                     self.engine.base_data['id'],
                                     self.propeller.base_data['id'],
                                     NPV, fuel_cost, self.base_design, self.retrofit_design_human(), self.retrofit_date_human(), self.route_change_date, lap_time], output_file_path)            
        return

    def calc_flexible_design_m_route_change_monte(self, index, hull_list, engine_list, propeller_list, simulation_duration_years, devided_simulation_times, base_design_key, retrofit_design_key, retrofit_mode):
        column_names     = ['simulation_time',
                            'hull_id',
                            'engine_id',
                            'propeller_id',
                            'NPV',
                            'fuel_cost',
                            'base_design',
                            'retrofit_design',
                            'retrofit_date',
                            'change_route_date']
        dtype            = np.dtype({'names': ('simulation_time', 'hull_id', 'engine_id', 'propeller_id', 'NPV', 'fuel_cost'),
                           'formats': (np.int, np.int, np.int , np.int, np.float, np.float)})
        simulation_times = devided_simulation_times[index]
        for simulation_time in simulation_times:
            # common process
            component_ids                          = get_component_ids_from_design_key(base_design_key)
            self.hull, self.engine, self.propeller = get_component_from_id_array(map(int, component_ids), hull_list, engine_list, propeller_list)
            self.simulate_log_index  = simulation_time

            # for route fluc
            if hasattr(self, 'change_route_prob'):
                np.random.seed(calc_seed(simulation_time))
                self.set_change_route_period()

            # for no retrofit design
            start_time                 = time.clock()
            np.random.seed(calc_seed(simulation_time))
            end_date                   = add_year(str_to_date(self.sinario.predicted_data['date'][0]), simulation_duration_years)
            self.operation_date_array  = generate_operation_date(self.sinario.predicted_data['date'][0], end_date)            
            self.retrofit_mode         = RETROFIT_MODE['none']
            self.set_route_distance('A')
            self.bf_prob               = self.load_bf_prob()
            np.random.seed(calc_seed(COMMON_SEED_NUM))
            NPV, fuel_cost             = self.simmulate(None, None, None, None, None, None)                        
            ## write npv and fuel_cost file
            output_dir_path            = "%s/no_retrofit/%s" % (self.output_dir_path, base_design_key)
            initializeDirHierarchy(output_dir_path)
            write_array_to_csv(self.NPV.dtype.names, self.NPV, "%s/npv.csv" % (output_dir_path))
            write_array_to_csv(self.fuel_cost.dtype.names, self.fuel_cost, "%s/fuel_cost.csv" % (output_dir_path))
            # write simmulation result
            output_dir_path  = "%s/no_retrofit" % (self.output_dir_path)
            output_file_path = "%s/%s_core%d.csv" % (output_dir_path, 'initial_design', index)
            lap_time         = convert_second(time.clock() - start_time)
            write_csv(column_names, [simulation_time, 
                                     self.hull.base_data['id'],
                                     self.engine.base_data['id'],
                                     self.propeller.base_data['id'],
                                     NPV, fuel_cost, self.base_design, '--', '--', self.route_change_date, lap_time], output_file_path)
            # for flexible design
            start_time                 = time.clock()
            np.random.seed(calc_seed(simulation_time))
            self.retrofit_mode         = retrofit_mode
            self.set_route_distance('A')
            self.bf_prob               = self.load_bf_prob()
            np.random.seed(calc_seed(simulation_time))
            NPV, fuel_cost             = self.simmulate(None, None, None, None, None, retrofit_design_key)        
            ## write npv and fuel_cost file
            output_dir_path            = "%s/flexible/%s" % (self.output_dir_path, base_design_key)
            initializeDirHierarchy(output_dir_path)
            write_array_to_csv(self.NPV.dtype.names, self.NPV, "%s/npv.csv" % (output_dir_path))
            write_array_to_csv(self.fuel_cost.dtype.names, self.fuel_cost, "%s/fuel_cost.csv" % (output_dir_path))
            # write simmulation result
            output_dir_path  = "%s/flexible" % (self.output_dir_path)
            output_file_path = "%s/%s_core%d.csv" % (output_dir_path, 'initial_design', index)
            lap_time         = convert_second(time.clock() - start_time)
            retrofit_design  = self.retrofit_design_human
            write_csv(column_names, [simulation_time, 
                                     self.hull.base_data['id'],
                                     self.engine.base_data['id'],
                                     self.propeller.base_data['id'],
                                     NPV, fuel_cost, self.base_design, self.retrofit_design_human(), self.retrofit_date_human(), self.route_change_date, lap_time], output_file_path)            
        return

    # conduct final simulation 
    def calc_whole_simulation_m(self, index, hull_list, engine_list, propeller_list, simulation_duration_years, devided_simulation_times):
        debug_mode = False
        # common process
        column_names     = ['simulation_time',
                            'hull_id',
                            'engine_id',
                            'propeller_id',
                            'NPV',
                            'fuel_cost',
                            'base_design',
                            'retrofit_design',
                            'retrofit_date',
                            'change_route_date']
        dtype            = np.dtype({'names': ('simulation_time', 'hull_id', 'engine_id', 'propeller_id', 'NPV', 'fuel_cost'),
                                     'formats': (np.int, np.int, np.int , np.int, np.float, np.float)})
        simulation_times = devided_simulation_times[index]        
        
        for simulation_time in simulation_times:
            # common process
            base_mode                = 'middle'
            self.simulate_log_index  = simulation_time
            np.random.seed(calc_seed(simulation_time))
            # define maket price on route A
            generate_market_scenarios(self.sinario, self.world_scale, self.flat_rate, self.sinario_mode, simulation_duration_years)
            self.world_scale_base = self.world_scale
            self.flat_rate_base   = self.flat_rate

            print "\n\n%20s: %d" % ('simulation'.upper(), simulation_time)

            # define maket price on route B
            self.world_scale_other.generate_sinario_with_oil_corr(self.sinario_mode, self.sinario.history_data[-1], self.sinario.predicted_data)
            self.flat_rate_other.generate_flat_rate(self.sinario_mode, simulation_duration_years)

            ### flexible for Route A (route_A) ###
            start_time   = time.clock()
            conduct_mode = 'route_a'

            self.set_market_prices('A')
            np.random.seed(calc_seed(simulation_time))
            base_design_key      = RETROFIT_DESIGNS['rough'][base_mode]
            retrofit_design_keys = { k:v for k,v in RETROFIT_DESIGNS['rough'].items() if not k == base_mode}
            self.retrofit_mode   = RETROFIT_MODE['significant_rule']
            self.reset_change_route_period()
            # set design
            component_ids                          = get_component_ids_from_design_key(base_design_key)
            self.hull, self.engine, self.propeller = get_component_from_id_array(map(int, component_ids), hull_list, engine_list, propeller_list)
            # conduct simulation
            print self.check_different_market_prices()
            NPV, fuel_cost       = self.simmulate(None, None, None, None, None, retrofit_design_keys)        
            # debug
            print "%20s: %4s years" % (conduct_mode, str(self.change_route_period) if (hasattr(self, 'change_route_period') and self.change_route_period is not None) else '----')
            ## write npv and fuel_cost file
            output_dir_path  = "%s/%s" % (self.output_dir_path, conduct_mode)
            output_file_path = "%s/simulation_result_core%d.csv" % (output_dir_path, index)
            initializeDirHierarchy(output_dir_path)
            lap_time         = convert_second(time.clock() - start_time)
            # debug
            if debug_mode:
                #self.display_debug_info('route A'.upper(), retrofit_design_keys)
                pass
            else:
                write_csv(column_names, [simulation_time, 
                                         self.hull.base_data['id'],
                                         self.engine.base_data['id'],
                                         self.propeller.base_data['id'],
                                         NPV, fuel_cost, self.base_design, self.retrofit_design_human(), self.retrofit_date_human(), self.route_change_date, lap_time], output_file_path)                
            ### flexible for Route A (route_A) ###

            ### flexible for Route B (route_B) ###
            start_time   = time.clock()
            conduct_mode = 'route_b'
            self.set_market_prices('B')

            np.random.seed(calc_seed(simulation_time))
            base_design_key      = RETROFIT_DESIGNS_FOR_ROUTE_CHANGE['rough'][base_mode]
            retrofit_design_keys = { k:v for k,v in RETROFIT_DESIGNS_FOR_ROUTE_CHANGE['rough'].items() if not k == base_mode}
            self.bf_prob         = self.load_bf_prob(True)
            self.retrofit_mode   = RETROFIT_MODE['significant_rule']
            self.reset_change_route_period()
            # set design
            component_ids                          = get_component_ids_from_design_key(base_design_key)
            self.hull, self.engine, self.propeller = get_component_from_id_array(map(int, component_ids), hull_list, engine_list, propeller_list)
            # conduct simulation
            print self.check_different_market_prices()
            NPV, fuel_cost       = self.simmulate(None, None, None, None, None, retrofit_design_keys)        
            # debug
            print "%20s: %4s years" % (conduct_mode, str(self.change_route_period) if (hasattr(self, 'change_route_period') and self.change_route_period is not None) else '----')
            ## write npv and fuel_cost file
            output_dir_path  = "%s/%s" % (self.output_dir_path, conduct_mode)
            output_file_path = "%s/simulation_result_core%d.csv" % (output_dir_path, index)
            initializeDirHierarchy(output_dir_path)
            lap_time         = convert_second(time.clock() - start_time)

            # debug
            if debug_mode:
                #self.display_debug_info('route B'.upper(), retrofit_design_keys)
                pass
            else:
                write_csv(column_names, [simulation_time, 
                                         self.hull.base_data['id'],
                                         self.engine.base_data['id'],
                                         self.propeller.base_data['id'],
                                         NPV, fuel_cost, self.base_design, self.retrofit_design_human(), self.retrofit_date_human(), self.route_change_date, lap_time], output_file_path)                            
            ### flexible for Route B (route_B) ###

            ### flexible for Route A + B based on probability (route_AB_prob) ###
            start_time   = time.clock()
            conduct_mode = 'route_ab_prob'
            self.set_market_prices('A')
            # change route change periods
            np.random.seed(calc_seed(simulation_time))
            self.enable_change_route()
            self.change_route_prob = CHANGE_ROUTE_PROB
            self.set_change_route_period()
            np.random.seed(calc_seed(simulation_time))
            base_design_key      = RETROFIT_DESIGNS['rough'][base_mode]
            retrofit_design_keys = { k:v for k,v in RETROFIT_DESIGNS['rough'].items() if not k == base_mode}
            self.retrofit_mode   = RETROFIT_MODE['route_change_merged']
            self.bf_prob         = self.load_bf_prob()
            # set design
            component_ids                          = get_component_ids_from_design_key(base_design_key)
            self.hull, self.engine, self.propeller = get_component_from_id_array(map(int, component_ids), hull_list, engine_list, propeller_list)
            # conduct simulation
            print self.check_different_market_prices()
            NPV, fuel_cost       = self.simmulate(None, None, None, None, None, retrofit_design_keys)        
            # debug
            print "%20s: %4s years" % (conduct_mode, str(self.change_route_period) if (hasattr(self, 'change_route_period') and self.change_route_period is not None) else '----')
            ## write npv and fuel_cost file
            output_dir_path  = "%s/%s" % (self.output_dir_path, conduct_mode)
            output_file_path = "%s/simulation_result_core%d.csv" % (output_dir_path, index)
            initializeDirHierarchy(output_dir_path)
            lap_time         = convert_second(time.clock() - start_time)
            # debug
            if debug_mode:
                #self.display_debug_info('route A, B prob'.upper(), retrofit_design_keys)
                pass
            else:
                write_csv(column_names, [simulation_time, 
                                         self.hull.base_data['id'],
                                         self.engine.base_data['id'],
                                         self.propeller.base_data['id'],
                                         NPV, fuel_cost, self.base_design, self.retrofit_design_human(), self.retrofit_date_human(), self.route_change_date, lap_time], output_file_path)            
            ### flexible for Route A + B based on probability (route_AB_prob) ###

            ### flexible for Route A + B based on market value (route_AB_market) ###
            start_time   = time.clock()
            conduct_mode = 'route_ab_market'
            self.set_market_prices('A')
            # change route change periods
            np.random.seed(calc_seed(simulation_time))
            self.enable_change_route()
            self.reset_change_route_period()
            self.change_sea_mode   = 'market_fluc'
            base_design_key        = RETROFIT_DESIGNS['rough'][base_mode]
            retrofit_design_keys   = { k:v for k,v in RETROFIT_DESIGNS['rough'].items() if not k == base_mode}
            self.retrofit_mode     = RETROFIT_MODE['route_change_merged']
            self.bf_prob           = self.load_bf_prob()
            # set design
            component_ids                          = get_component_ids_from_design_key(base_design_key)
            self.hull, self.engine, self.propeller = get_component_from_id_array(map(int, component_ids), hull_list, engine_list, propeller_list)
            # conduct simulation
            print self.check_different_market_prices()
            NPV, fuel_cost       = self.simmulate(None, None, None, None, None, retrofit_design_keys)        
            # debug
            print "%20s: %4s years" % (conduct_mode, str(self.change_route_period) if (hasattr(self, 'change_route_period') and self.change_route_period is not None) else '----')
            ## write npv and fuel_cost file
            output_dir_path  = "%s/%s" % (self.output_dir_path, conduct_mode)
            output_file_path = "%s/simulation_result_core%d.csv" % (output_dir_path, index)
            initializeDirHierarchy(output_dir_path)
            lap_time         = convert_second(time.clock() - start_time)
            # debug
            if debug_mode:
                #self.display_debug_info('route A, B market'.upper(), retrofit_design_keys)
                pass
            else:
                write_csv(column_names, [simulation_time, 
                                         self.hull.base_data['id'],
                                         self.engine.base_data['id'],
                                         self.propeller.base_data['id'],
                                         NPV, fuel_cost, self.base_design, self.retrofit_design_human(), self.retrofit_date_human(), self.route_change_date, lap_time], output_file_path)            
                
            ### flexible for Route A + B based on market value (route_AB_market) ###

            ### flexible for Route A + B based on answer of market value (route_AB_answer) ###
            start_time   = time.clock()
            conduct_mode = 'route_ab_answer'
            self.set_market_prices('A')
            # change route change periods
            np.random.seed(calc_seed(simulation_time))
            self.enable_change_route()
            self.reset_change_route_period()
            self.change_sea_mode   = 'market_fluc_ans'
            base_design_key        = RETROFIT_DESIGNS['rough'][base_mode]
            retrofit_design_keys   = { k:v for k,v in RETROFIT_DESIGNS['rough'].items() if not k == base_mode}
            self.retrofit_mode     = RETROFIT_MODE['route_change_merged']
            self.bf_prob           = self.load_bf_prob()
            # set design
            component_ids                          = get_component_ids_from_design_key(base_design_key)
            self.hull, self.engine, self.propeller = get_component_from_id_array(map(int, component_ids), hull_list, engine_list, propeller_list)
            # conduct simulation
            print self.check_different_market_prices()
            NPV, fuel_cost       = self.simmulate(None, None, None, None, None, retrofit_design_keys)        
            # debug
            print "%20s: %4s years" % (conduct_mode, str(self.change_route_period) if (hasattr(self, 'change_route_period') and self.change_route_period is not None) else '----')
            ## write npv and fuel_cost file
            output_dir_path  = "%s/%s" % (self.output_dir_path, conduct_mode)
            output_file_path = "%s/simulation_result_core%d.csv" % (output_dir_path, index)
            initializeDirHierarchy(output_dir_path)
            lap_time         = convert_second(time.clock() - start_time)
            # debug
            if debug_mode:
                #self.display_debug_info('route A, B answer'.upper(), retrofit_design_keys)
                pass
            else:
                write_csv(column_names, [simulation_time, 
                                         self.hull.base_data['id'],
                                         self.engine.base_data['id'],
                                         self.propeller.base_data['id'],
                                         NPV, fuel_cost, self.base_design, self.retrofit_design_human(), self.retrofit_date_human(), self.route_change_date, lap_time], output_file_path)            
            ### flexible for Route A + B based on market value (route_AB_market) ###
            self.set_market_prices('A')

        return

    ## multi processing method ##
    def only_create_velocity_combinations(self):
        # load components list
        hull_list              = load_hull_list()
        engine_list            = load_engine_list()
        propeller_list         = load_propeller_list()        

        devided_component_ids = []
        for hull_info in hull_list:
            for engine_info in engine_list:
                for propeller_info in propeller_list:
                    devided_component_ids.append([hull_info['id'], engine_info['id'], propeller_info['id']])
        devided_component_ids = np.array_split(devided_component_ids, PROC_NUM)

        # initialize
        pool = mp.Pool(PROC_NUM)

        # multi processing #
        callback              = [pool.apply_async(self.calc_velocity_combinations_m, args=(index, devided_component_ids, hull_list, engine_list, propeller_list)) for index in xrange(PROC_NUM)]
        pool.close()
        pool.join()

        return 

    def calc_velocity_combinations_m(self, index, devided_component_ids, hull_list, engine_list, propeller_list):
        for component_ids in devided_component_ids[index]:
            hull, engine, propeller = get_component_from_id_array(component_ids, hull_list, engine_list, propeller_list)
            self.create_velocity_combination(hull, engine, propeller)
            print "velocity combination of %10s has just generated." % (generate_combination_str(hull, engine, propeller))
        return 

    # draw rps - velocity combinations
    def draw_combinations(self, hull, engine, propeller, combinations, dir_name):
        combination_str = generate_combination_str(hull, engine, propeller)
        dir_name        = "%s/%s"     % (dir_name, combination_str)
        filename        = "%s/%s_rps_v.png" % (dir_name, combination_str)
        title           = "%s of %s"  % ("revolution and velocity combination".title(),
                                         combination_str)

        x_label = "rpm".upper()
        y_label = "%s %s" % ('velocity'.upper(), '[knot]')
        # initialize directory
        initializeDirHierarchy(dir_name)
        graphInitializer(title,
                         x_label,
                         y_label)
        colors = ['b', 'r']
        for key, load_condition in LOAD_CONDITION.items():
            draw_data = np.array(combinations[load_condition])
            x_data    = draw_data.transpose()[0]
            y_data    = draw_data.transpose()[1]
            plt.plot(x_data, y_data, label=load_condition, color=colors[key])
        plt.legend(shadow=True)
        plt.legend(loc='upper left')
        plt.ylim([0, 30])
        plt.savefig(filename)
        plt.close()
        return

    def draw_BHP_rpm_graph(self, hull, engine, propeller, combinations, dir_name):
        combination_str = generate_combination_str(hull, engine, propeller)
        dir_name        = "%s/%s"     % (dir_name, combination_str)
        filename        = "%s/%s_BHP_rpm.png" % (dir_name, combination_str)
        title           = "BHP %s of %s"  % ("and rpm combination".title(),
                                             combination_str)

        x_label = "rpm".upper()
        y_label = "%s %s" % ('bhp'.upper(), '[kW]')
        # initialize directory
        initializeDirHierarchy(dir_name)
        graphInitializer(title,
                         x_label,
                         y_label)
        colors = ['b', 'r']
        load_condition = LOAD_CONDITION[0]
        draw_data = np.array(combinations[load_condition])
        # rpm
        x_data    = draw_data.transpose()[0]
        # calc BHP
        y_data = np.array([])
        for rpm in x_data:
            bhp = self.get_modified_bhp(rpm, engine)
            y_data = np.append(y_data, bhp)
        plt.plot(x_data, y_data, color=colors[0])
        plt.savefig(filename)
        plt.close()
        return                
    
    # draw EHP-velocity graphs
    def draw_EHP_Knot_graph(self, hull, engine, propeller, combinations, dir_name):
        combination_str = generate_combination_str(hull, engine, propeller)
        dir_name        = "%s/%s"     % (dir_name, combination_str)
        filename        = "%s/%s_EHP_v.png" % (dir_name, combination_str)
        title           = "EHP %s of %s"  % ("and velocity combination".title(),
                                             combination_str)

        x_label = "velocity".upper() + "[knot]"
        y_label = "%s %s" % ('ehp'.upper(), '[kW]')
        # initialize directory
        initializeDirHierarchy(dir_name)
        graphInitializer(title,
                         x_label,
                         y_label)
        colors = ['b', 'r']
        for key, load_condition in LOAD_CONDITION.items():
            draw_data = np.array(combinations[load_condition])
            # knot 
            x_data    = draw_data.transpose()[1]
            # calc EHP
            y_data = np.array([])
            for v_knot in x_data:
                y_data = np.append(y_data, self.calc_EHP(hull, engine, propeller, v_knot, load_condition, combinations))
            plt.plot(x_data, y_data, label=load_condition, color=colors[key])
        plt.legend(shadow=True)
        plt.legend(loc='upper left')
        plt.savefig(filename)
        plt.close()
        return        
        
    def write_combinations_as_json(self, hull, engine, propeller, combinations, dir_name):
        combination_str = generate_combination_str(hull, engine, propeller)
        dir_name        = "%s/%s"     % (dir_name, combination_str)
        output_path     = "%s/%s_combinations.json" % (dir_name, combination_str)

        # to serialize numpy ndarray #
        json_serialized_combinations = {}
        for condition in combinations.keys():
            json_serialized_combinations[condition] = combinations[condition]

        f = open(output_path, 'w')
        json_data = json.dumps(json_serialized_combinations, indent=4)
        f.write(json_data)
        f.close()        
        return

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

    def check_significant_retrofit(self):
        if not self.retrofittable():
            return False
        retrofit_flag = False
        retrofit_duration_years = min(self.calc_years_to_retire(), 3)
        retrofit_simulate_count = 10

        base_seed_num = COMMON_SEED_NUM * 2

        # get retrofit components
        retrofit_hull, retrofit_engine, retrofit_propeller = self.get_retrofit_components()
        npv_for_current_design  = {}
        npv_for_retrofit_design = {}
        for scenario_num in range(retrofit_simulate_count):
            # current design #
            # fix the random seed #
            np.random.seed(base_seed_num + scenario_num + self.simulate_log_index)
            ## generate scenairo, world scale and flat rate
            sinario                    = Sinario(self.sinario.history_data)
            world_scale                = WorldScale(self.world_scale.history_data)
            flat_rate                  = FlatRate(self.flat_rate.history_data)
            generate_market_scenarios(sinario, world_scale, flat_rate, self.sinario_mode, retrofit_duration_years)
            # conduct simulation #
            agent                                = Agent(sinario, world_scale, flat_rate, RETROFIT_MODE['none'], self.sinario_mode, self.bf_mode, self.hull, self.engine, self.propeller)
            start_date                           = search_near_index(self.current_date, sinario.predicted_data['date'])
            end_date                             = add_year(self.current_date, retrofit_duration_years)            
            agent.operation_date_array           = generate_operation_date(start_date, end_date)
            agent.simulate_log_index             = self.simulate_log_index
            # get component
            NPV, fuel_cost                       = agent.simmulate()
            npv_for_current_design[scenario_num] = NPV
            
            # retrofitted design
            # fix the random seed #
            np.random.seed(base_seed_num + scenario_num + self.simulate_log_index)
            generate_market_scenarios(sinario, world_scale, flat_rate, self.sinario_mode, retrofit_duration_years)
            # conduct simulation #
            agent                                 = Agent(sinario, world_scale, flat_rate, RETROFIT_MODE['none'], self.sinario_mode, self.bf_mode, retrofit_hull, retrofit_engine, retrofit_propeller)
            start_date                            = search_near_index(self.current_date, sinario.predicted_data['date'])
            end_date                              = add_year(self.current_date, retrofit_duration_years)                        
            agent.operation_date_array            = generate_operation_date(start_date, end_date)
            agent.simulate_log_index              = self.simulate_log_index            
            # get component
            NPV, fuel_cost                        = agent.simmulate()
            npv_for_retrofit_design[scenario_num] = NPV
       
        # output NPV 
        dockin_date_str = "%s-%s" % (self.current_date, self.latest_dockin_date)
        output_dir_path = "%s/dock-in-log/simulate%d/%s" % (self.output_dir_path, self.simulate_log_index, dockin_date_str)
        initializeDirHierarchy(output_dir_path)
        write_file_as_json(npv_for_current_design, "%s/npv_current_design.json" % (output_dir_path))
        write_file_as_json(npv_for_retrofit_design, "%s/npv_retrofit_design.json" % (output_dir_path))
        # output NPV

        # update flag
        # subtract retrofits cost
        #retrofit_cost = self.check_component_changes(retrofit_hull, retrofit_engine, retrofit_propeller)
        #NPV           -= retrofit_cost

        average_npv_of_rd = np.average(npv_for_retrofit_design.values())
        average_npv_of_ct = np.average(npv_for_current_design.values())

        retrofit_flag = False
        if average_npv_of_rd > average_npv_of_ct:
            retrofit_flag = True
            # write file
            output_file_path = "%s/dock-in-log/simulate%d/retrofit_log.txt" % (self.output_dir_path, self.simulate_log_index)
            f = open(output_file_path, 'w')
            f.write("retrofit is conducted on %s \n" % (self.current_date))
            f.write("average NPV of current design: %20lf \n" % (average_npv_of_ct))
            f.write("average NPV of retrofit design: %20lf \n" % (average_npv_of_rd))
            f.close()            
        return retrofit_flag

    def check_significant_rule_retrofit(self):
        retrofit_flag = False
        mode          = None
        # Block
        if not self.retrofittable():
            return retrofit_flag, mode

        # Block for route change
        ## retrofit occured after route changes
        if not self.retrofittable_after_route_change():
            return retrofit_flag, mode        

        # get analysis oil data (origin_date -> current_date)
        origin_oilprice      = self.sinario.predicted_data[0]['price']
        current_index        = search_near_index(self.current_date, self.sinario.predicted_data['date'])
        start_index = search_near_index(self.current_date - datetime.timedelta(days=365*2), self.sinario.predicted_data['date'])
        analysis_period_data = self.sinario.predicted_data[np.where(self.sinario.predicted_data['date']==start_index)[0]:np.where(self.sinario.predicted_data['date']==current_index)[0]]
        avg_oilprice         = np.average(analysis_period_data['price'])
        
        # calc analysis trend
        analysis_data = np.array([[index, v['price']] for index, v in enumerate(analysis_period_data)])
        cons, trend   = estimate(analysis_data.transpose()[0], analysis_data.transpose()[1], 1)

        ## debug
        '''
        print "trend: %3.3lf" % (trend)
        print "trend_rule: %3.3lf" % (self.rules['trend'])
        print "origin_price: %3.3lf, avg_oilprice: %3.3lf" % (origin_oilprice, avg_oilprice)
        print "high: %3.3lf" % (origin_oilprice * (1 + self.rules['delta']))
        print "low: %3.3lf" % (origin_oilprice * (1 - self.rules['delta']))
        '''

        # change flag
        ## High
        if trend > self.rules['trend']:
            mode = 'high'
            retrofit_flag = True
        elif trend < self.rules['trend'] * (-1):
            mode = 'low'
            retrofit_flag = True
        else:
            if (avg_oilprice > origin_oilprice * (1 + self.rules['delta'])):
                mode          = 'high'
                retrofit_flag = True
            elif avg_oilprice < origin_oilprice * (1 - self.rules['delta']):
                mode          = 'low'
                retrofit_flag = True

        # avoid comparison with another route designs
        if self.change_sea_flag:
            return retrofit_flag, mode

        # remove current mode
        if not (retrofit_flag and self.retrofit_design_keys.has_key(mode)):
            retrofit_flag = False
            
        # remove the same design
        if not (retrofit_flag and self.current_design_key() != self.retrofit_design_keys[mode]):
            retrofit_flag = False

        return retrofit_flag, mode

    def check_route_change_retrofit(self):
        retrofit_flag = False
        mode          = None
        # Block
        if not self.retrofittable():
            return retrofit_flag, mode

        if self.retrofittable_after_route_change():
            retrofit_flag = True
            mode = 'middle'

        return retrofit_flag, mode

    def check_route_change_retrofit_merged(self):
        retrofit_flag = False
        mode          = None
        # Block
        if not self.retrofittable():
            return retrofit_flag, mode

        # get analysis oil data (origin_date -> current_date)
        origin_oilprice      = self.sinario.predicted_data[0]['price']
        current_index        = search_near_index(self.current_date, self.sinario.predicted_data['date'])
        start_index = search_near_index(self.current_date - datetime.timedelta(days=365*2), self.sinario.predicted_data['date'])
        analysis_period_data = self.sinario.predicted_data[np.where(self.sinario.predicted_data['date']==start_index)[0]:np.where(self.sinario.predicted_data['date']==current_index)[0]]
        avg_oilprice         = np.average(analysis_period_data['price'])
        
        # calc analysis trend
        analysis_data = np.array([[index, v['price']] for index, v in enumerate(analysis_period_data)])
        cons, trend   = estimate(analysis_data.transpose()[0], analysis_data.transpose()[1], 1)

        ## debug
        '''
        print "trend: %3.3lf" % (trend)
        print "trend_rule: %3.3lf" % (self.rules['trend'])
        print "origin_price: %3.3lf, avg_oilprice: %3.3lf" % (origin_oilprice, avg_oilprice)
        print "high: %3.3lf" % (origin_oilprice * (1 + self.rules['delta']))
        print "low: %3.3lf" % (origin_oilprice * (1 - self.rules['delta']))
        '''

        # change flag
        ## High
        if trend > self.rules['trend']:
            mode = 'high'
            retrofit_flag = True
        elif trend < self.rules['trend'] * (-1):
            mode = 'low'
            retrofit_flag = True
        else:
            if (avg_oilprice > origin_oilprice * (1 + self.rules['delta'])):
                mode          = 'high'
                retrofit_flag = True
            elif avg_oilprice < origin_oilprice * (1 - self.rules['delta']):
                mode          = 'low'
                retrofit_flag = True

        # avoid comparison with another route designs
        if self.after_route_change():
            if not retrofit_flag:
                retrofit_flag = True
                mode          = 'middle'
            return retrofit_flag, mode

        # remove current mode
        if not (retrofit_flag and self.retrofit_design_keys.has_key(mode)):
            retrofit_flag = False
            
        # remove the same design
        if not (retrofit_flag and self.current_design_key() != self.retrofit_design_keys[mode]):
            retrofit_flag = False

        return retrofit_flag, mode

    def calc_EHP(self, hull, engine, propeller, v_knot, load_condition, combination):
        raw_ehp     = hull.calc_raw_EHP(v_knot, load_condition)
        index       = np.where(np.array(combination[load_condition]).transpose()[1]==v_knot)[0][0]
        rpm         = combination[load_condition][index][0]
        rps         = rpm2rps(rpm)
        velocity_ms = knot2ms(v_knot)
        J           = propeller.calc_advance_constant(velocity_ms, rps)

        # consider efficiency
        bhp          = propeller.EHP2BHP(raw_ehp, propeller, rps, J, velocity_ms)
        modified_bhp = engine.consider_efficiency(rpm, bhp)
        ret_ehp      = propeller.BHP2EHP(modified_bhp, propeller, rps, J, velocity_ms)
        return ret_ehp    

    def check_component_changes(self, hull, engine, propeller):
        retrofit_cost = 0
        if not self.hull.base_data == hull.base_data:
            # not implemented yet
            print_with_notice("Error: hull is retrofitted, no way")
            raise

        if not self.engine.base_data == engine.base_data:
            retrofit_cost += RETROFIT_COST['engine']
        if not self.propeller.base_data == propeller.base_data:
            retrofit_cost += RETROFIT_COST['propeller']
        return retrofit_cost

    # fare = world_scale * flat_rate (e.g. WS50)
    def calc_fare(self):
        # return world_scale * flat_rate
        return calc_fare_with_params(self.current_world_scale, self.current_flat_rate)

    def retrofit_mode_to_human(self):
        return [key for key, value in RETROFIT_MODE.iteritems() if value == self.retrofit_mode][0]

    def update_CF_log(self, CF_day):
        if CF_day is None:
            return
        ## make zero CF when the ship condition changes
        #CF_day           = 0 if CF_day is None else CF_day
        column_names     = ['date', 'cash_flow']
        write_data       = [self.current_date, CF_day]
        output_file_path = "%s/CF_log.csv" % (self.output_dir_path)
        write_csv(column_names, write_data, output_file_path)        
        return

    # beaufort mode
    def load_bf_prob(self, next_route_flg=False):
        bf_prob = None
        if self.bf_mode == BF_MODE['rough']:
            if next_route_flg:
                alpha = 7.0
                beta  = 3.0
            else:
                alpha = 5.5
                beta  = 4.5
        elif self.bf_mode == BF_MODE['calm']:
            return bf_prob

        return load_bf_prob_with_param(alpha, beta)

    # consider beaufort for velocity
    def modify_by_external(self, v_knot):
        # for no external modification
        if self.bf_mode is None or self.bf_prob is None:
            return v_knot
        current_bf          = prob_with_weight(self.bf_prob)
        current_wave_height = get_wave_height(current_bf, self.bf_info)
        delta_v = calc_y(current_wave_height, [V_DETERIO_FUNC_COEFFS['cons'], V_DETERIO_FUNC_COEFFS['lin'], V_DETERIO_FUNC_COEFFS['squ']], V_DETERIO_M)
        # reduce for bow
        delta_v = self.hull.consider_bow_for_wave(delta_v, self.load_condition)
        return v_knot + delta_v

    # consider dock-to-dock deterioration    
    def update_d2d(self):
        navigation_elapsed_days = self.ballast_trip_days + self.return_trip_days + PORT_DWELL_DAYS
        day_coeff = float( navigation_elapsed_days / 365 )
        # velocity
        # deterioration rate of velocity per year
        d_rate_v                 = 1.0
        delta_v                  = d_rate_v * day_coeff
        self.d2d_det['v_knot']  += delta_v
        # rpm
        d_rate_rpm               = np.random.uniform(2.0, 4.0)
        delta_rpm                = d_rate_rpm * day_coeff
        self.d2d_det['rpm']     += delta_rpm
        # EHP
        d_rate_ehp               = np.random.uniform(20, 60)
        delta_ehp                = d_rate_ehp * day_coeff
        self.d2d_det['ehp']     += delta_ehp
        return

    def clear_d2d(self):
        self.d2d_det = {'v_knot': 0, 'rpm': 0, 'ehp': 0}
        return

    def update_age_effect(self):
        # velocity
        delta_v = 0.1 * DOCK_IN_PERIOD
        self.age_eff['v_knot'] += delta_v
        # rpm
        delta_rpm = np.random.uniform(0.5, 2.0) * DOCK_IN_PERIOD
        self.age_eff['rpm'] += delta_rpm
        # EHP
        delta_ehp = 2.5 * DOCK_IN_PERIOD
        self.age_eff['ehp'] += delta_ehp
        return

    def clear_age_effect(self):
        self.age_eff = {'v_knot': 0, 'rpm': 0, 'ehp': 0}
        return
    
    ## consider deterioration    
    def modify_by_deterioration(self, rpm, v_knot):
        modified_rpm    = rpm
        modified_v_knot = v_knot
        # dock-to-dock
        ## rpm
        modified_rpm -= self.d2d_det['rpm']
        ## v_knot
        modified_v_knot -= self.d2d_det['v_knot']
        # age effect
        ## rpm
        modified_rpm -= self.age_eff['rpm']
        ## v_knot
        modified_v_knot -= self.age_eff['v_knot']

        return modified_rpm, modified_v_knot

    def display_current_design(self):
        print "Hull: %s, Engine: %s, Propeller: %s" % (self.hull.base_data['id'], self.engine.base_data['id'], self.propeller.base_data['id'])
        return

    def current_design_key(self):
        return generate_combination_str(self.hull, self.engine, self.propeller)

    def display_market_factors(self):
        print "market factors on %s" % str(self.current_date)
        for key in MARKET_FACTOR_KEYS:
            designated_key = "self.current_%s" % (key)
            print "%20s: %20s" % (key.upper(), eval(designated_key))
        return
    
    def generate_significant_senarios(self, simulation_end_date, scenario_type):
        scenario          = []
        world_scale       = []
        flat_rate         = []
       
        prediction_date_array = generate_operation_date(self.current_date, simulation_end_date)
        if scenario_type == 'maintain':
            for prediction_date in prediction_date_array:
                for market_factor in MARKET_FACTOR_KEYS:
                    designated_market_factor = market_factor if not market_factor == 'oilprice' else 'scenario'
                    current_value            = eval("self.current_%s" % (market_factor))
                    eval(designated_market_factor).append( ( datetime_to_human(prediction_date), current_value) )
        else:
            fluc_ratio = RISE_RATIO if scenario_type == 'rise' else DECLINE_RATIO
            for prediction_date in prediction_date_array:
                x             = np.where(prediction_date_array==prediction_date)[0][0]
                initial_price = self.sinario.history_data[0]['price']
                oilprice      = calc_simple_oilprice(len(prediction_date_array), x, initial_price, fluc_ratio)
                scenario.append( (datetime_to_human(prediction_date), oilprice) )
                world_scale.append( (datetime_to_human(prediction_date), self.world_scale.calc_ws_with_oilprice(oilprice)) )
                flat_rate.append( (datetime_to_human(prediction_date), self.flat_rate.history_data[-1]['fr']) )
        # configure format
        scenario    = np.array( scenario, dtype=self.sinario.predicted_data.dtype)
        world_scale = np.array( world_scale, dtype=self.world_scale.predicted_data.dtype)
        flat_rate   = np.array( flat_rate, dtype=self.flat_rate.predicted_data.dtype)

        return scenario, world_scale, flat_rate

    # hull, engine, propeller
    def get_retrofit_components(self):
        if self.retrofit_design_key is None:
            return None
        # load components list
        hull_list           = load_hull_list()
        engine_list         = load_engine_list()
        propeller_list      = load_propeller_list()                
        retrofit_component_ids = map(int, get_component_ids_from_design_key(self.retrofit_design_key))
        return get_component_from_id_array(retrofit_component_ids, hull_list, engine_list, propeller_list)

    def conduct_retrofit(self, retrofit_design=None):
        if retrofit_design is None:
            self.hull, self.engine, self.propeller = self.get_retrofit_components()
        else:
            # load components list
            hull_list                              = load_hull_list()
            engine_list                            = load_engine_list()
            propeller_list                         = load_propeller_list()                
            retrofit_component_ids                 = map(int, get_component_ids_from_design_key(retrofit_design))
            self.hull, self.engine, self.propeller = get_component_from_id_array(retrofit_component_ids, hull_list, engine_list, propeller_list)
            self.retrofit_date                     = self.current_date
            self.retrofit_design                   = retrofit_design
        return

    def retrofittable(self):
        return (self.retrofit_mode != RETROFIT_MODE['none']) and (self.retrofit_count_limit != 0)

    # calc years to retire with integer
    def calc_years_to_retire(self):
        return (self.retire_date - self.current_date).days / 365

    def draw_velocity_log(self, output_dir_path):
        colors      = {}
        temp_colors = ['b', 'r', 'g']
        for index, key in enumerate(self.log.keys()):
            colors[key] = temp_colors[index]
        combination_str = generate_combination_str(self.hull, self.engine, self.propeller)
        dir_name = "%s/velocity_logs" % (output_dir_path)
        initializeDirHierarchy(dir_name)

        # generate log as csv file
        output_csv_path = "%s/%s.csv" % (dir_name, combination_str)
        output_data = np.array([ (_d[0]['date'], _d[0]['rpm'], _d[0]['velocity'], 'full') for _d in self.log['full']] + [ (_d[0]['date'], _d[0]['rpm'], _d[0]['velocity'], 'ballast') for _d in self.log['ballast']])
        output_data = sorted(output_data, key=lambda x : x[0])
        write_simple_array_csv(['date', 'rpm', 'velocity', 'load_condition'], output_data, output_csv_path)

        for element in ['rpm', 'velocity']:
            title   = "%s %s log" % (combination_str, element)
            x_label = "Date"
            unit    = '' if element == 'rpm' else 'knot'
            y_label = "%s %s" % (element, unit)
            graphInitializer(title,
                             x_label,
                             y_label)
            output_file_path = "%s/%s_%s.png" % (dir_name, combination_str, element)
            for condition in self.log.keys():
                draw_data = [ [datetime.datetime.strptime(data['date'][0], '%Y/%m/%d'), data[element][0]] for data in self.log[condition]]
                draw_data = np.array(sorted(draw_data, key= lambda x : x[0]))
                plt.bar(draw_data.transpose()[0], draw_data.transpose()[1], color=colors[condition])
            plt.xlim(self.origin_date, self.retire_date)
            plt.savefig(output_file_path)
        return

    def origin_debug(self):
        ret_flag = False
        if self.current_date == self.origin_date:
            if (self.hull is not None) and (self.engine is not None) and (self.propeller is not None):
                print generate_combination_str(self.hull, self.engine, self.propeller)
            ret_flag = True
        return ret_flag

    def retrofit_date_human(self):
        return datetime_to_human(self.retrofit_date) if self.retrofit_date is not None else '--'

    def retrofit_design_human(self):
        return self.retrofit_design if self.retrofit_date is not None else '--'

    # change route based on prob
    def change_route(self):
        if not self.can_change_route():
            return
        
        if self.should_route_change():
            # beaufort mode
            self.bf_prob              = self.load_bf_prob(True)
            self.set_route_distance('B')
            self.change_sea_count     = 0
            self.route_change_date    = self.current_date
            self.retrofit_design_keys = RETROFIT_DESIGNS_FOR_ROUTE_CHANGE['rough']
            change_route_period = calc_delta_from_origin(self.current_date)
            self.change_route_period  = change_route_period if change_route_period != 0 else None
            if hasattr(self, 'world_scale_base'):
                self.set_market_prices('B')

        return

    def should_route_change(self):
        if hasattr(self, 'change_sea_mode') and (self.change_sea_mode is not None):
            if self.change_sea_mode == 'market_fluc':
                ret_flag = True if self.check_route_fare_trend() else False
            elif self.change_sea_mode == 'market_fluc_ans':
                ret_flag = True if self.check_route_fare_trend_ans() else False
            return ret_flag

        if not hasattr(self, 'change_route_period'):
            return False

        if (self.change_route_period == 0) or (self.change_route_period is None):
            return False

        # time elapsed
        return self.current_date > add_year(self.origin_date, self.change_route_period)
                 
    def check_route_fare_trend_ans(self):
        ret_flag = False
        current_index_ws   = search_near_index(self.current_date, self.world_scale.predicted_data['date'])
        start_index_raw    = np.where(self.world_scale.predicted_data['date']==current_index_ws)[0]
        analysis_period_ws = self.world_scale.predicted_data[start_index_raw:]
        other_world_sacle  = self.world_scale_other.predicted_data[start_index_raw:]

        flat_rate_induces = []
        for ws_date in analysis_period_ws['date']:
            # current route
            current_index_fr = search_near_index(str_to_date(ws_date), self.flat_rate.predicted_data['date'])
            current_index_fr = np.where(self.flat_rate.predicted_data['date']==current_index_fr)[0]
            flat_rate_induces.append(current_index_fr[0])

        flat_rate_induces = np.array(flat_rate_induces)
        current_fares     = calc_fare_with_params(analysis_period_ws['ws'], self.flat_rate.predicted_data[flat_rate_induces]['fr'])
        next_fares        = calc_fare_with_params(self.world_scale_other.predicted_data[start_index_raw:]['ws'], self.flat_rate_other.predicted_data[flat_rate_induces]['fr'])

        # condition to change the route
        if np.average(next_fares) > np.average(current_fares):
            ret_flag = True

        # debug
        '''
        print "fare trend: %3.3lf" % (np.average(next_fares) / np.average(current_fares))
        '''
        
        return ret_flag

    def check_route_fare_trend(self):
        ret_flag = False
        current_index_ws   = search_near_index(self.current_date, self.world_scale.predicted_data['date'])
        start_index_ws     = search_near_index(self.current_date - datetime.timedelta(days=365*2), self.world_scale.predicted_data['date'])
        start_index_raw    = np.where(self.world_scale.predicted_data['date']==start_index_ws)[0]
        end_index_raw      = np.where(self.world_scale.predicted_data['date']==current_index_ws)[0]
        analysis_period_ws = self.world_scale.predicted_data[start_index_raw:end_index_raw]

        other_world_sacle = self.world_scale_other.predicted_data[start_index_raw:end_index_raw]

        flat_rate_induces = []
        for ws_date in analysis_period_ws['date']:
            # current route
            current_index_fr = search_near_index(str_to_date(ws_date), self.flat_rate.predicted_data['date'])
            current_index_fr = np.where(self.flat_rate.predicted_data['date']==current_index_fr)[0]
            flat_rate_induces.append(current_index_fr[0])

        flat_rate_induces = np.array(flat_rate_induces)

        current_fares = calc_fare_with_params(analysis_period_ws['ws'], self.flat_rate.predicted_data[flat_rate_induces]['fr'])
        next_fares    = calc_fare_with_params(self.world_scale_other.predicted_data[start_index_raw:end_index_raw]['ws'], self.flat_rate_other.predicted_data[flat_rate_induces]['fr'])

        # condition to change the route
        if np.average(next_fares) > np.average(current_fares) * CHANGE_ROUTE_MARKET_RATE:
            ret_flag = True

        # debug
        '''
        print "fare trend: %3.3lf" % (np.average(next_fares) / np.average(current_fares))
        '''

        return ret_flag

    def can_change_route(self):
        return self.change_sea_count > 0

    # check the route change already conducted
    def after_route_change(self):
        return (self.change_sea_count == 0) and (self.change_sea_flag)

    # check the route change already conducted and retrofittable
    def retrofittable_after_route_change(self):
        # Block(retrofittable when change_sea_flag is False)
        if not self.change_sea_flag:
            return True

        return self.after_route_change()

    def display_debug_info(self, debug_mode, retrofit_design_keys):
        print "\n%40s" % ('-'*40)
        print "%20s" % (debug_mode)
        print "%20s: %50s" % ('base design'.title(), self.display_current_design())
        print "%20s: %50s" % ('retrofit mode'.title(), self.retrofit_mode_to_human())
        print "%20s: %50s" % ('retrofit design keys'.title(), retrofit_design_keys)
        print "%20s: %50s" % ('bf mode'.title(), self.bf_prob)
        print "%20s: %50s" % ('origin date'.title(), self.sinario.history_data['date'][-1])
        print "%20s: %50s" % ('retire date'.title(), self.sinario.predicted_data['date'][-1])
        print "%20s: %50s" % ('change route period'.title(), str(self.change_route_period) if hasattr(self, 'change_route_period') else 'None')
        print "%20s: %50s" % ('change sea mode'.title(), self.change_sea_mode if hasattr(self, 'change_sea_mode') else 'None')
        market = 'A' if self.world_scale == self.world_scale_base else 'B'
        print "%20s: %50s" % ('market'.title(), market.upper())
        return

    def set_market_prices(self, route_type):
        if route_type == 'A':
            self.world_scale = self.world_scale_base
            self.flat_rate   = self.flat_rate_base
        else:
            self.world_scale = self.world_scale_other
            self.flat_rate   = self.flat_rate_other
        return

    def enable_change_route(self):
        self.change_sea_flag   = True
        self.change_sea_count  = 1
        return

    def check_different_market_prices(self):
        world_scale_flag = all(self.world_scale_base.predicted_data == self.world_scale_other.predicted_data)
        flat_rate_flag = all(self.flat_rate_base.predicted_data == self.flat_rate_other.predicted_data)
        return world_scale_flag and flat_rate_flag

    def set_route_distance(self, route):
        if route == 'A':
            self.current_navigation_distance = NAVIGATION_DISTANCE_A
        elif route == 'B':
            self.current_navigation_distance = NAVIGATION_DISTANCE_B
        self.round_trip_distance   = self.current_navigation_distance * 2.0
        self.left_distance_to_port = self.current_navigation_distance
        return
