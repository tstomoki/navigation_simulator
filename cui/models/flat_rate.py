# import common modules #
import sys
import math
from pdb import *
import matplotlib.pyplot as plt
import numpy as np
from types import *
# import common modules #

# import own modules #
sys.path.append('../public')
from my_modules import *
from constants  import *
# import own modules #

# constants #
RESULTSDIR = '../results/'
# constants #

class FlatRate:
    def __init__(self, history_data=None, neu=None, sigma=None, u=None, d=None, p=None):
        self.default_xlabel = "date".title()
        self.default_ylabel = "flat rate".title() + " [%]"
        if history_data is None:
            self.history_data = load_flat_rate_history_data()
            self.draw_history_data()
        else:
            self.history_data = history_data
        # initialize parameters
        if (neu is None or sigma is None or u is None or d is None or p is None):
            self.calc_params_from_history()
        else:
            self.neu, self.sigma, self.u, self.d, self.p = neu, sigma, u, d, p

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
            
    def show_history_data(self):
        print '----------------------'
        print '- - history data - - '
        print '----------------------'

        for data in self.history_data:
            print "%10s : %10lf" % (data['date'], data['price'])

    def show_predicted_data(self):
        print '----------------------'
        print '- - predicted data - - '
        print '----------------------'

        for data in self.predicted_data:
            print "%10s : %10lf" % (data['date'], data['price'])


    def draw_history_data(self):
        title = "flat rate historical data".title()
        graphInitializer(title,
                         self.default_xlabel,
                         self.default_ylabel)

        draw_data = [ [datetime.datetime.strptime(data['date'], '%Y/%m/%d'), data['fr']] for data in self.history_data]
        draw_data = np.array(sorted(draw_data, key= lambda x : x[0]))
        
        plt.plot(draw_data.transpose()[0],
                 draw_data.transpose()[1],
                 color='#9370DB', lw=5, markersize=0, marker='o')
        plt.xlim([draw_data.transpose()[0].min(), draw_data.transpose()[0].max()])
        plt.ylim([0, 100])

        output_file_path = "%s/graphs/%s.png" % (RESULTSDIR, title)
        plt.savefig(output_file_path)
        return

    def draw_predicted_data(self):
        title = "flat rate predicted data".title()
        graphInitializer(title,
                         self.default_xlabel,
                         self.default_ylabel)

        draw_data = [ [datetime.datetime.strptime(data['date'], '%Y/%m/%d'), data['fr']] for data in self.predicted_data]
        draw_data = np.array(sorted(draw_data, key= lambda x : x[0]))

        plt.plot(draw_data.transpose()[0],
                 draw_data.transpose()[1],
                 color='#9370DB', lw=5, markersize=0, marker='o')
        plt.xlim([draw_data.transpose()[0].min(), draw_data.transpose()[0].max()])
        plt.ylim([0, 100])        
        output_file_path = "%s/graphs/%s.png" % (RESULTSDIR, title)
        plt.savefig(output_file_path)
        return

    # generate predicted sinario
    def generate_flat_rate(self, sinario_mode, predict_years=DEFAULT_PREDICT_YEARS):
        # default predict_years is 15 years [180 months]
        self.predict_years  = predict_years

        # predicted data type
        dt   = np.dtype({'names': ('date', 'fr'),
                         'formats': ('S10' , np.float)})
        self.predicted_data = np.array([], dtype=dt)

        # latest date from history_data
        latest_history_date_str, latest_flatrate = self.history_data[-1]
        latest_history_date                      = datetime.datetime.strptime(latest_history_date_str, '%Y/%m/%d')

        end_date         = add_year(latest_history_date, predict_years)
        current_date     = latest_history_date
        current_flatrate = latest_flatrate
        while end_date > current_date:
            current_date    += datetime.timedelta(days=1)
            current_date_str = datetime.datetime.strftime(current_date, '%Y/%m/%d')

            # change by mode
            if sinario_mode == DERIVE_SINARIO_MODE['high']:
                current_flatrate = None
            elif sinario_mode == DERIVE_SINARIO_MODE['low']:
                current_flatrate = None
            elif sinario_mode == DERIVE_SINARIO_MODE['maintain']:
                current_flatrate = current_flatrate
            else:
                current_flatrate = self.calc_flatrate(current_flatrate)
            self.predicted_data = np.append(self.predicted_data, np.array([(current_date_str, current_flatrate)], dtype=dt))
            
        return

    # generate predicted sinario
    def generate_significant_flat_rate(self, sinario_mode, significant_flat_rate=None, predict_years=DEFAULT_PREDICT_YEARS):
        # default predict_years is 15 years [180 months]
        self.predict_years  = predict_years

        # predicted data type
        dt   = np.dtype({'names': ('date', 'fr'),
                         'formats': ('S10' , np.float)})
        self.predicted_data = np.array([], dtype=dt)

        # latest date from history_data
        latest_history_date_str, latest_flatrate = self.history_data[-1]
        latest_history_date                      = datetime.datetime.strptime(latest_history_date_str, '%Y/%m/%d')

        end_date         = add_year(latest_history_date, predict_years)
        current_date     = latest_history_date
        current_flatrate = latest_flatrate
        while end_date > current_date:
            current_date    += datetime.timedelta(days=1)
            current_date_str = datetime.datetime.strftime(current_date, '%Y/%m/%d')

            # change by mode
            if sinario_mode == 'medium':
                current_flatrate = current_flatrate
            else:
                current_flatrate = significant_flat_rate
            self.predicted_data = np.append(self.predicted_data, np.array([(current_date_str, current_flatrate)], dtype=dt))
        return


    def calc_flatrate(self, current_flatrate):
        return self.u * current_flatrate if prob(self.p) else self.d * current_flatrate
            
    # calc new and sigma from history data
    def calc_params_from_history(self):
        index   = 0
        delta_t = 1.0 / 12
        values  = np.array([])
        for date, flat_rate in self.history_data:
            if index == 0:
                # initialize the rate
                s_0 = flat_rate
            else:
                s_t      = flat_rate
                base_val = math.log(s_t / s_0)
                values   = np.append(values, base_val)
                # update the price
                s_0      = flat_rate
            index += 1

        # substitute inf to nan in values
        values     = inf_to_nan_in_array(values)
        self.neu   = np.nanmean(values)
        self.sigma = np.nanstd(values)
        self.u     = np.exp(self.sigma * np.sqrt(delta_t))
        self.d     = np.exp(self.sigma * (-1) * np.sqrt(delta_t))
        self.p     = 0.5 + 0.5 * (self.neu / self.sigma) * np.sqrt(delta_t)
        return

    # multiple oil price drawing part    
    def draw_multiple_flat_rates(self):
        draw_data = np.array([])
        title     = "flat rate multiple scenarios".title()
        graphInitializer(title,
                         self.default_xlabel,
                         self.default_ylabel)
        draw_data = [ [datetime.datetime.strptime(data['date'], '%Y/%m/%d'), data['fr']] for data in self.history_data]
        draw_data = np.array(sorted(draw_data, key= lambda x : x[0]))
        xlim_date = draw_data.transpose()[0].min()
        plt.plot(draw_data.transpose()[0],
                 draw_data.transpose()[1],
                 color='#9370DB', lw=5, markersize=0, marker='o')
        colors = ['b', 'r', 'k', 'c', 'g', 'm', 'y', 'orange', 'aqua', 'brown']
        plt.axvline(x=datetime.datetime.strptime(self.history_data[-1]['date'], '%Y/%m/%d'), color='k', linewidth=4, linestyle='--')
        sinario_log = {}
        for index in range(10):
            sinario_mode = DERIVE_SINARIO_MODE['binomial']
            # fix the random seed #
            np.random.seed(index)
            self.generate_flat_rate(sinario_mode)
            draw_data   = [ [datetime.datetime.strptime(data['date'], '%Y/%m/%d'), data['fr']] for data in self.predicted_data]
            draw_data   = np.array(sorted(draw_data, key= lambda x : x[0]))
            sinario_log[index] = draw_data
            plt.plot(draw_data.transpose()[0],
                     draw_data.transpose()[1],
                     color=colors[-index], lw=5, markersize=0, marker='o')
            plt.xlim([xlim_date, draw_data.transpose()[0].max()])
        output_file_path = "%s/graphs/%s.png" % (RESULTSDIR, title)
        plt.savefig(output_file_path)

        # display variables
        print "%10s: %10lf" % ('neu', self.neu)
        print "%10s: %10lf" % ('sigma', self.neu)
        print "%10s: %10lf" % ('u', self.u)
        print "%10s: %10lf" % ('d', self.d)
        print "%10s: %10lf" % ('p', self.p)
            
        return    
