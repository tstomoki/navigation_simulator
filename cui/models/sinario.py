# import common modules #
import sys
import math
import pdb
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

class Sinario:
    def __init__(self, history_data=None, neu=None, sigma=None, u=None, d=None, p=None):
        self.default_xlabel   = "date".title()
        self.default_ylabel   = "oil price".title() + " [USD]"
        if history_data is None:
            self.history_data = load_monthly_history_data()
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
        title = "oil price history data".title()
        graphInitializer(title,
                         self.default_xlabel,
                         self.default_ylabel)

        draw_data = [ [datetime.datetime.strptime(data['date'], '%Y/%m/%d'), data['price']] for data in self.history_data]
        draw_data = np.array(sorted(draw_data, key= lambda x : x[0]))
        
        plt.plot(draw_data.transpose()[0],
                 draw_data.transpose()[1],
                 color='#9370DB', lw=5, markersize=0, marker='o')
        plt.xlim([draw_data.transpose()[0].min(), draw_data.transpose()[0].max()])                

        output_file_path = RESULTSDIR + title + '.png'
        plt.savefig(output_file_path)

    def draw_predicted_data(self):
        title = "oil price predicted data".title()
        graphInitializer("history data",
                         self.default_xlabel,
                         self.default_ylabel)

        draw_data = [ [datetime.datetime.strptime(data['date'], '%Y/%m/%d'), data['price']] for data in self.predicted_data]
        draw_data = np.array(sorted(draw_data, key= lambda x : x[0]))

        plt.plot(draw_data.transpose()[0],
                 draw_data.transpose()[1],
                 color='#9370DB', lw=5, markersize=0, marker='o')
        plt.xlim([draw_data.transpose()[0].min(), draw_data.transpose()[0].max()])        

        output_file_path = RESULTSDIR + title + '.png'
        plt.savefig(output_file_path)

    def draw_generated_data(self):
        title = "oil price generated data".title()
        graphInitializer(title,
                         self.default_xlabel,
                         self.default_ylabel)

        plt.axvline(x=datetime.datetime.strptime(self.history_data[-1]['date'], '%Y/%m/%d'), color='k', linewidth=4, linestyle='--')
        draw_data = [ [datetime.datetime.strptime(data['date'], '%Y/%m/%d'), data['price']] for data in np.r_[self.history_data, self.predicted_data]]
        draw_data = np.array(sorted(draw_data, key= lambda x : x[0]))

        plt.plot(draw_data.transpose()[0],
                 draw_data.transpose()[1],
                 color='#9370DB', lw=5, markersize=0, marker='o')
        plt.xlim([draw_data.transpose()[0].min(), draw_data.transpose()[0].max()])        

        output_file_path = "%s/graphs/%s.png" % (RESULTSDIR, title)
        plt.savefig(output_file_path)
        return

    # generate predicted sinario
    def generate_sinario(self, sinario_mode, predict_years=DEFAULT_PREDICT_YEARS):
        # default predict_years is 15 years [180 months]
        self.predict_years  = predict_years

        # predicted data type
        dt   = np.dtype({'names': ('date', 'price'),
                         'formats': ('S10' , np.float)})
        self.predicted_data = np.array([], dtype=dt)
        
        predict_months_num = int(self.predict_years * 12)

        # latest date from history_data
        latest_history_date_str, latest_oilprice = self.history_data[-1]
        latest_history_date                      = datetime.datetime.strptime(latest_history_date_str, '%Y/%m/%d')

        current_date  = latest_history_date
        current_oilprice = latest_oilprice
        for predict_month_num in range(predict_months_num):
            current_date        = add_month(current_date)
            current_date_str    = datetime.datetime.strftime(current_date, '%Y/%m/%d')

            # change oil_price by mode
            if sinario_mode == DERIVE_SINARIO_MODE['high']:
                current_oilprice = HIGH_OIL_PRICE
            elif sinario_mode == DERIVE_SINARIO_MODE['low']:
                current_oilprice = LOW_OIL_PRICE
            elif sinario_mode == DERIVE_SINARIO_MODE['maintain']:
                current_oilprice = current_oilprice
            else:
                current_oilprice    = self.calc_oilprice(current_oilprice)
            # change oil_price by mode
                
            self.predicted_data = np.append(self.predicted_data, np.array([(current_date_str, current_oilprice)], dtype=dt))
            
        return

    # generate predicted sinario
    def generate_significant_sinario(self, oilprice_mode, significant_oilprice=None, predict_years=DEFAULT_PREDICT_YEARS):
        # default predict_years is 15 years [180 months]
        self.predict_years  = predict_years

        # predicted data type
        dt   = np.dtype({'names': ('date', 'price'),
                         'formats': ('S10' , np.float)})
        self.predicted_data = np.array([], dtype=dt)
        
        predict_months_num = int(self.predict_years * 12)

        # latest date from history_data
        latest_history_date_str, latest_oilprice = self.history_data[-1]
        latest_history_date                      = datetime.datetime.strptime(latest_history_date_str, '%Y/%m/%d')

        current_date     = latest_history_date
        current_oilprice = latest_oilprice
        if isinstance(significant_oilprice, list):
            xlist = np.array([0, predict_months_num])
            tlist = np.array(significant_oilprice)
            wlist = estimate(xlist, tlist, 1)
            significant_oilprice_array = {index: calc_y(index, wlist, 1) for index in range(predict_months_num)}
            
        for predict_month_num in range(predict_months_num):
            current_date        = add_month(current_date)
            current_date_str    = datetime.datetime.strftime(current_date, '%Y/%m/%d')

            # change oil_price by mode
            if oilprice_mode == 'oilprice_medium':
                current_oilprice = current_oilprice
            elif oilprice_mode == 'oilprice_dec' or oilprice_mode == 'oilprice_inc':                
                current_oilprice = round(significant_oilprice_array[predict_month_num], 3)
            else:
                current_oilprice = significant_oilprice

            # change oil_price by mode
            self.predicted_data = np.append(self.predicted_data, np.array([(current_date_str, current_oilprice)], dtype=dt))
            
        return
            
    # calc new and sigma from history data
    def calc_params_from_history(self):
        index   = 0
        delta_t = 1.0 / 12
        values  = np.array([])
        for date, oil_price in self.history_data:
            if index == 0:
                # initialize the price
                s_0 = oil_price
            else:
                s_t      = oil_price
                base_val = math.log(s_t / s_0)
                values   = np.append(values, base_val)
                # update the price
                s_0      = oil_price
            index += 1

        # substitute inf to nan in values
        values = inf_to_nan_in_array(values)
        self.neu    = np.nanmean(values)
        self.sigma  = np.nanstd(values)
        self.u      = np.exp(self.sigma * np.sqrt(delta_t))
        self.d      = np.exp(self.sigma * (-1) * np.sqrt(delta_t))
        self.p      = 0.5 + 0.5 * (self.neu / self.sigma) * np.sqrt(delta_t)
        return

    def calc_oilprice(self, current_oilprice):
        return self.u * current_oilprice if prob(self.p) else self.d * current_oilprice

    # multiple oil price drawing part    
    def draw_multiple_scenarios(self, world_scale=None):
        draw_data = np.array([])
        title     = "oil price multiple scenarios".title()
        graphInitializer(title,
                         self.default_xlabel,
                         self.default_ylabel)
        draw_data = [ [datetime.datetime.strptime(data['date'], '%Y/%m/%d'), data['price']] for data in self.history_data]
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
            self.generate_sinario(sinario_mode)
            draw_data   = [ [datetime.datetime.strptime(data['date'], '%Y/%m/%d'), data['price']] for data in self.predicted_data]
            draw_data   = np.array(sorted(draw_data, key= lambda x : x[0]))
            sinario_log[index] = draw_data
            plt.plot(draw_data.transpose()[0],
                     draw_data.transpose()[1],
                     color=colors[-index], lw=5, markersize=0, marker='o')
            plt.xlim([xlim_date, draw_data.transpose()[0].max()])
        output_file_path = "%s/graphs/%s.png" % (RESULTSDIR, title)
        plt.savefig(output_file_path)

        if not world_scale is None:
            draw_data = np.array([])
            title     = "world scale multiple scenarios".title()
            graphInitializer(title,
                             world_scale.default_xlabel,
                             world_scale.default_ylabel)
            draw_data = [ [datetime.datetime.strptime(data['date'], '%Y/%m/%d'), data['ws']] for data in world_scale.history_data]
            draw_data = np.array(sorted(draw_data, key= lambda x : x[0]))
            xlim_date = draw_data.transpose()[0].min()
            plt.plot(draw_data.transpose()[0],
                     draw_data.transpose()[1],
                     color='#9370DB', lw=5, markersize=0, marker='o')
            colors = ['b', 'r', 'k', 'c', 'g', 'm', 'y', 'orange', 'aqua', 'brown']
            plt.axvline(x=datetime.datetime.strptime(world_scale.history_data[-1]['date'], '%Y/%m/%d'), color='k', linewidth=4, linestyle='--')
            for index in range(10):
                oilprice_array = sinario_log[index]
                draw_data      = np.array([])
                world_scale.generate_sinario_with_oil_corr(sinario_mode, self.history_data[-1], oilprice_array)
                draw_data = [ [datetime.datetime.strptime(data['date'], '%Y-%m-%d'), data['ws']] for data in world_scale.predicted_data]
                draw_data = np.array(sorted(draw_data, key= lambda x : x[0]))                
                plt.plot(draw_data.transpose()[0],
                         draw_data.transpose()[1],                
                         color=colors[-index], lw=5, markersize=0, marker='o')
                plt.xlim([xlim_date, draw_data.transpose()[0].max()])
                output_file_path = "%s/graphs/%s.png" % (RESULTSDIR, title)
            plt.savefig(output_file_path)            
            
        return

    def draw_significant_oilprice_modes(self, sinario_modes):
        draw_data = np.array([])
        title     = "oil price significant scenarios".title()
        colors    = ['r', 'b', 'g','m', 'y', 'orange', 'aqua', 'brown']
        sinario_modes = [ 'oilprice_' + s for s in ['high', 'low', 'middle', 'inc', 'dec']]
        graphInitializer(title,
                         self.default_xlabel,
                         self.default_ylabel)
        # oilprice
        oil_price_history_data           = self.history_data
        sinario                         = Sinario(oil_price_history_data)            
        significant_high_oilprice_index = oil_price_history_data[np.argmax(oil_price_history_data['price'])]
        significant_high_oilprice       = significant_high_oilprice_index['price'] * MULTIPLY_INDEX
        significant_low_oilprice_index  = oil_price_history_data[np.argmin(oil_price_history_data['price'])]
        significant_low_oilprice        = significant_low_oilprice_index['price']
        for index, sinario_mode in enumerate(sinario_modes):
            if sinario_mode == 'oilprice_low':
                # set modes
                significant_oilprice    = significant_low_oilprice
            elif sinario_mode == 'oilprice_high':
                # set modes
                significant_oilprice    = significant_high_oilprice
            elif sinario_mode == 'oilprice_dec':
                # set modes
                significant_oilprice    = [significant_high_oilprice, significant_low_oilprice]
            elif sinario_mode == 'oilprice_inc':
                # set modes
                significant_oilprice    = [significant_low_oilprice, significant_high_oilprice]
            elif sinario_mode == 'oilprice_middle':
                # set modes
                significant_oilprice    = sinario.history_data[-1]['price']
                
            sinario.generate_significant_sinario(sinario_mode, significant_oilprice)
            draw_data   = [ [datetime.datetime.strptime(data['date'], '%Y/%m/%d'), data['price']] for data in sinario.predicted_data]
            draw_data   = np.array(sorted(draw_data, key= lambda x : x[0]))
            if sinario_mode[9:] == 'middle':
                label = 'stage'
            else:
                label = sinario_mode[9:]
            plt.plot(draw_data.transpose()[0],
                     draw_data.transpose()[1],
                     color=colors[index], lw=5, markersize=0, marker='o', label=label)
            xlim_date = draw_data.transpose()[0].min()
            plt.xlim([xlim_date, draw_data.transpose()[0].max()])
        output_file_path = "%s/graphs/%s.png" % (RESULTSDIR, title)
        plt.legend(shadow=True)
        plt.legend(loc='center right')
        plt.savefig(output_file_path)            
        return
