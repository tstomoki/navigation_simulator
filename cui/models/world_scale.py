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

class WorldScale:
    def __init__(self, history_data=None, neu=None, sigma=None, u=None, d=None, p=None, alpha=None, beta=None):
        self.default_xlabel = "date".title()
        self.default_ylabel = "world scale".title()
        self.history_data   = load_world_scale_history_data() if history_data is None else history_data
        # initialize parameters
        if (neu is None or sigma is None or u is None or d is None or p is None or alpha is None or beta is None):
            self.calc_params_from_history()
        else:
            self.neu, self.sigma, self.u, self.d, self.p, self.alpha, self.beta = neu, sigma, u, d, p, alpha, beta

    def show_history_data(self):
        print '----------------------'
        print '- - history data - - '
        print '----------------------'

        for data in self.history_data:
            print "%10s : %10lf" % (data['date'], data['ws'])

    def draw_history_data(self):
        title = "world scale history data".title()
        graphInitializer("history data",
                         self.default_xlabel,
                         self.default_ylabel)

        draw_data = [ [datetime.datetime.strptime(data['date'], '%Y/%m/%d'), data['ws']] for data in self.history_data]
        draw_data = np.array(sorted(draw_data, key= lambda x : x[0]))
        
        plt.plot(draw_data.transpose()[0],
                 draw_data.transpose()[1],
                 color='#9370DB', lw=5, markersize=0, marker='o')
        plt.xlim([draw_data.transpose()[0].min(), draw_data.transpose()[0].max()])

        output_file_path = RESULTSDIR + title + '.png'
        plt.savefig(output_file_path)

    # generate predicted sinario
    def generate_sinario(self, sinario_mode, predict_years=DEFAULT_PREDICT_YEARS):
        # default predict_years is 15 years [180 months]
        self.predict_years  = predict_years

        # predicted data type
        dt   = np.dtype({'names': ('date', 'ws'),
                         'formats': ('S10' , np.float)})
        self.predicted_data = np.array([], dtype=dt)
        
        predict_months_num = self.predict_years * 12

        # latest date from history_data
        latest_history_date_str, latest_ws = self.history_data[-1]
        latest_history_date                = datetime.datetime.strptime(latest_history_date_str, '%Y/%m/%d')

        current_date = latest_history_date
        current_ws   = latest_ws
        for predict_month_num in range(predict_months_num):
            current_date     = add_month(current_date)
            current_date_str = datetime.datetime.strftime(current_date, '%Y/%m/%d')

            # change ws by mode
            if sinario_mode == DERIVE_SINARIO_MODE['high']:
                current_ws = None
            elif sinario_mode == DERIVE_SINARIO_MODE['low']:
                current_ws = None
            elif sinario_mode == DERIVE_SINARIO_MODE['maintain']:
                current_ws = current_ws
            else:
                current_ws = self.calc_ws(current_ws)

            # change ws by mode
            self.predicted_data = np.append(self.predicted_data, np.array([(current_date_str, round(current_ws, 4))], dtype=dt))
            
        return

    def calc_ws(self, current_ws):
        return self.u * current_ws if prob(self.p * 100) else self.d * current_ws
    
    # calc new and sigma from history data
    def calc_params_from_history(self):
        index   = 0
        delta_t = 1.0 / 12
        values  = np.array([])
        for date, oil_price in self.history_data:
            if index == 0:
                # initialize the price
                s_0 = oil_price
                next
            else:
                s_t      = oil_price
                base_val = math.log(s_t / s_0)
                values   = np.append(values, base_val)
                # update the price
                s_0      = oil_price
            index += 1

        # substitute inf to nan in values
        values = inf_to_nan_in_array(values)


        #[WIP] calc alpha and beta
        alpha = 0.1932
        beta  = 6.713
        
        self.neu    = np.nanmean(values)
        self.sigma  = np.nanstd(values)
        self.u      = np.exp(self.sigma * np.sqrt(delta_t))
        self.d      = np.exp(self.sigma * (-1) * np.sqrt(delta_t))
        self.p      = 0.5 + 0.5 * (self.neu / self.sigma) * np.sqrt(delta_t)
        self.alpha, self.beta = alpha, beta

        return

    # set flat_rate [%]
    def set_flat_rate(self, flat_rate):
        self.flat_rate = flat_rate
        return
    
    # flat_rate [%]
    def calc_fare(self, oil_price, flat_rate):
        return (self.alpha * oil_price + self.beta) * (flat_rate / 100.0)
