# import common modules #
import sys
import pdb
import matplotlib.pyplot as plt
# import common modules #

# import own modules #
sys.path.append('../public')
from my_modules import *
# import own modules #

# constants #
RESULTSDIR = '../results/'
# constants #

class Sinario:
    def __init__(self, history_data=None):
        self.default_xlabel = "date".title()
        self.default_ylabel = "oil price".title() + " [$/barrel]"
        self.history_data   = load_history_data() if history_data is None else history_data

    def show_history_data(self):
        print '----------------------'
        print '- - history data - - '
        print '----------------------'

        for data in self.history_data:
            print "%10s : %10lf" % (data['date'], data['price'])

    def draw_history_data(self):
        title = "history data".title()
        graphInitializer("history data",
                         self.default_xlabel,
                         self.default_ylabel)

        draw_data = [ [datetime.datetime.strptime(data['date'], '%Y/%m/%d'), data['price']] for data in self.history_data]
        draw_data = np.array(sorted(draw_data, key= lambda x : x[0]))
        
        plt.plot(draw_data.transpose()[0],
                 draw_data.transpose()[1],
                 color='#9370DB', lw=5, markersize=0, marker='o')

        output_file_path = RESULTSDIR + title + '.png'
        plt.savefig(output_file_path)
