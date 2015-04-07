# import common modules #
import sys
import pdb
# import common modules #

# import own modules #
sys.path.append('../public')
from my_modules import *
# import own modules #

class Sinario:
    def __init__(self, history_data=None):
        self.history_data = load_history_data() if history_data is None else history_data

    def show_history_data(self):
        print '----------------------'
        print '- - history data - - '
        print '----------------------'

        for data in self.history_data:
            print "%10s : %10lf" % (data['date'], data['price'])
