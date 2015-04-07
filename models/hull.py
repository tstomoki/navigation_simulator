# import common modules #
import sys
import pdb
# import common modules #

# import own modules #
sys.path.append('../public')
from my_modules import *
# import own modules #

class Hull:
    def __init__(self, hull_num):
        # read designated hull infomation
        self.hull_num = hull_num

    def showinfo(self):
        print '----------------------'
        print "  hull-%d infomation" % (self.hull_num)
        print '----------------------'
