# import common modules #
import sys
import pdb
# import common modules #

# import own modules #
sys.path.append('../public')
import my_modules
# import own modules #

class Propeller:
    def __init__(self, propeller_num):
        # read designated propeller infomation
        self.propeller_num = propeller_num

    def showinfo(self):
        print '----------------------'
        print "  propeller-%d infomation" % (self.propeller_num)
        print '----------------------'
