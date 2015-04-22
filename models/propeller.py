# import common modules #
import sys
import pdb
# import common modules #

# import own modules #
sys.path.append('../public')
from my_modules import *
# import own modules #

class Propeller:
    def __init__(self, propeller_info):
        # read designated propeller infomation
        if not propeller_info is None:
            self.propeller_id   = propeller_info['id']
            self.propeller_data = self.get_propeller(propeller_info)            

    def showinfo(self):
        print '----------------------'
        print "  propeller-%d infomation" % (self.propeller_num)
        print '----------------------'

    def get_propeller(self, propeller_info):
        self.base_data = propeller_info
        return         
