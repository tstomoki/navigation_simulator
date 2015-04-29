# import common modules #
import sys
import pdb
# import common modules #

# import own modules #
sys.path.append('../public')
from my_modules import *
# import own modules #

class Propeller:
    def __init__(self, propeller_list, propeller_id):
        # read designated propeller infomation
        if not propeller_list is None:
            self.propeller_id   = propeller_id
            self.propeller_data = self.get_propeller_with_id(propeller_list)
            
    def showinfo(self):
        print '----------------------'
        print "  propeller-%d infomation" % (self.propeller_num)
        print '----------------------'

    def get_propeller_with_id(self, propeller_list):
        self.base_data = propeller_list[np.where(propeller_list['id']==self.propeller_id)][0]
        return        
        
    def get_propeller(self, propeller_info):
        self.base_data = propeller_info
        return         
