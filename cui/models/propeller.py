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

    def calc_KT(self, J):
        return self.base_data['KT0'] + self.base_data['KT1'] * J + self.base_data['KT2'] * math.pow(J,2)

    def calc_KQ(self, J):
        return self.base_data['KQ0'] + self.base_data['KQ1'] * J + self.base_data['KQ2'] * math.pow(J,2)
    
    # calc advance constant
    def calc_advance_constant(self, velocity_ms, rps):
        return (WAKE_COEFFICIENT * velocity_ms) / (rps * propeller.base_data['D'])    

    def calc_eta(rps, velocity_ms, KT, KQ):
        return THRUST_COEFFICIENT * ( velocity_ms / (2 * math.pi) ) * (1.0 / (rps * self.base_data['D']) ) * ( (KT) / (KQ) )
