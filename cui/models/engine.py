# import common modules #
import sys
import pdb
# import common modules #

# import own modules #
sys.path.append('../public')
from my_modules import *
from constants  import *
# import own modules #

class Engine:
    def __init__(self, engine_list, engine_id):
        # read designated engine infomation
        if not engine_list is None:
            self.engine_id   = engine_id
            self.engine_data = self.get_engine_with_id(engine_list)        

    def showinfo(self):
        print '----------------------'
        print "  engine-%d infomation" % (self.engine_num)
        print '----------------------'

    def get_engine_with_id(self, engine_list):
        self.base_data = engine_list[np.where(engine_list['id']==self.engine_id)][0]
        return
    
    def get_engine(self, engine_info):
        self.base_data = engine_info
        return 
        
    def calc_sfoc(self, bhp):
        load = self.calc_load(bhp)
        return self.base_data['sfoc0'] + self.base_data['sfoc1'] * load + self.base_data['sfoc2'] * math.pow(load, 2)

    # return bhp / max_load  
    def calc_load(self, bhp):
        return float(bhp) / self.base_data['max_load']
