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
    def __init__(self, engine_info):
        # read designated engine infomation
        if not engine_info is None:
            self.engine_id   = engine_info['id']
            self.engine_data = self.get_engine(engine_info)        

    def showinfo(self):
        print '----------------------'
        print "  engine-%d infomation" % (self.engine_num)
        print '----------------------'

    def get_engine(self, sinario, world_scale):
        print 'get_engine'

    def get_engine(self, engine_info):
        self.base_data = engine_info
        return 
        
    def calc_sfoc(self, bhp):
        load = self.calc_load(bhp)
        return sfoc

    def calc_load(self, bhp):
        pdb.set_trace()
