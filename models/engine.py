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
    def __init__(self, engine_id=None):
        # read designated engine infomation
        if not engine_id is None:
            self.engine_id      = engine_id
            self.engine_data    = get_engine_with_id(engine_id)

    def showinfo(self):
        print '----------------------'
        print "  engine-%d infomation" % (self.engine_num)
        print '----------------------'

    def get_engine(self, sinario, world_scale):
        print 'get_engine'
        
    def get_engine_with_id(self, engine_id):
        print 'get_engine_with_id'
