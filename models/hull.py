# import common modules #
import sys
import pdb
# import common modules #

# import own modules #
sys.path.append('../public')
from my_modules import *
from constants  import *
# import own modules #

class Hull:
    def __init__(self, hull_id=None, hull_list=None):
        # read designated hull infomation
        if not hull_id is None:
            self.hull_id   = hull_id
            self.hull_data = self.get_hull_with_id(hull_id, hull_list)

    def showinfo(self):
        print '----------------------'
        print "  hull-%d infomation" % (self.hull_num)
        print '----------------------'

    def get_hull(self, sinario, world_scale):
        print 'get_hull'
        
    def get_hull_with_id(self, hull_id, hull_list):
        self.base_data = hull_list[np.where(hull_list['id']==hull_id)]
        return 

