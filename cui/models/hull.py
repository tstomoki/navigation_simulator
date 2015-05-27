# import common modules #
import sys
import pdb
from types import *
# import common modules #

# import own modules #
sys.path.append('../public')
from my_modules import *
from constants  import *
# import own modules #

# hull data type #
# dt = np.dtype({'names'  : ('id'  , 'Loa'   , 'Lpp'   , 'Disp'  , 'DWT'   , 'Bmld'  , 'Dmld'  , 'draft_full', 'draft_ballast', 'Cb'    , 'S'     , 'ehp0_ballast', 'ehp1_ballast', 'ehp2_ballast', 'ehp3_ballast', 'ehp4_ballast', 'ehp0_full', 'ehp1_full', 'ehp2_full', 'ehp3_full', 'ehp4_full'), #
#                'formats': (np.int16, np.float, np.float, np.float, np.float, np.float, np.float, np.float    ,  np.float      , np.float, np.float, np.float      , np.float      , np.float      , np.float      , np.float      , np.float   , np.float   , np.float   , np.float   , np.float)}) #


class Hull:
    def __init__(self, hull_list, hull_id):
        # read designated hull infomation
        if not hull_list is None:
            self.hull_id   = hull_id
            self.hull_data = self.get_hull_with_id(hull_list)

    # display base data
    def display_variables(self):
        for variable_key in self.__dict__.keys():
            instance_variable_key = "self.%s" % (variable_key)
            instance_variable     = eval(instance_variable_key)
            if isinstance(instance_variable, NoneType):
                print "%25s: %20s" % (instance_variable_key, 'NoneType')
            elif isinstance(instance_variable, np.ndarray):
                print "%25s: %20s" % (instance_variable_key, 'Numpy with length (%d)' % (len(instance_variable)))                
            elif isinstance(instance_variable, DictType):
                key_str = ', '.join([_k for _k in instance_variable.keys()])
                print "%25s: %20s" % (instance_variable_key, 'DictType with keys % 10s' % (key_str))
            else:
                print "%25s: %20s" % (instance_variable_key, str(instance_variable))                
        return
            
    def showinfo(self):
        print '----------------------'
        print "  hull-%d infomation" % (self.hull_num)
        print '----------------------'

    def get_hull(self, sinario, world_scale):
        print 'get_hull'
        
    def get_hull_with_id(self, hull_list):
        self.base_data = hull_list[np.where(hull_list['id']==self.hull_id)][0]
        return

    def calc_EHP(self, v_knot, load_condition):
        ret_ehp = 0
        v_ms    = knot2ms(v_knot)
        # EHP is quartic equation #
        for index in range(4):
            variable_name = "self.base_data['ehp%d_%s']" % (index, load_condition)
            ret_ehp      += eval(variable_name) * math.pow(v_ms, index)
        return ret_ehp
