# import common modules #
import sys
import pdb
# import common modules #

# import own modules #
sys.path.append('../public')
import my_modules
# import own modules #

class Engine:
    def __init__(self, engine_num):
        # read designated engine infomation
        self.engine_num = engine_num

    def showinfo(self):
        print '----------------------'
        print "  engine-%d infomation" % (self.engine_num)
        print '----------------------'
