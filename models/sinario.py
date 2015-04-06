# import common modules #
import sys
import pdb
# import common modules #

# import own modules #
sys.path.append('../public')
import my_modules
# import own modules #

class Sinario:
    def __init__(self, sinario_num):
        # read designated sinario infomation
        self.sinario_num = sinario_num

    def showinfo(self):
        print '----------------------'
        print "  sinario-%d infomation" % (self.sinario_num)
        print '----------------------'
