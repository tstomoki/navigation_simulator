# import common modules #
import time
import sys
from pdb import *
import matplotlib
# server configuration #
import getpass
current_user = getpass.getuser()
if current_user == 'tsaito':
    matplotlib.use('Agg')
# server configuration #
from optparse import OptionParser
# import common modules #
# import own modules #
sys.path.append('../public')
from my_modules import *
from constants  import *
from cubic_module import *
# import own modules #

def run(options):
    start_time       = time.clock()

    draw_approx_velocity_decrease()
    
    seconds = convert_second(time.clock() - start_time)
    print_with_notice("Program finished (takes %10s seconds for process)" % (seconds))

def draw_approx_velocity_decrease():
    target_path = '../data/components_lists/velocity_decreases.csv'
    if not os.path.exists(target_path):
        print "ERROR: %s does not exist" % (target_path)
        raise
    # dtype
    dt = np.dtype({'names'  : ('wave_height', 'v_dec'),
                   'formats': (np.float, np.float) })
    v_decrease = np.genfromtxt(target_path,
                               delimiter=',',
                               dtype=dt,
                               skiprows=1)
    M        = 2
    cons, linear, square = estimate(v_decrease['wave_height'], v_decrease['v_dec'], M)
    print cons, linear, square

    # draw
    title       = "V decrease".title()
    x_label     = "significant wave height ".upper() + "[m]"
    y_label     = '$\Delta$V [knot]'
    graphInitializer(title, x_label, y_label)
    x = np.linspace(0, 4.0, 100)
    y = [calc_y(_v, [cons, linear, square], M) for _v in x]
    plt.plot(x,y)
    plt.savefig('../data/components_lists/velocity_decreases.png')
    return
    
# authorize exeucation as main script
if __name__ == '__main__':
    parser = OptionParser()
    (options, args) = parser.parse_args()
    run(options)    
