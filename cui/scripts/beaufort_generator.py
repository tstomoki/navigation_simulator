#!/usr/bin/python
import matplotlib
# server configuration #
import getpass
current_user = getpass.getuser()
if current_user == 'tsaito':
    matplotlib.use('Agg')
# server configuration #
import pylab
import sys
import numpy as np
import scipy
import pandas as pd
from pdb import *
from scipy.stats import beta
sys.path.append('../public')
from my_modules  import *

'''
s = np.random.poisson(14, 10000)
count, bins, ignored = matplotlib.pyplot.hist(s, 8, normed=True)
matplotlib.pyplot.show()
sys.exit()
'''

def run():
    args = {'Calm': 4, 'Rough': 7}
    designated_kinds(args)
    all_kinds()

def designated_kinds(args):
    output_dir_path = "../results/beauforts"
    initializeDirHierarchy(output_dir_path)
    output_file_path = "%s/alpha_%d_%d.png" % (output_dir_path, args['Calm'], args['Rough'])
    title  = "incidence rate of Beaufort at designated points".title()
    x_label   = "beaufort scale".upper()
    y_label   = "probability".upper()
    graphInitializer(title,
                     x_label,
                     y_label)
    
    result = {}
    color = 'r'
    for label, alpha in args.items():
        beta           = 10 - alpha
        x              = [ 0.01*i for i in range(100)]
        beta_func      = [scipy.stats.beta.pdf(xi, alpha, beta) / 2.0 for xi in x]
        separated_beta = separate_list(beta_func,7)
        bfs            = [[index, sum(_d) * 0.01] for index, _d in enumerate(separated_beta)]
        bfs            = np.array(bfs)
        x_data         = bfs.transpose()[0]
        y_data         = bfs.transpose()[1]
        matplotlib.pyplot.bar(x_data, y_data, label=label, color=color, alpha=0.4)
        color = 'b'
    xticks = ["BF%d" % (_x) for _x in x_data ]    
    matplotlib.pyplot.xticks(x_data+0.5, xticks)
    matplotlib.pyplot.ylim(0, 0.20)
    plt.legend(shadow=True)
    plt.legend(loc='upper right')    
    matplotlib.pyplot.savefig(output_file_path)
    matplotlib.pyplot.clf()    
    
def all_kinds():
    alpha_a  = np.arange(1,10, 0.5)
    output_dir_path = "../results/beauforts"
    initializeDirHierarchy(output_dir_path)
    result = {}
    for alpha in alpha_a:
        beta = 10 - alpha
        x                    = [ 0.01*i for i in range(100)]
        beta_func            = [scipy.stats.beta.pdf(xi, alpha, beta) / 2.0 for xi in x]
        separated_beta       = separate_list(beta_func,7)
        bfs    = [[index, sum(_d) * 0.01] for index, _d in enumerate(separated_beta)]
        bfs    = np.array(bfs)
        x_data = bfs.transpose()[0]
        y_data = bfs.transpose()[1]
        title  = "incidence rate of Beaufort".title()
        output_file_path = "%s/alpha_%1.1lf_beta_%1.1lf.png" % (output_dir_path, alpha, beta)
        xticks = ["BF%d" % (_x) for _x in x_data ]
        x_label   = "beaufort".upper()
        y_label   = "probability".upper()
        matplotlib.pyplot.xlabel(x_label, fontweight="bold")
        matplotlib.pyplot.ylabel(y_label, fontweight="bold")
        matplotlib.pyplot.xticks(x_data+0.5, xticks)
        matplotlib.pyplot.bar(x_data, y_data)
        matplotlib.pyplot.ylim(0, 0.40)
        matplotlib.pyplot.savefig(output_file_path)
        matplotlib.pyplot.clf()
        result_key = "a_%1.1lf_b_%1.1lf" % (alpha, beta)
        result[result_key] = {int(_d[0]): _d[1]for _d in bfs}

    output_file_path = "%s/result.json" % (output_dir_path)
    write_file_as_json(result, output_file_path)
   
if __name__ == '__main__':
    run()
