#!/usr/bin/python
import matplotlib
import pylab
import sys
import numpy as np
import scipy
import pandas as pd
from pdb import *
from scipy.stats import beta
sys.path.append('../public')
from my_modules  import *

def separate_list(raw_list, num):
    ret_data = []
    delta = round( len(raw_list) / float(num) )
    index = 0
    while index < len(raw_list):
        index = int(index)
        if (index+delta) > len(raw_list):
            ret_data.append(raw_list[index:])
        else:
            ret_data.append(raw_list[index:int(index+delta)])
        index += delta
    return ret_data

'''
s = np.random.poisson(14, 10000)
count, bins, ignored = matplotlib.pyplot.hist(s, 8, normed=True)
matplotlib.pyplot.show()
sys.exit()
'''
alpha_a  = range(1,10)
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
    output_file_path = "%s/alpha_%d_beta_%d.png" % (output_dir_path, alpha, beta)
    xticks = ["BF%d" % (_x) for _x in x_data ]
    matplotlib.pyplot.xticks(x_data+0.5, xticks)
    matplotlib.pyplot.bar(x_data, y_data)
    matplotlib.pyplot.ylim(0, 0.40)
    matplotlib.pyplot.savefig(output_file_path)
    matplotlib.pyplot.clf()
    result_key = "a_%d_b_%d" % (alpha, beta)
    result[result_key] = {int(_d[0]): _d[1]for _d in bfs}

output_file_path = "%s/result.json" % (output_dir_path)
write_file_as_json(result, output_file_path)
