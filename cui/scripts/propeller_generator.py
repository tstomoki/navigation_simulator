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
    start_time = time.clock()
    output_dir_path = "%s/results" % (COMPONENT_PATH)
    initializeDirHierarchy(output_dir_path)
    output_json_path = "%s/result.json" % (output_dir_path)
    # load KT, KQ list
    kt_list, kq_list = load_k_list(COMPONENT_PATH)

    # parameters
    P_D_a, EAR_a, D_a, Z_a = generate_parameters()

    # load if file exists
    exist_data = {}
    if os.path.exists(output_json_path):
        exist_data = load_json_file(output_json_path)
        
    # J (0.05, 1.4)
    J_a = [round(_v, 3) for _v in np.linspace(0.05, 1.4)]
    
    propeller_id = 0
    result = {}
    for Z in Z_a:
        for P_D in P_D_a:
            for EAR in EAR_a:
                for D in D_a:
                    # load characteristics
                    if exist_data.has_key(str(propeller_id)):
                        result[propeller_id] = clean_load(exist_data[str(propeller_id)], 'str')
                        result[propeller_id]['coef'] = clean_load(result[propeller_id]['coef'], 'float')
                    else:
                        approx_coef, coef = calc_ks(kt_list, kq_list, P_D, EAR, D, Z, J_a)
                        result[propeller_id] = {'P_D': P_D,
                                                'EAR': EAR,
                                                'D': D,
                                                'Z': Z,
                                                'coef': coef,
                                                'approx_coef': approx_coef}
                    propeller_id += 1
    # output json
    write_file_as_json(result, output_json_path)
    # draw graph
    draw_k_comparison(result, J_a, output_dir_path)
    draw_propeller_kinds(result, output_dir_path)
    draw_propeller_characteristics(result, J_a, output_dir_path)                    
    seconds = convert_second(time.clock() - start_time)
    print_with_notice("Program finished (takes %10s seconds for process)" % (seconds))

def load_k_list(dir_path):
    kt_list_path = "%s/KT_list.csv" % (dir_path)
    kq_list_path = "%s/KQ_list.csv" % (dir_path)        
    if not os.path.exists(kt_list_path):
        print "ERROR: %s does not exist" % (kt_list_path)
        raise
    if not os.path.exists(kq_list_path):
        print "ERROR: %s does not exist" % (kq_list_path)
        raise    
        
    # dtype
    dt = np.dtype({'names'  : ('id', 'coef', 's', 't', 'u', 'v'),
                   'formats': (np.int16, np.float, np.int16, np.int16, np.int16, np.int16) })
    kt_list = np.genfromtxt(kt_list_path,
                            delimiter=',',
                            dtype=dt,
                            skiprows=1)
    kq_list = np.genfromtxt(kq_list_path,
                            delimiter=',',
                            dtype=dt,
                            skiprows=1)    
    return kt_list, kq_list

def generate_parameters():
    # Pitch-Diamter Ratio (0.880 ~ 1.020, 0.014)
    P_D_a = np.linspace(0.880, 1.020, 10)
    # Blade Area Ratio (0.540 ~ 0.620, 0.008)
    EAR_a = np.linspace(0.540, 0.620, 11)
    # Propeller Diameter (8.8 ~ 10.0, 0.2)
    D_a   = np.linspace(8.8, 10.0, 7)
    # Propeller Blade Num (4, 5)
    Z_a   = np.linspace(4, 5, 2)
    
    return P_D_a, EAR_a, D_a, Z_a

def calc_ks(kt_list, kq_list, P_D, EAR, D, Z, J_a):
    approx_coef = {}
    coef = {}

    for J in J_a:
        coef[J] = {}
        coef[J]['KT'], coef[J]['KQ'] = calc_k_from_list(kt_list, kq_list, P_D, EAR, D, Z, J)

    # approx coeff #
    ## KQ
    kq_coef =  [ [J, coef[J]['KQ']] for J in J_a]
    approx_coef['kq'] = calc_approx_coef(kq_coef)
    
    ## KT
    kt_coef =  [ [J, coef[J]['KT']] for J in J_a]
    approx_coef['kt']  = calc_approx_coef(kt_coef)
    return approx_coef, coef

def calc_k_from_list(kt_list, kq_list, P_D, EAR, D, Z, J):
    s_a = np.unique(kt_list['s'])
    t_a = np.unique(kt_list['t'])
    u_a = np.unique(kt_list['u'])
    v_a = np.unique(kt_list['v'])

    kt = 0
    kq = 0
    for s in s_a:
        for t in t_a:
            for u in u_a:
                for v in v_a:
                    kt_des_index = np.where( (kt_list['s'] == s)
                                             & (kt_list['t'] == t)
                                             & (kt_list['u'] == u)
                                             & (kt_list['v'] == v))
                    kq_des_index = np.where( (kq_list['s'] == s)
                                             & (kq_list['t'] == t)
                                             & (kq_list['u'] == u)
                                             & (kq_list['v'] == v))
                    kt_des = kt_list[kt_des_index]
                    kq_des = kq_list[kq_des_index]

                    # calc coef
                    multi_element = math.pow(J, s) * math.pow(P_D, t) * math.pow(EAR, u) * math.pow(Z, v)
                    if not len(kt_des) == 0:
                        kt += kt_des['coef'][0] * multi_element
                    if not len(kq_des) == 0:
                        kq += kq_des['coef'][0] * multi_element
    return kt, kq

def draw_propeller_characteristics(results, J_a, output_dir_path):
    output_dir_path = "%s/characteristics" % (output_dir_path)
    initializeDirHierarchy(output_dir_path)
    title       = "wageningen Propeller B \n ()".title()
    x_label     = "advance coeff [J]".upper()
    y_label     = "ENGINE OUTPUT [kW]"

    for propeller_id, values in results.items():
        J_a = sorted(values['coef'].keys())     
        fig, ax1 = plt.subplots()
        approx_list_kt = [values['approx_coef']['kt']['constant'], values['approx_coef']['kt']['linear'], values['approx_coef']['kt']['square']]
        KT = [ calc_y(_j, approx_list_kt, 2) for _j in J_a]
        lns1 = ax1.plot(J_a, KT, 'b-', label='KT')
        ax1.set_xlabel(x_label)
        # Make the y-axis label and tick labels match the line color.
        ax1.set_ylabel('thrust coeff(KT) efficiency'.upper(), color='k')
        ax1.set_ylim(0, 1.0)
        
        ax2 = ax1.twinx()
        approx_list_kq = [values['approx_coef']['kq']['constant'], values['approx_coef']['kq']['linear'], values['approx_coef']['kq']['square']]
        KQ = [ calc_y(_j, approx_list_kq, 2) for _j in J_a]        
        ax2.plot(J_a, KQ, 'r-', label='KQ')
        ax2.set_ylabel('torque coeff(KQ)'.upper(), color='k')
        ax2.set_ylim(0, 0.07)
        plt.title("ID:%5.3d Blades: %d EAR:%5.3lf PD: %5.3lf D: %5.3lf" % (propeller_id, values['Z'], values['EAR'], values['P_D'], values['D']),       fontweight="bold")
        output_filepath = "%s/propeller_%d.png" % (output_dir_path, propeller_id)
        plt.grid(True)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.legend(shadow=True)
        plt.savefig(output_filepath)
        plt.clf()
        plt.close()
    return 

def clean_load(raw_data, cast_mode):
    ret_data = {}
    for _k in raw_data.keys():
        if isinstance(raw_data[_k], dict):
            ret_data[eval(cast_mode)(_k)] = clean_load(raw_data[_k], 'str')
        else:
            ret_data[eval(cast_mode)(_k)] = raw_data[_k]
        
    return ret_data

def draw_propeller_kinds(result, output_dir_path):
    x_label = "pitch diameter ratio".upper()
    y_label = "blade area ratio".upper()
    z_label = "propeller diameter ".upper() + '[m]'
    # draw scatter
    for blade_num in np.unique([ _v['Z'] for _k, _v in result.items()]):
        output_filepath = "%s/design_kinds_Z%d.png" % (output_dir_path, blade_num)
        draw_data       = []
        for propeller_id, values in result.items():
            if not values['Z'] == blade_num:
                continue
            draw_data.append([ values['P_D'], values['EAR'], values['D'] ])
        draw_data = np.array(draw_data)
        xlist = draw_data.transpose()[0]
        ylist = draw_data.transpose()[1]
        zlist = draw_data.transpose()[2]
        draw_3d_scatter(xlist, ylist, zlist, x_label, y_label, z_label, [], [], None, None, 0.4)
        plt.savefig(output_filepath)
        plt.clf()
        plt.close()    
    return

def draw_k_comparison(result, J_a, output_dir_path):
    x_label         = "propeller id".upper()
    y_label         = "advance coeff [J]".upper()
    z_label         = "thrust coeff(KT)".upper()
    colors          = ['r', 'b', 'k', 'g', 'y', 'orange']
    blade_nums      = np.unique([ _v['Z'] for _k, _v in result.items()])
    output_filepath = "%s/k_comparison.png" % (output_dir_path)
    # draw scatter
    fig = plt.figure()
    ax  = Axes3D(fig)    
    for blade_num in blade_nums:
        label_flag = False
        for propeller_id, values in result.items():
            draw_data       = []
            if not values['Z'] == blade_num:
                continue
            for J in J_a:
                draw_data.append([ propeller_id, J, values['coef'][J]['KT'] ])
            draw_data = np.array(draw_data)                
            X         = np.array(draw_data.transpose()[0])
            Y         = np.array(draw_data.transpose()[1])
            Z         = np.array(draw_data.transpose()[2])
            #ax.scatter3D(np.ravel(X),np.ravel(Y),np.ravel(Z))
            label     = "blade %d" % (blade_num)
            if not label_flag:
                label_flag = True
                ax.plot(np.ravel(X),np.ravel(Y),np.ravel(Z), color=colors[int(blade_num - min(blade_nums))], label=label)
            else:
                ax.plot(np.ravel(X),np.ravel(Y),np.ravel(Z), color=colors[int(blade_num - min(blade_nums))])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.legend(loc='upper right')
    plt.legend(shadow=True)    
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()            
    return

def calc_approx_coef(data):
    ret_coef = {}
    data     = np.array(data)
    M        = 2
    xlist    = data.transpose()[0]
    ylist    = data.transpose()[1]
    ret_coef['constant'], ret_coef['linear'], ret_coef['square'] = estimate(xlist, ylist, M)
    return ret_coef

# authorize exeucation as main script
if __name__ == '__main__':
    parser = OptionParser()
    (options, args) = parser.parse_args()
    run(options)    
