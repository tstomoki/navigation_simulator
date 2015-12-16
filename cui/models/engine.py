# import common modules #
import sys
import pdb
# import common modules #

# import own modules #
sys.path.append('../public')
from my_modules import *
from constants  import *
# import own modules #
from types import *

class Engine:
    def __init__(self, engine_list, engine_id):
        # read designated engine infomation
        if not engine_list is None:
            self.engine_id   = engine_id
            self.engine_data = self.get_engine_with_id(engine_list)
        self.rpm_array = np.arange(DEFAULT_RPM_RANGE['from'], self.base_data['N_max']+RPM_RANGE_STRIDE, RPM_RANGE_STRIDE)
        # generate load_combination considering efficiency
        self.modified_bhp_array = self.generate_modified_bhp()

    def showinfo(self):
        print '----------------------'
        print "  engine-%d infomation" % (self.engine_num)
        print '----------------------'

    def get_engine_with_id(self, engine_list):
        self.base_data = engine_list[np.where(engine_list['id']==self.engine_id)][0]
        return
    
    def get_engine(self, engine_info):
        self.base_data = engine_info
        return 
        
    def calc_sfoc(self, bhp):
        y_delta = self.get_delta_from_name()
        load    = self.calc_load(bhp)
        load    = self.engine_derating(load, y_delta)
        ret_val = self.base_data['sfoc0'] + self.base_data['sfoc1'] * load + self.base_data['sfoc2'] * math.pow(load, 2)
        return ret_val

    def engine_derating(self, load, y_delta):
        bias = 0.0
        delta_rate = (100.0 - y_delta*25.0 + bias) / 100.0
        return load * delta_rate

    # return bhp / max_load  
    def calc_load(self, bhp):
        return float(bhp) / self.base_data['max_load']

    # return rpm / N_max
    def calc_relative_engine_speed(self, rpm):
        return float(rpm) / self.base_data['N_max']

    def calc_bhp(self, rpm):
        nearest_rpm      = find_nearest(self.modified_bhp_array['rpm'],rpm)
        index            = np.where(self.modified_bhp_array['rpm']==nearest_rpm)
        designated_array = self.modified_bhp_array[index]
        modified_bhp     = designated_array['modified_bhp'][0]
        return modified_bhp

    def generate_modified_bhp(self):
        output_dir_path         = "%s/engine/engine%s" % (COMBINATIONS_DIR_PATH, self.base_data['name'])
        initializeDirHierarchy(output_dir_path)
        csv_file_path = "%s/engine%s_efficiency.csv" % (output_dir_path, self.base_data['name'])
        dtype  = np.dtype({'names': ('rpm'    , 'relative_engine_speed', 'linear_bhp', 'efficiency', 'modified_bhp'),
                           'formats':(np.float, np.float               , np.float    , np.float    , np.float)})
        # use if the csv file exists        
        if os.path.exists(csv_file_path):
            ret_data = np.genfromtxt(csv_file_path,
                                     delimiter=',',
                                     dtype=dtype,
                                     skiprows=1)
            return ret_data
        
        efficiency_coefficients = estimate(np.array(RELATIVE_ENGINE_EFFICIENCY.keys()),
                                           np.array(RELATIVE_ENGINE_EFFICIENCY.values()),
                                           ENGINE_CURVE_APPROX_DEGREE)
        x_label = "rpm".upper()
        y_label = "efficiency".upper()
        draw_approx_curve(efficiency_coefficients,
                          'efficiency curve', output_dir_path,
                          np.linspace(0,1,100), ENGINE_CURVE_APPROX_DEGREE,
                          x_label, y_label)
        # linear approx #
        xlist = np.array([0,
                          self.calc_relative_engine_speed(self.base_data['sample_rpm0']),
                          self.calc_relative_engine_speed(self.base_data['sample_rpm1'])])
        ylist = np.array([0,
                          self.base_data['sample_bhp0'],
                          self.base_data['sample_bhp1']])
        linear_approx_params = estimate(xlist, ylist, 1)
        x_label = "rpm".upper()
        y_label = "BHP [kW]"
        draw_approx_curve(linear_approx_params,
                          'linear approximation', output_dir_path,
                          np.array([self.calc_relative_engine_speed(x) for x in np.linspace(0,95,100)]), 1,
                          x_label, y_label)
        
        ret_data = np.array([], dtype=dtype)
        for rpm in self.rpm_array:
            relative_engine_speed = self.calc_relative_engine_speed(rpm)
            linear_bhp            = calc_y(relative_engine_speed, linear_approx_params, 1)
            efficiency            = calc_y(relative_engine_speed, efficiency_coefficients, ENGINE_CURVE_APPROX_DEGREE)
            add_elem              = np.array([(rpm,
                                               relative_engine_speed,
                                               linear_bhp,
                                               efficiency,
                                               linear_bhp * efficiency)],
                                             dtype=dtype)
            ret_data              = append_for_np_array(ret_data, add_elem)
        # draw engine rpm combinations #
        self.draw_engine_rpm_combination(ret_data, output_dir_path)
        # write array to csv
        write_array_to_csv(dtype.names, ret_data, csv_file_path)
        return ret_data
        
    def draw_engine_rpm_combination(self, ret_data, output_dir_path):
        # initialize path
        output_file_path = "%s/engine_%s.png" % (output_dir_path, self.base_data['name'])

        x_label = "rpm".upper()
        y_label = "%s %s" % ('bhp'.upper(), '[kW]')
        title   = "BHP and RPM %s of engine %s"  % ("combination".title(),
                                                    self.base_data['name'])
        graphInitializer(title,
                         x_label,
                         y_label)
        plt.title(title)
        x_data    = ret_data['rpm']
        y_data    = ret_data['modified_bhp']
        plt.plot(x_data, y_data)
        plt.savefig(output_file_path)
        plt.close()
        return

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
    
    def consider_efficiency(self, rpm, bhp):
        index            = np.where(self.modified_bhp_array['rpm']==rpm)
        designated_array = self.modified_bhp_array[index]
        return bhp * designated_array['efficiency'][0]

    def get_delta_from_name(self):
        engine_name = self.base_data['name']
        delta       = float(re.compile(r'.+_(\d+)').search(engine_name).groups()[-1])
        ret_delta = delta - BASE_PEAK
        return ret_delta

    def change_sfoc_rate(self):
        change_rate = SFOC_BASE_DEC + self.get_delta_from_name()
        change_rate /= 4000
        return 1.0 + change_rate
