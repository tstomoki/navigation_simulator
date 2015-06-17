# dir path from models
RESULT_DIR_PATH          = '../results'
AGNET_LOG_DIR_PATH       = "%s/agent_log"    % (RESULT_DIR_PATH)
COMBINATIONS_DIR_PATH    = "%s/combinations" % (RESULT_DIR_PATH)
WAVE_DIR_PATH            = "%s/wave"         % (RESULT_DIR_PATH)
CORRELATION_DIR_PATH     = "%s/correlation"  % (RESULT_DIR_PATH)

# dir path from scripts
NOHUP_LOG_DIR_PATH    = '../nohup'

# for multiprocessing
import getpass
current_user = getpass.getuser()
PROC_NUM = 15 if current_user == 'tsaito' else 2
# for multiprocessing

OPERATION_DURATION_YEARS = 15
DEFAULT_PREDICT_YEARS    = OPERATION_DURATION_YEARS

# navigation infomation [mile]
NAVIGATION_DISTANCE = 6590
## for dev
#NAVIGATION_DISTANCE = 1000

DERIVE_SINARIO_MODE   = {'high': 0,
                         'low': 1,
                         'binomial': 2,
                         'maintain': 3}

RETROFIT_MODE = {'none': 0,
                 'propeller': 1,
                 'propeller_and_engine': 2}

RETROFIT_COST = {'propeller': 200000,
                 'engine': 1000000}

# simmulation duration for retrofits
SIMMULATION_DURATION_YEARS_FOR_RETROFITS = 5
SIMMULATION_TIMES_FOR_RETROFITS          = 10
SIMMULATION_RANK_THRESHOLD_FOR_RETROFITS = 10

# days taken to load 
LOAD_DAYS = 2

# dock-in
## dock-in period [years] 
DOCK_IN_PERIOD   = 2
## dock-in duration [month]
DOCK_IN_DURATION = 1

# 150 [dollars/barrel]
HIGH_OIL_PRICE = 150
# 100 [dollars/barrel]
LOW_OIL_PRICE = 80


# load condition [ballast, full]
LOAD_CONDITION = {0: 'ballast',
                  1: 'full'   }
# initial load condition 'ballast'
INITIAL_LOAD_CONDITION = 0

# default velocity range [knot] #
VELOCITY_RANGE_STRIDE  = 0.10
DEFAULT_VELOCITY_RANGE = {'from'  : 5.0,
                          'to'    : 25.0,
                          'stride': VELOCITY_RANGE_STRIDE}
# default rps range #
RPM_RANGE_STRIDE  = 0.5
DEFAULT_RPM_RANGE = {'from'  : 25.0,
                     'to'    : 80.0,
                     'stride': RPM_RANGE_STRIDE}

# minimum array require rate
MINIMUM_ARRAY_REQUIRE_RATE = 3.0

# thrust coefficient(1-t)
THRUST_COEFFICIENT = 0.8

# wake coefficient(1-w)
WAKE_COEFFICIENT = 0.97

# eta
ETA_S = 0.97

# icr
DEFAULT_ICR_RATE = 0.05

## fix value ##
# dry dock maintenance [USD/year] #
DRY_DOCK_MAINTENANCE = 800000
# maintenance [USD/year] #
MAINTENANCE          = 240000
# crew labor cost [USD/year] #
CREW_LABOR_COST      = 2400000 
# Insurance [USD/year] #
INSURANCE            = 240000
## fix value ##

# Port [USD]
PORT_CHARGES    = 100000
PORT_DWELL_DAYS = 2

# Log columns
LOG_COLUMNS = ['ballast', 'full']

# discount rate
DISCOUNT_RATE = 0.05

# model variables
## agent
AGENT_VARIABLES = ['dockin_flag',
                   'velocity_combination',
                   'origin_date',
                   'loading_days',
                   'load_condition',
                   'operation_date_array',
                   'cash_flow',
                   'ballast_trip_days',
                   'round_trip_distance',
                   'retrofit_mode',
                   'retire_date',
                   'log',
                   'total_NPV',
                   'sinario',
                   'loading_flag',
                   'current_date',
                   'hull',
                   'icr',
                   'previous_oilprice',
                   'engine',
                   'latest_dockin_date',
                   'voyage_date',
                   'world_scale',
                   'current_distance',
                   'rpm_array',
                   'total_cash_flow',
                   'total_distance',
                   'propeller',
                   'return_trip_days',
                   'total_elapsed_days',
                   'left_distance_to_port',
                   'current_fare',
                   'oilprice_full',
                   'elapsed_days',
                   'sinario_mode',
                   'oilprice_ballast',
                   'NPV',
                   'velocity_array']

# cull threshold [knot]
CULL_THRESHOLD = 6

# Rank weight
APPR_RANK_WEIGHT = 2
NPV_RANK_WEIGHT  = 1

# default simulate count for searching initial design
DEFAULT_SIMULATE_COUNT = 100
SIMMULATION_DURATION_YEARS_FOR_INITIAL_DESIGN = 5
# narrowed down design ratio for initial design
NARROWED_DOWN_DESIGN_RATIO            = 0.1
NARROWED_DOWN_DURATION_YEARS          = 2
NARROWED_DOWN_DURATION_SIMULATE_COUNT = 10
MINIMUM_NARROWED_DOWN_DESIGN_NUM      = 100

# engine rpm curve approx degree
ENGINE_CURVE_APPROX_DEGREE = 3
RELATIVE_ENGINE_EFFICIENCY = {1.0: 1.0,
                              0.85: 0.7,
                              0.65: 0.35,
                              0.45: 0.2,
                              0:0}
