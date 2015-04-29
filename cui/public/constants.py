# dir path from models
RESULT_DIR_PATH    = '../results'
AGNET_LOG_DIR_PATH = "%s/agent_log" % (RESULT_DIR_PATH)

# dir path from scripts
NOHUP_LOG_DIR_PATH    = '../nohup'

DEFAULT_PREDICT_YEARS    = 15

OPERATION_DURATION_YEARS = 15

DERIVE_SINARIO_MODE   = {'high': 0,
                         'low': 1,
                         'binomial': 2,
                         'maintain': 3}

RETROFIT_MODE = {'none': 0,
                 'propeller': 1,
                 'propeller_and_engine': 2}

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
VELOCITY_RANGE = {'from'  : 8.0,
                  'to'    : 18.0,
                  'stride': 0.05}
# default rps range #
RPM_RANGE = {'from'  : 68.0,
             'to'    : 80.0,
             'stride': 0.1}

# thrust coefficient(1-t)
THRUST_COEFFICIENT = 0.8

# wake coefficient(1-w)
WAKE_COEFFICIENT = 0.97

# eta
ETA_S = 0.97

# navigation infomation [mile]
NAVIGATION_DISTANCE = 6590
## for dev
#NAVIGATION_DISTANCE = 1000

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
