DEFAULT_PREDICT_YEARS = 15

DERIVE_SINARIO_MODE   = {'high': 0,
                         'low': 1,
                         'binomial': 2,
                         'maintain': 3}

RETROFIT_MODE = {'none': 0,
                 'propeller': 1,
                 'propeller_and_engine': 2}


# 150 [dollars/barrel]
HIGH_OIL_PRICE = 150
# 100 [dollars/barrel]
LOW_OIL_PRICE = 80


# load condition [ballast, full]
LOAD_CONDITION = {'ballast': 0,
                  'full'   : 1}
# initial load condition
INITIAL_LOAD_CONDITION = LOAD_CONDITION['ballast']

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
