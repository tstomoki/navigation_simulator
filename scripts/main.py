# import common modules #
import sys
import pdb
# import common modules #

# import own modules #
sys.path.append('../public')
import my_modules
# import own modules #

# import models #
sys.path.append('../models')
from hull      import Hull
from sinario   import Sinario
from engine    import Engine
from propeller import Propeller
# import models #


hull = Hull(1)
hull.showinfo()

engine = Engine(1)
engine.showinfo()
