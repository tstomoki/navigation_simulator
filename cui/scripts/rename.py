# import common modules #
import sys
import pdb
from optparse import OptionParser
# import common modules #
# import own modules #
sys.path.append('../public')
from my_modules import *
from constants  import *
# import own modules #

def rename_files(dir_path):
    if not os.path.exists(dir_path):
        print "abort: there is no such directory, %s" % (dir_path)
        sys.exit()
    files = os.listdir(dir_path)
    for target_filename in files:
        try:
            target_filepath  = "%s/%s" % (dir_path, target_filename)
            combination_key, = re.compile(r'H\[.\](.+)\.json').search(target_filename).groups()
            renamed_filepath = "%s/H1%s.json" % (dir_path, combination_key)
            os.rename(target_filepath, renamed_filepath)
        except:
            continue
    return

def run(options):
    # get option variables #
    target_dir_path = options.target_dir
    rename_files(target_dir_path)
    return

# authorize exeucation as main script
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="target_dir",
                      help="target directory path", default=None)
    (options, args) = parser.parse_args()
    run(options)
