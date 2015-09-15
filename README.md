# navigation_simulator
navigation simulator based on actual operated VLCC

This is a navigation simulator based on actual operated VLCC to evaluate ship designs

# initialize
## generate beaufort
```
$ cd scripts
$ python beaufort_generator.py
```

## generate propellers
```
$ cd scripts
$ python propeller_generator.py
```

## configure processors num for multiprocessing
# check processor num for CentOS
```
$ cat /proc/cpuinfo | grep processor
```