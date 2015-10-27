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
### check processor num for CentOS
```
$ cat /proc/cpuinfo | grep processor
```

# search optimal ship design

```
$ python main.py 
.
. (takes few days)
.
(after the script finished, aggregate the results by the command)
$ python treat_result.py -A True -P ../results/agent_log/#{timestamp}
```

# Simulation considering retrofit
## search designs for significant market price
```
$ python main.py -T True
.
. (takes few days)
.
$ python treat_results.py -P ../results/agent_log/#{timestamp} -S True
```

## cautions
 - Please do not forget deletion of `results/combinations/designs` directory for component changes
 

WIP
