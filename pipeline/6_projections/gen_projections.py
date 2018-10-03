#!/miniconda/bin/python

# Imports

import sys, os
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, checkJobResultsForErrors,compressJobResults
import Psql
from subprocess import call, Popen
import subprocess
import numpy as np
import collections
import h5py
import time
import uuid
import math


import checkerconfig


SUBMITJOBANDREADY = os.path.join(sys.path[0],'../../src/utils/submitJobOnClusterAndReady.py')

# Main

def main():
    
    import argparse
    
    if len(sys.argv) != 2:
        sys.exit(1)
  
    configFilename = sys.argv[1]

    checkercfg = checkerconfig.checkerConf( configFilename)  

   
    logsFiledir = checkercfg.getDirectory( "logs" )
    tmpdir = checkercfg.getDirectory( "temp" )
    log = logSystem(sys.stdout)
    log.debug(os.getcwd())
    all_tables = checkercfg.getTableList("all")
    granularity = 1
    
    run_dir = os.path.join(tmpdir,"projections")
    filename = os.path.join(tmpdir , str(uuid.uuid4()))
    # Get t
    S = 0
    with open(filename, "w") as f:
        for key in all_tables:
           
            f.write(key + "\n")
            S += 1
    
    if os.path.exists(run_dir) == False:
        c = os.makedirs(run_dir)
    
    t = math.ceil(float(S)/granularity)
    os.chdir(run_dir)
  
    jobName = 'projections'
    
    log = logSystem(sys.stdout)
    
    WD = os.path.dirname(os.path.realpath(__file__))

    
    task_script = WD + "/../../src/projections/proj.py "
    logFilename = os.path.join(logsFiledir,jobName+".qsub")
    
    scriptFile = 'singularity exec ' + checkerconfig.SING_IMAGE + ' python ' + task_script + ' --table \$i --bw 0.1 --manifold tsne --unique --filesdir ' + tmpdir 
    
    cmdStr = checkerconfig.SETUPARRAYJOB % { 'JOB_NAME':jobName, 'NUM_TASKS':t,
                                          'TASKS_LIST':filename,
                                          'COMMAND':scriptFile}
    
        
    execAndCheck(cmdStr,log)
    
    log.info( " - Launching the job %s on the cluster " % (jobName) )
    cmdStr = SUBMITJOBANDREADY+" "+run_dir+" "+jobName+" "+logFilename
    execAndCheck(cmdStr,log)
    
    
    checkJobResultsForErrors(run_dir,jobName,log) 
    compressJobResults(run_dir,jobName,['tasks'],log)   

  

if __name__ == '__main__':
    main()
