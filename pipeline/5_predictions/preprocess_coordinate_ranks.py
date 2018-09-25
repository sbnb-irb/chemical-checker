#!/miniconda/bin/python

# Imports

import sys, os
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, draw,checkJobResultsForErrors,compressJobResults,all_coords
import Psql
import numpy as np
import collections
import subprocess
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

    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    
    
    tmpdir = checkercfg.getDirectory( "temp" )
    
    inferTasksDir = os.path.join(tmpdir,"infer")
    if not os.path.exists(inferTasksDir):
        os.makedirs(inferTasksDir)
    

    
    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)
        
   
    
    WD = os.path.dirname(os.path.realpath(__file__))
    
    task_dir = os.path.join(inferTasksDir,'coordinate_ranks')


    if os.path.exists(task_dir) == False:
        c = os.makedirs(task_dir)
        
    os.chdir(task_dir)
    task_script = WD + "/../../src/infer_similarity/coordinate_ranks.py"
    
    granularity = 5
    
    filename = os.path.join(tmpdir , str(uuid.uuid4()))
    
    coordinates = all_coords()
    
    coordinates = sorted(coordinates)
    
    S = 0
    with open(filename, "w") as f:
        for i in xrange(len(coordinates)-1):
            for j in range(i+1,len(coordinates)):
                f.write("%s---%s---%s\n" % (coordinates[i], coordinates[j],dbname))
                S += 1

    
    t = math.ceil(float(S)/granularity)
    jobName = 'coordrank'
    
    
    logFilename = os.path.join(logsFiledir,jobName+".qsub")
    
    scriptFile = 'singularity exec ' + checkerconfig.SING_IMAGE + ' python ' +task_script + ' \$i ' 
    
    cmdStr = checkerconfig.SETUPARRAYJOB % { 'JOB_NAME':jobName, 'NUM_TASKS':t,
                                          'TASKS_LIST':filename,
                                          'COMMAND':scriptFile}
    
        
    execAndCheck(cmdStr,log)
    
    log.info( " - Launching the job %s on the cluster " % (jobName) )
    cmdStr = SUBMITJOBANDREADY+" "+task_dir+" "+jobName+" "+logFilename
    execAndCheck(cmdStr,log)
    
    
    checkJobResultsForErrors(task_dir,jobName,log)    
    compressJobResults(task_dir,jobName,['tasks'],log)


    
  

if __name__ == '__main__':
    main()
