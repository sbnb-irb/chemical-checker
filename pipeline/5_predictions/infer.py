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

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))


# Main

def main():
    
    import argparse
    
    if len(sys.argv) != 2:
        sys.exit(1)
  
    configFilename = sys.argv[1]

    checkercfg = checkerconfig.checkerConf( configFilename)  

    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    vnumber = checkercfg.getVariable("General",'release')

    
    tmpdir = checkercfg.getDirectory( "temp" )
    
    inferTasksDir = os.path.join(tmpdir,"infer")
    if not os.path.exists(inferTasksDir):
        os.makedirs(inferTasksDir)
    
    inferfolder = checkerconfig.RELEASESPATH+"/"+vnumber+"/infer"

    
    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)
        
   
    
    WD = os.path.dirname(os.path.realpath(__file__))
    
    task_dir = os.path.join(inferTasksDir,'inference')


    if os.path.exists(task_dir) == False:
        c = os.makedirs(task_dir)
        
    os.chdir(task_dir)
    task_script = WD + "/../../src/infer_similarity/inference.py"
    
    granularity = 10
    
    filename = os.path.join(tmpdir , str(uuid.uuid4()))
    
    coordinates = all_coords()
    
    coordinates = sorted(coordinates)
    chunks = 3
    vname = 'sig'
    perf = "NULL"
    mone = "mone_y"
    genex = "genex_n"
    cc_inchikeys_file = inferfolder + "/models/inchikeys.tsv"
    
    with open(cc_inchikeys_file, "r") as f:
        inchikeys = [l.rstrip("\n") for l in f]

    
    S = 0
    with open(filename, "w") as f:
        for c in chunker(inchikeys, chunks):
            s = "---".join([dbname,inferfolder,vname, perf, mone, genex] + c)
            f.write("%s\n" % s)
            S += 1


    
    t = math.ceil(float(S)/granularity)
    jobName = 'infer'
    
    
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
