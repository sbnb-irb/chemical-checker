#!/usr/bin/env python
#
# Runs all the tasks of this step
#

# Imports
import os
import sys
import glob
import subprocess


sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck,checkJobResultsForErrors,compressJobResults,all_coords
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


# Constants
#        TABLE,         NUM_TOPICS,MAX_FREQ,MULTIPASS
tasks = [
  ( 'Preprocess Coordinate Correlation',                  'coordinate_correlation.py',                      'preprocess_coordinate_correlation' ),
  ( 'Preprocess Coordinate Conditionals',                 'coordinate_conditionals.py',                     'preprocess_coordinate_conditionals' ),
  ( 'Preprocess Coordinate Paired Conditionals',          'coordinate_paired_conditionals.py',              'preprocess_coordinate_paired_conditionals' ),
  ( 'Preprocess Coordinate Ranks',                        'coordinate_ranks.py',                            'preprocess_coordinate_ranks' ),
  ( 'Preprocess Coordinate Cluster Paired Conditionals',  'coordinate_clust_paired_conditionals.py',        'preprocess_coordinate_clust_paired_conditionals' ),
  ( 'Preprocess Expected Distributions',                  'expected_distributions.py',                      'preprocess_expected_distributions' )]


# Functions
def usage(progName):
  print "Usage: "+progName+" <config_ini>"

def main():
 # Check arguments
  # Check arguments
  if len(sys.argv) != 2:
    usage(sys.argv[0])
    sys.exit(1)
  
  configFilename = sys.argv[1]

  checkercfg = checkerconfig.checkerConf(configFilename )  

  readyFiledir = checkercfg.getDirectory( "ready" )
  tmpdir = checkercfg.getDirectory( "temp" )
  
  inferTasksDir = os.path.join(tmpdir,"infer")
  if not os.path.exists(inferTasksDir):
        os.makedirs(inferTasksDir)
  
  version = checkercfg.getVariable("General",'release')
  dbname = checkerconfig.dbname + "_" + version
  vname = 'sig'
  versionpath = checkerconfig.RELEASESPATH + "/" + version
    
  errTasks = set()
  finished   = set()
  
  logsFiledir = checkercfg.getDirectory( "logs" )
  

  log = logSystem(sys.stdout)
  log.debug(os.getcwd())
  
  WD = os.path.dirname(os.path.realpath(__file__))

  dirName = os.path.abspath(sys.argv[0]).split("/")[-2]
  
  
  while True:
    for task in tasks:
        
        if task[2] in finished:
            continue
        readyFilename = os.path.join(readyFiledir,dirName+"_"+task[2]+".ready")
        if os.path.exists(readyFilename):
          log.info( "Ready file for task %s does exist. Skipping this task..." % task[2] )
          finished.add(task[2])
          continue
        else:
            
            jobTasksDir = os.path.join(inferTasksDir,task[2])
            if not os.path.exists(jobTasksDir):
                os.makedirs(jobTasksDir)
        
        
            os.chdir(jobTasksDir)
        
            jobName = "task_" + task[2]
            
            jobReadyFile   = os.path.join(jobTasksDir,jobName+".ready")
            jobErrorFile   = os.path.join(jobTasksDir,jobName+".error")
            jobStartedFile = os.path.join(jobTasksDir,jobName+".started")
            
            if not os.path.exists(jobStartedFile):
                log.info("====>>>> "+task[0]+" <<<<====")
                
                for filename in glob.glob(os.path.join(jobTasksDir,"task_"+task[2] + "*")) :
                    os.remove(filename)
                
                logFilename = os.path.join(logsFiledir,jobName+".qsub")
                
                task_script = WD + "/../../src/infer_similarity/" + task[1]
                
                granularity = 1
    
                filename = os.path.join(tmpdir , str(uuid.uuid4()))
                
                coordinates = all_coords()
                
                coordinates = sorted(coordinates)
                
                S = 0
                with open(filename, "w") as f:
                    if task[2].find("coordinate") >= 0 :
                        if task[2].find("conditionals") >= 0 :
                            for i in xrange(len(coordinates)):
                                for j in xrange(len(coordinates)):
                                    f.write("%s---%s---%s\n" % (coordinates[i], coordinates[j],dbname))
                                    S += 1
                            
                        else:
                            for i in xrange(len(coordinates)-1):
                                for j in range(i+1,len(coordinates)):
                                    f.write("%s---%s---%s\n" % (coordinates[i], coordinates[j],dbname))
                                    S += 1
                    else:
                        for coord in coordinates:
                            f.write("%s---%s---%s\n" % (coord, vname, versionpath))
                            S += 1
            
                
                t = math.ceil(float(S)/granularity)
                
                
                logFilename = os.path.join(logsFiledir,jobName+".qsub")
                
                scriptFile = 'singularity exec ' + checkerconfig.SING_IMAGE + ' python ' +task_script + ' \$i ' 
                
                cmdStr = checkerconfig.SETUPARRAYJOB % { 'JOB_NAME':jobName, 'NUM_TASKS':t,
                                                      'TASKS_LIST':filename,
                                                      'COMMAND':scriptFile}
    
                execAndCheck(cmdStr,log)
                
                log.info( " - Launching the job %s on the cluster " % (jobName) )
                cmdStr = SUBMITJOBANDREADY+" "+jobTasksDir+" "+jobName+" "+logFilename + " &"
                execAndCheck(cmdStr,log)
                cmdStr = 'touch '+jobStartedFile
                execAndCheck(cmdStr,log)
            else:
                if os.path.exists(jobErrorFile):
                    # Notify error 
                    log.error( "Preprocessing for task %s failed" % task[2])
                    errTasks.add(task[2])
                    finished.add(task[2])
                    continue
                
                if os.path.exists(jobReadyFile):
                        #checkJobResultsForErrors(jobTasksDir,jobName,log)
                        cmdStr = 'touch '+readyFilename
                        execAndCheck(cmdStr,log)
                        finished.add(task[2])
                        compressJobResults(jobTasksDir,jobName,['tasks'],log)
                        log.info("====>>>> "+task[0]+"...done! <<<<====")
                        
    if len(finished) == len(tasks):
        break
    else:
        time.sleep(checkerconfig.POLL_TIME_INTERVAL)
      
  if len(errTasks) > 0:
    log.critical( "Error while preprocessing for the following tasks: "+str(", ").join(sorted(errTasks)))
    sys.exit(1)
    

 
  
# Main
main()


