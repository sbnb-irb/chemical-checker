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
from checkerUtils import logSystem, execAndCheck,checkJobResultsForErrors
import time

import checkerconfig


SUBMITJOBANDREADY = os.path.join(sys.path[0],'../../src/utils/submitJobOnClusterAndReady.py')



# Functions
def usage(progName):
  print "Usage: "+progName+" <config_ini>"

def main():
  # Check arguments
  if len(sys.argv) != 2:
    usage(sys.argv[0])
    sys.exit(1)
  
  configFilename = sys.argv[1]

  checkercfg = checkerconfig.checkerConf(configFilename )  

  readyFiledir = checkercfg.getDirectory( "ready" )
  tempdir = checkercfg.getDirectory( "temp" )
  
  all_tables = checkercfg.getTableList("all")
  
  dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    
  errTasks = set()
  finished   = set()
  
  
  logsFiledir = checkercfg.getDirectory( "logs" )
  jobTasksDir = os.path.join(tempdir,"clusters_tasks")
  if not os.path.exists(jobTasksDir):
        os.makedirs(jobTasksDir)

  log = logSystem(sys.stdout)
  log.debug(os.getcwd())
  
  WD = os.path.dirname(os.path.realpath(__file__))

  dirName = os.path.abspath(sys.argv[0]).split("/")[-2]
  balance = checkercfg.getVariable(dirName,'balance')

  
  for filename in glob.glob(os.path.join(jobTasksDir,"*.ready")) :
        os.remove(filename)
  for filename in glob.glob(os.path.join(jobTasksDir,"*.started")) :
        os.remove(filename)
  for filename in glob.glob(os.path.join(jobTasksDir,"*.error")) :
        os.remove(filename)
  
  call_clust_script = WD + "/call_clusters.py"
  
  os.chdir(jobTasksDir)
  
  while True:
    for task in all_tables:
        
        if task in finished:
            continue
        readyFilename = os.path.join(readyFiledir,dirName+"_"+task+".ready")
        if os.path.exists(readyFilename):
          log.info( "Ready file for task %s does exist. Skipping this task..." % task )
          finished.add(task)
          continue
        else:
        
            jobName = "clust_" + task
            
            jobReadyFile   = os.path.join(jobTasksDir,jobName+".ready")
            jobErrorFile   = os.path.join(jobTasksDir,jobName+".error")
            jobStartedFile = os.path.join(jobTasksDir,jobName+".started")
            
            if not os.path.exists(jobStartedFile):
                log.info("====>>>> "+task+" <<<<====")
                
                for filename in glob.glob(os.path.join(jobTasksDir,"clust_"+task + "*")) :
                    os.remove(filename)
                
                logFilename = os.path.join(logsFiledir,jobName+".qsub")
                
                scriptFile = 'singularity exec ' + checkerconfig.SING_IMAGE + ' python ' +call_clust_script + ' '  + task + " " + balance + " " + tempdir
    
                cmdStr = checkerconfig.SETUPSINGLEJOB % { 'JOB_NAME':jobName, 'COMMAND':scriptFile}
    
        
                execAndCheck(cmdStr,log)
    
                log.info( " - Launching the job %s on the cluster " % (jobName) )
                cmdStr = SUBMITJOBANDREADY+" "+jobTasksDir+" "+jobName+" "+logFilename + " &"
                execAndCheck(cmdStr,log)
                cmdStr = 'touch '+jobStartedFile
                execAndCheck(cmdStr,log)
            else:
                if os.path.exists(jobErrorFile):
                    # Notify error 
                    log.error( "Generating signatures for table %s failed" % task)
                    errTasks.add(task)
                    finished.add(task)
                    continue
                
                if os.path.exists(jobReadyFile):
                        #checkJobResultsForErrors(jobTasksDir,jobName,log)
                        cmdStr = 'touch '+readyFilename
                        execAndCheck(cmdStr,log)
                        finished.add(task)
                        log.info("====>>>> "+task+"...done! <<<<====")
                        
    if len(finished) == len(all_tables):
        break
    else:
        time.sleep(checkerconfig.POLL_TIME_INTERVAL)
      
  if len(errTasks) > 0:
    log.critical( "Error while generating signatures for the following tables: "+str(", ").join(sorted(errTasks)))
    sys.exit(1)
    

  readyFilename = os.path.join(readyFiledir,dirName+".ready")
  log.debug(readyFilename)
  cmdStr = "touch "+readyFilename  
  subprocess.call(cmdStr,shell=True)
  
  
# Main
main()

