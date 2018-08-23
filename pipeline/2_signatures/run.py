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
sys.path.append(os.path.join(sys.path[0],"../../src/signatures"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck,checkJobResultsForErrors
import sig
import time

import checkerconfig


SUBMITJOBANDREADY = os.path.join(sys.path[0],'../../src/utils/submitJobOnClusterAndReady.py')


# Constants
#        TABLE,         NUM_TOPICS,MAX_FREQ,MULTIPASS
tasks = [('moa',           200,    None,    False),
         ('metabgenes',    200,    None,    False),
         ('crystals',      500,    None,    False),
         ('binding',       800,    None,    False),
         ('htsbioass',     800,    None,    False),
         ('molroles',      600,    None,    False),
         ('molpathways',   500,    None,    False),
         ('pathways',      200,    None,    False),
         ('bps',           500,    None,    False),
         ('networks',      600,    None,    False),
         ('transcript',    4600,   None,    False),
         ('cellpanel',     None,   None,    False),
         ('chemgenet',     800,    None,    False),
         ('morphology',    None,   None,    False),
         ('cellbioass',    100,    None,    False),
         ('therapareas',   250,    None,    False),
         ('indications',   600,    None,    False),
         ('sideeffects',   700,    None,    False),
         ('phenotypes',    800,    None,    False),
         ('ddis',          250,    None,    False),
         ('fp2d',          1600,   0.8,     True),
         ('fp3d',          1000,   0.8,     True),
         ('scaffolds',     1500,   0.8,     True),
         ('subskeys',      70,     0.9,     True),
         ('physchem',      None,   None,    False)
         ]


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
  tempdir = checkercfg.getDirectory( "temp" )
  
  all_tables = checkercfg.getTableList("all")
  
  dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    
  errTasks = set()
  finished   = set()
  
  logsFiledir = checkercfg.getDirectory( "logs" )
  jobTasksDir = os.path.join(tempdir,"sig_tasks")
  if not os.path.exists(jobTasksDir):
        os.makedirs(jobTasksDir)

  log = logSystem(sys.stdout)
  log.debug(os.getcwd())
  
  WD = os.path.dirname(os.path.realpath(__file__))

  dirName = os.path.abspath(sys.argv[0]).split("/")[-2]
  
  for filename in glob.glob(os.path.join(jobTasksDir,"*.ready")) :
        os.remove(filename)
  for filename in glob.glob(os.path.join(jobTasksDir,"*.started")) :
        os.remove(filename)
  for filename in glob.glob(os.path.join(jobTasksDir,"*.error")) :
        os.remove(filename)
  
  call_sig_script = WD + "/call_sig.py"
  
  os.chdir(jobTasksDir)
  
  while True:
    for task in tasks:
        
        if task[0] in finished:
            continue
        readyFilename = os.path.join(readyFiledir,dirName+"_"+task[0]+".ready")
        if os.path.exists(readyFilename):
          log.info( "Ready file for task %s does exist. Skipping this task..." % task[0] )
          finished.add(task[0])
          continue
        else:
        
            jobName = "task_" + task[0]
            
            jobReadyFile   = os.path.join(jobTasksDir,jobName+".ready")
            jobErrorFile   = os.path.join(jobTasksDir,jobName+".error")
            jobStartedFile = os.path.join(jobTasksDir,jobName+".started")
            
            if not os.path.exists(jobStartedFile):
                log.info("====>>>> "+task[0]+" <<<<====")
                
                for filename in glob.glob(os.path.join(jobTasksDir,"task_"+task[0] + "*")) :
                    os.remove(filename)
                
                logFilename = os.path.join(logsFiledir,jobName+".qsub")
                
                scriptFile = 'singularity exec ' + checkerconfig.SING_IMAGE + ' python ' +call_sig_script + ' ' + dbname + " " + task[0] + " " + str(task[1]) + " " + str(task[2]) + " " + str(task[3]) + " " + tempdir
    
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
                    log.error( "Generating signatures for table %s failed" % task[0])
                    errTasks.add(task[0])
                    finished.add(task[0])
                    continue
                
                if os.path.exists(jobReadyFile):
                        checkJobResultsForErrors(jobTasksDir,jobName,log)
                        cmdStr = 'touch '+readyFilename
                        execAndCheck(cmdStr,log)
                        finished.add(task[0])
                        log.info("====>>>> "+task[0]+"...done! <<<<====")
                        
    if len(finished) == len(tasks):
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

