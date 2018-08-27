#!/usr/bin/env python
#
# Runs all the tasks of this step
#

# Imports
import os
import sys
import subprocess


sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, all_coords


import checkerconfig



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
  
  release = checkercfg.getVariable("General",'release')
  dbname = checkerconfig.dbname + "_" + release

  log = logSystem(sys.stdout)
  log.debug(os.getcwd())
  
  tempdir = checkercfg.getDirectory( "temp" )
  
  simTasksDir = os.path.join(tempdir,"sim_tasks")
  if not os.path.exists(simTasksDir):
        os.makedirs(simTasksDir)

  dirName = os.path.abspath(sys.argv[0]).split("/")[-2]
  
  bOk = True
  for task in all_coords():
    log.info("====>>>> "+task+" <<<<====")
    readyFilename = os.path.join(readyFiledir,dirName+"_"+task+".ready")
    if os.path.exists(readyFilename):
      log.info( "Ready file for task %s does exist. Skipping this task..." % task )
      continue
    # Then I execute the current task
    try:
      jobTasksDir = os.path.join(simTasksDir,task)
      if not os.path.exists(jobTasksDir):
            os.makedirs(jobTasksDir)
      scriptName = os.path.join(sys.path[0],"../../src/similarity/similarity.py") + " --dbname " + dbname + " --coordinates " + task + " --rundir " + jobTasksDir + " --vnumber " + release
      print scriptName
      p = subprocess.Popen( [scriptName], stderr=subprocess.STDOUT )
      (pid,retcode) = os.waitpid(p.pid, 0)
      if retcode != 0:
        bOk = False
        if retcode > 0:
          log.error( "Script similarity for coord %s produced some ERROR, please check (exit code %d)!" % (task,retcode) )
        elif retcode < 0:
          log.error( "Script similarity was terminated by signal %d" % (-retcode))
        sys.exit(1)
      cmdStr = "touch "+readyFilename
      subprocess.call(cmdStr,shell=True)
    except OSError, e:
      log.critical( "Execution of Script %s failed: %s" % (tasks[i][1],e) )
      sys.exit(1)
    log.info("====>>>> "+task+"...done! <<<<====")
  
  if bOk:
    readyFilename = os.path.join(readyFiledir,dirName+".ready")
    log.debug(readyFilename)
    cmdStr = "touch "+readyFilename
    subprocess.call(cmdStr,shell=True)
  
# Main
main()

