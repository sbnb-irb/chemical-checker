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
from checkerUtils import logSystem, execAndCheck


import checkerconfig


# Constants
tasks = [
  ( 'Create Cluster embeddings',                  'cluster_embedding.py',             'cluster_embedding' ),
  ( 'Generate Projections',                       'gen_projections.py',               'gen_projections' )
 
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
  

  log = logSystem(sys.stdout)
  log.debug(os.getcwd())

  dirName = os.path.abspath(sys.argv[0]).split("/")[-2]
  
  bOk = True
  for i in range(0,len(tasks)):
    log.info("====>>>> "+tasks[i][0]+" <<<<====")
    readyFilename = os.path.join(readyFiledir,dirName+"_"+tasks[i][2]+".ready")
    if os.path.exists(readyFilename):
      log.info( "Ready file for task %s does exist. Skipping this task..." % tasks[i][2] )
      continue
    # Then I execute the current task
    try:
      scriptName = os.path.join(sys.path[0],tasks[i][1])
      p = subprocess.Popen( [scriptName,configFilename], stderr=subprocess.STDOUT )
      (pid,retcode) = os.waitpid(p.pid, 0)
      if retcode != 0:
        bOk = False
        if retcode > 0:
          log.error( "Script %s produced some ERROR, please check (exit code %d)!" % (tasks[i][1],retcode) )
        elif retcode < 0:
          log.error( "Script %s was terminated by signal %d" % (tasks[i][1],-retcode))
        sys.exit(1)
      cmdStr = "touch "+readyFilename
      subprocess.call(cmdStr,shell=True)
    except OSError, e:
      log.critical( "Execution of Script %s failed: %s" % (tasks[i][1],e) )
      sys.exit(1)
    log.info("====>>>> "+tasks[i][0]+"...done! <<<<====")
  
  if bOk:
    readyFilename = os.path.join(readyFiledir,dirName+".ready")
    log.debug(readyFilename)
    cmdStr = "touch "+readyFilename
    subprocess.call(cmdStr,shell=True)
  
# Main
main()

