#!/usr/bin/env python
#
# Runs all the tasks of this step
#

# Imports
import os
import sys
import subprocess


sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../../src/signatures"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck
import sig

import checkerconfig


# Constants


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
  
  dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
  
  all_tables = checkercfg.getTableList("all")
  

  log = logSystem(sys.stdout)
  log.debug(os.getcwd())

  dirName = os.path.abspath(sys.argv[0]).split("/")[-2]
  
  bOk = True
  for table in all_tables:
    log.info("====>>>> "+table+" <<<<====")
    readyFilename = os.path.join(readyFiledir,dirName+"_"+table+".ready")
    if os.path.exists(readyFilename):
      log.info( "Ready file for task %s does exist. Skipping this task..." % table )
      continue
    # Then I execute the current task
    try:
        
      sig.generate_signatures(dbname = dbname,table = table,log = log,tmpDir = tempdir)
      
      cmdStr = "touch "+readyFilename
      subprocess.call(cmdStr,shell=True)
    except OSError, e:
      log.critical( "Execution of Script %s failed: %s" % (table,e) )
      sys.exit(1)
    log.info("====>>>> "+table+"...done! <<<<====")
  
  if bOk:
    readyFilename = os.path.join(readyFiledir,dirName+".ready")
    log.debug(readyFilename)
    cmdStr = "touch "+readyFilename
    subprocess.call(cmdStr,shell=True)
  
# Main
main()

