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
  
  dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    

  log = logSystem(sys.stdout)
  log.debug(os.getcwd())

  dirName = os.path.abspath(sys.argv[0]).split("/")[-2]
  
  bOk = True
  for task in tasks:
    log.info("====>>>> "+task[0]+" <<<<====")
    readyFilename = os.path.join(readyFiledir,dirName+"_"+task[0]+".ready")
    if os.path.exists(readyFilename):
      log.info( "Ready file for task %s does exist. Skipping this task..." % task[0] )
      continue
    # Then I execute the current task
    try:
        
      sig.generate_signatures(dbname = dbname,table = task[0],num_topics = task[1],max_freq=task[2],multipass = task[3],log = log,tmpDir = tempdir)
      
      cmdStr = "touch "+readyFilename
      subprocess.call(cmdStr,shell=True)
    except OSError, e:
      log.critical( "Execution of Script %s failed: %s" % (task[0],e) )
      sys.exit(1)
    log.info("====>>>> "+task[0]+"...done! <<<<====")
  
  if bOk:
    readyFilename = os.path.join(readyFiledir,dirName+".ready")
    log.debug(readyFilename)
    cmdStr = "touch "+readyFilename
    subprocess.call(cmdStr,shell=True)
  
# Main
main()

