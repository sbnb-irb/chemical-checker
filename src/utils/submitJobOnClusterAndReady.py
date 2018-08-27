#!/usr/bin/env python

# imports
import sys
import os

from checkerUtils import logSystem, execAndCheck, getNumOfLines

sys.path.append(os.path.join(sys.path[0],"../../pipeline"))
import checkerconfig

# constants

# functions
def usage(progName):
  print "Usage: "+progName+" <directory> <job_name> <log_file>"
  print

def main():
  # Body of function main
  if len( sys.argv ) != 4:
    usage(sys.argv[0])
    sys.exit(-1)

  directory   = os.path.abspath(sys.argv[1])
  jobName     = sys.argv[2]
  logFilename = os.path.abspath(sys.argv[3])
  
  log = logSystem(sys.stdout)
  
  MASTERNODE = checkerconfig.MASTERNODE
  
  # And we start it
  cmdStr = checkerconfig.SUBMITJOB % { 'JOB_NAME':jobName }
  wrapperCmd = "cd "+directory+"; "+cmdStr+" &> "+logFilename
  if MASTERNODE != "":
    wrapperCmd = "ssh "+MASTERNODE+" '%s'" % wrapperCmd
  try:
    execAndCheck(wrapperCmd,log)
  except:
    log.error( "ERROR running this script in cluster: " + cmdStr  )

  # Then we check the log file
  logFile = open( logFilename )
  bExitOk = True
  for line in logFile:
    
    if line.find( "exited with exit code" ) != -1:
        #print line
        #print jobId
        exitCode = int(line.rstrip("\n\r.").split()[6])
        #print exitCode
        
        if exitCode != 0:
            bExitOk = False
            break

  logFile.close()
  
  
  # Once we are done we put a file <job_name>.ready in the excution directory
  if bExitOk:
    cmdStr = 'touch '+os.path.join(directory,jobName+".ready")
    execAndCheck(cmdStr,log)
  else:
    cmdStr = 'touch '+os.path.join(directory,jobName+".error")
    execAndCheck(cmdStr,log)
  
# main stream
main()
