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
  wrapperCmd = "cd "+directory+"; "+cmdStr+" > "+logFilename
  if MASTERNODE != "":
    wrapperCmd = "ssh "+MASTERNODE+" '%s'" % wrapperCmd
  execAndCheck(wrapperCmd,log)

  # Then we check the log file
  logFile = open( logFilename )
  bOks = {}
  for line in logFile:
    if line.find( "Your job-array ") != -1:
      #Your job-array 104037.1-2360:1 ("celegans_pdbsearch") has been submitted
      jobSpec = line.rstrip("\n\r").split()[2]
      jobId = jobSpec.split(".")[0]
      taskStart = int(jobSpec.split(".")[1].split("-")[0])
      taskEnd = int(jobSpec.split(".")[1].split("-")[1].split(":")[0])
      for i in range(taskStart,taskEnd+1):
        taskId = jobId+"."+str(i)
        bOks[taskId] = False
    elif line.find( "Your job ") != -1:
      #Your job 104043 ("prova") has been submitted
      jobId = line.rstrip("\n\r").split()[2]
      bOks[jobId] = False
    elif line[:3] == "Job":
      jobId = line.rstrip("\n\r").split()[1]
      if line.find( "exited with exit code" ) != -1:
        #print line
        #print jobId
        exitCode = int(line.rstrip("\n\r.").split()[6])
        #print exitCode
        if jobId.find(".") == -1:
          for k in bOks:
            if exitCode == 0:
              bOks[k] = True
        else:
          if exitCode == 0:
            bOks[jobId] = True
      else:
        bOks[jobId] = False
  logFile.close()
  
  bExitOk = True
  for k in bOks:
    #print k
    #print bOks[k]
    if not bOks[k]:
      bExitOk = False
      break
  
  # Once we are done we put a file <job_name>.ready in the excution directory
  if bExitOk:
    cmdStr = 'touch '+os.path.join(directory,jobName+".ready")
    execAndCheck(cmdStr,log)
  else:
    cmdStr = 'touch '+os.path.join(directory,jobName+".error")
    execAndCheck(cmdStr,log)
  
# main stream
main()
