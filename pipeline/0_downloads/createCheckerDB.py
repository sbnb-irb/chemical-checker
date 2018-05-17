#!/miniconda/bin/python

import os,sys,string,commands

sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck
sys.path.append(os.path.join(sys.path[0],"../"))


import checkerconfig

if len(sys.argv) != 2:
    usage(sys.argv[0])
    sys.exit(1)
  
configFilename = sys.argv[1]

checkercfg = checkerconfig.checkerConf(configFilename )  


logsFiledir = checkercfg.getDirectory( "logs" )

log = logSystem(sys.stdout)

downloadsdir = checkercfg.getDirectory( "downloads" )

new_version = checkercfg.dbname + "_" + checkercfg.getVariable("General",'release')
previous_version = checkercfg.dbname + "_" + checkercfg.getVariable("General",'previous_release')


os.chdir(downloadsdir)
    
    
log.info( "Take previous database & Create New Database")

dumpfile = os.path.join(downloadsdir,previous_version + '.sql.gz')
        
logFilename = os.path.join(logsFiledir,"createCheckerDB.log")

job2run = '" pg_dump -h aloy-dbsrv ' +   previous_version + ' | gzip -c > ' + dumpfile + ' && '
job2run += "createdb -h aloy-dbsrv " + new_version + " && "
job2run += 'gunzip -c ' + dumpfile + " | psql -h aloy-dbsrv " + new_version + ' "'
# And we start it
cmdStr = os.path.join(sys.path[0],"../../src/utils/")+ "setupSingleJob.py -x -N db-create " + job2run
      
# Then I move to the directory where I want the output generated
wrapperCmd = "cd "+downloadsdir+"; "+cmdStr + "; " +checkerconfig.SUBMITJOB + " job-db-create.sh  > " + logFilename
    
if checkerconfig.MASTERNODE != "":
    wrapperCmd = "ssh "+checkerconfig.MASTERNODE+" '%s'" % wrapperCmd
ret = execAndCheck(wrapperCmd,log)
# Then we check the log file
logFile = open( logFilename )
      
taskOK = False
    
for line in logFile:
    if line.find( "exited with exit code 0") != -1:
        taskOK = True
        break
    
    
    if taskOK == False :
        sys.exit(1)

  