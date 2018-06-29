#!/miniconda/bin/python

import os,sys,string,commands

sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck
sys.path.append(os.path.join(sys.path[0],"../"))


import checkerconfig

if len(sys.argv) != 3:
    usage(sys.argv[0])
    sys.exit(1)
  
configFilename = sys.argv[2]

checkercfg = checkerconfig.checkerConf(configFilename )  


logsFiledir = checkercfg.getDirectory( "logs" )

log = logSystem(sys.stdout)

downloadsdir = checkercfg.getDirectory( "downloads" )

check_dir = os.path.exists(logsFiledir)

# Changed servers to mirror ftp.ebi.ac.uk (Roger)

if check_dir == False:
  c = os.makedirs(logsFiledir)

check_dir = os.path.exists(downloadsdir)

# Changed servers to mirror ftp.ebi.ac.uk (Roger)

if check_dir == False:
  c = os.makedirs(downloadsdir)

os.chdir(downloadsdir)
    
    
log.info( "Loading Chembl Database")

        
logFilename = os.path.join(logsFiledir,"loadChemblinDB.log")

job2run = '"dropdb --if-exists -h aloy-dbsrv chembl && '
job2run += "createdb -h aloy-dbsrv chembl && "
job2run += 'pg_restore -h aloy-dbsrv -d chembl ' + downloadsdir + '/chembl_*/chembl_*_postgresql/*.dmp"'
# And we start it
cmdStr = os.path.join(sys.path[0],"../../src/utils/")+ "setupSingleJob.py -x -N db-chembl " + job2run
      
# Then I move to the directory where I want the output generated
wrapperCmd = "cd "+downloadsdir+"; "+cmdStr + "; " +checkerconfig.SUBMITJOB + " job-db-chembl.sh  > " + logFilename
    
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

  