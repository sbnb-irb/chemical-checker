#!/miniconda/bin/python

import os,sys,string,commands

sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck
sys.path.append(os.path.join(sys.path[0],"../"))
import Psql


import checkerconfig

if len(sys.argv) != 3:
    usage(sys.argv[0])
    sys.exit(1)
  
configFilename = sys.argv[2]

checkercfg = checkerconfig.checkerConf(configFilename )  


logsFiledir = checkercfg.getDirectory( "logs" )

log = logSystem(sys.stdout)

downloadsdir = checkercfg.getDirectory( "downloads" )

new_version = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
previous_version = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'previous_release')


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
wrapperCmd = "cd "+downloadsdir+"; "+cmdStr + "; " +checkerconfig.SUBMITJOB + " job-db-create.sh "
    
if checkerconfig.MASTERNODE != "":
    wrapperCmd = "ssh "+checkerconfig.MASTERNODE+" '%s'" % wrapperCmd
ret = execAndCheck(wrapperCmd,log)
 
 
log.info( "Remove all data only keep some data tables")       
        
query = "SELECT table_name FROM information_schema.tables  WHERE table_schema = 'public'"

tables =  Psql.qstring(query,new_version)

removables = []

for table in tables:
    if table[0] not in checkerconfig.KEEP_TABLES:
        removables.append(table[0])
        
        

tmpfile = downloadsdir + "/temp_truncate.sql"
text = ""
f = open(tmpfile , 'w')
f.write("TRUNCATE " + ','.join(removables) + ";");
f.close()

truncate_cmd =  "psql -d " + new_version +" -h aloy-dbsrv -f " + tmpfile 


cmdStr = os.path.join(sys.path[0],"../../src/utils/")+ "setupSingleJob.py -x -N db-truncate " + truncate_cmd

logFilename2 = os.path.join(logsFiledir,"truncateCheckerDB.log")

      
# Then I move to the directory where I want the output generated
wrapperCmd = "cd "+downloadsdir+"; "+cmdStr + "; " +checkerconfig.SUBMITJOB + " job-db-truncate.sh  " 

  
print wrapperCmd
if checkerconfig.MASTERNODE != "":
    wrapperCmd = "ssh "+checkerconfig.MASTERNODE+" '%s'" % wrapperCmd
ret = execAndCheck(wrapperCmd,log)

