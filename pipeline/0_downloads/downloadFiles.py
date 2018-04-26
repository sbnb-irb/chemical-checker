#!/usr/bin/python

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

print "Entering download files"

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
    
    
for file in checkerconfig.downloads:
      
    if file[3] != '':
        if os.path.exists(downloadsdir + '/' + file[3]):
            continue
    
    download = 'wget '
      
    if file[1] != '':
        download += ( '--user ' + file[1] + ' --password ' + file[2])
    if file[3] != '':
        download += ( ' -O ' + file[3])
        
    log.info( " %s %s" % (download,file[0]))
    out = commands.getstatusoutput(download + ' "' + file[0] + '"')
    if out[0] != 0:
        log.error( "Step get " + file[3] + " in downloadFiles.py failed with message: " + out[1])
        sys.exit(1)
  
    if file[1].endswith(".zip") or file[3].endswith(".zip"):
        out = commands.getstatusoutput('unzip -o ' + file[3])
        if out[0] != 0:
            log.error( "Step zip in " + file[3] + " failed with message: " + out[1])
            sys.exit(1)
        continue
            
    if file[1].endswith(".tgz") or file[3].endswith(".tgz"):
        out = commands.getstatusoutput('tar -xzf ' + file[3])
        if out[0] != 0:
            log.error( "Step tgz in " + file[3] + " failed with message: " + out[1])
            sys.exit(1)
        continue
            
    if file[1].endswith("tar.gz") or file[3].endswith("tar.gz"):
        out = commands.getstatusoutput('tar -xzf ' + file[3])
        if out[0] != 0:
            log.error( "Step tar.gz in " + file[3] + " failed with message: " + out[1])
            sys.exit(1)
        continue
    
    if file[1].endswith(".gz") or file[3].endswith(".gz"):
        out = commands.getstatusoutput('gunzip -kf ' + file[3])
        if out[0] != 0:
            log.error( "Step gunzip in " + file[3] + " failed with message: " + out[1])
            sys.exit(1)
        continue
    
with open(os.path.join(downloadsdir,checkerconfig.kegg_atcs_download), "r") as f:
        drugs = set()
        for l in f:
            if l[0] == "F":
                drugs.update([l.split()[1]])
        drugs = sorted(drugs)

if not os.path.exists(os.path.join(downloadsdir,checkerconfig.kegg_mol_folder_download)):
    os.makedirs(os.path.join(downloadsdir,checkerconfig.kegg_mol_folder_download))
    
for drug in drugs:
    out = commands.getstatusoutput("wget -O " + os.path.join(downloadsdir,checkerconfig.kegg_mol_folder_download,drug + ".mol") + " http://rest.kegg.jp/get/" + drug + "/mol" )
    if out[0] != 0:
        log.error( "Step wget for mol  " + drug + " failed with message: " + out[1])
        sys.exit(1)
        
        
logFilename = os.path.join(logsFiledir,"loadChemblinDB.log")

job2run = "dropdb --if-exists -h aloy-dbsrv chembl\n"
job2run += "createdb -h aloy-dbsrv chembl\n"
job2run += "psql -h aloy-dbsrv -d chembl -f " + downloadsdir + "/chembl_*/chembl_*_postgresql/*.dmp"
# And we start it
cmdStr = os.path.join(sys.path[0],"../../src/utils/")+ "setupSingleJob.py -x -N db-chembl " + job2run
      
# Then I move to the directory where I want the output generated
wrapperCmd = "cd "+downloadsdir+"; "+cmdStr + "; " +checkerconfig.SUBMITJOB + " job-db-chembl.sh  > " + logFilename
    
if MASTERNODE != "":
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

  