#!/miniconda/bin/python

import os,sys,string,commands
import pandas as pd

sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck
sys.path.append(os.path.join(sys.path[0],"../"))


import checkerconfig

SUBMITJOBANDREADY = os.path.join(sys.path[0],'../../src/utils/submitJobOnClusterAndReady.py')


if len(sys.argv) != 3:
    usage(sys.argv[0])
    sys.exit(1)
  
configFilename = sys.argv[2]

checkercfg = checkerconfig.checkerConf(configFilename )  

logsFiledir = checkercfg.getDirectory( "logs" )

log = logSystem(sys.stdout)

downloadsdir = checkercfg.getDirectory( "downloads" )

check_dir = os.path.exists(logsFiledir)


if check_dir == False:
  c = os.makedirs(logsFiledir)

check_dir = os.path.exists(downloadsdir)


if check_dir == False:
  c = os.makedirs(downloadsdir)
  
networksdir = checkercfg.getDirectory( "networks" )

os.chdir(networksdir)

outputFilename = "list_networks.txt"

jobName = 'prepare-networks'
numTasks = 0

outputFile = open( outputFilename, "w" )
for i in [d for d in os.listdir('.') if os.path.isdir(d)]:
    outputFile.write(i+"\n")
    numTasks += 1
outputFile.close()
listFilename =os.path.join( '', outputFilename)
logFilename = os.path.join(logsFiledir,jobName+".qsub")

scriptFile = 'singularity exec ' + checkerconfig.SING_IMAGE + ' python ' + os.path.dirname(currentFile) + '/prepare_network.py -i ' +\
            os.path.join(networksdir,'$i','interactions.tsv') + ' -o ' + os.path.join(networksdir,'$i') +\
            ' -h ' + checkerconfig.HOTNET_PATH

cmdStr = checkerconfig.SETUPARRAYJOB % { 'JOB_NAME':jobName, 'NUM_TASKS':numTasks,
                                      'TASKS_LIST':listFilename,
                                      'COMMAND':scriptFile}
wrapperCmd = cmdStr

#if MASTERNODE != "":
#    wrapperCmd = "ssh "+MASTERNODE+" '%s'" % wrapperCmd
execAndCheck(wrapperCmd,log)

log.info( " - Launching the job %s on the cluster " % (jobName) )
cmdStr = SUBMITJOBANDREADY+" "+networksdir+" "+jobName+" "+logFilename
execAndCheck(cmdStr,log)


checkJobResultsForErrors(networksdir,jobName,log)

    

