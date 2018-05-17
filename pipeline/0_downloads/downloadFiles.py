#!/miniconda/bin/python

import os,sys,string,commands
import pandas as pd

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


if check_dir == False:
  c = os.makedirs(logsFiledir)

check_dir = os.path.exists(downloadsdir)


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

log.info( "Downloading Kegg Molecules: " + str(len(drugs)))
if not os.path.exists(os.path.join(downloadsdir,checkerconfig.kegg_mol_folder_download)):
    os.makedirs(os.path.join(downloadsdir,checkerconfig.kegg_mol_folder_download))
    
for drug in drugs:
    out = commands.getstatusoutput("wget -O " + os.path.join(downloadsdir,checkerconfig.kegg_mol_folder_download,drug + ".mol") + " http://rest.kegg.jp/get/" + drug + "/mol" )
    if out[0] != 0 and "404"  not in out[1] :
        log.error( "Step wget for mol  " + drug + " failed with message: " + out[1])
        sys.exit(1)

log.info( "Converting Zscore xls file to csv")

data_xls = pd.read_excel(os.path.join(downloadsdir,"output/DTP_NCI60_ZSCORE.xls"), index_col=0,usecols="A:F")
data_xls.to_csv(os.path.join(downloadsdir,checkerconfig.nci60_download), encoding='utf-8')

