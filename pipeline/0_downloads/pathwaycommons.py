#!/miniconda/bin/python


'''

Data from PathwayCommons.

Based, mainly, in Teresa Juan-Blanco's scripts.

'''

# Imports
import csv
import os, sys
import numpy as np
import glob
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck
sys.path.append(os.path.join(sys.path[0],"../"))
import Psql

import checkerconfig



# Variables

#### THIS IS FROM TERE'S SCRIPT!!!! ADAPT AS DESIRED.

HOST        = 'http://www.pathwaycommons.org/archives/PC2/v9/' # CHECK LATEST VERSION!!!!
DATA        = 'PathwayCommons9.All.hgnc.txt.gz'
OUTPUTDIR   = 'pathwaycommons'
DBS         = set(['Reactome','KEGG','NetPath','PANTHER','WikiPathways'])

log = ''

#### 



# Functions

##### THIS IS FROM TERE'S SCRIPT #### ADAPT AS DESIRED

def download_pathwaycommons():

    # Download Pathway Commons
    data_file = os.path.join(OUTPUTDIR,DATA)
    data_file = os.path.join(OUTPUTDIR,DATA.replace('.gz',''))
    if not os.path.exists(os.path.join(OUTPUTDIR,DATA)) and not os.path.exists(data_file):
        log.info( "Download %s from Pathway Commons..." % (DATA))
        cmdStr = "wget %s%s -P %s" % (HOST, DATA, OUTPUTDIR)
        config.execAndCheck( cmdStr )
    if not os.path.exists(data_file):
        log.info( 'Unzipping %s...' % DATA)
        config.ZipFileManage(os.path.join(OUTPUTDIR,DATA), OUTPUTDIR, 'unzip')
    if not os.path.exists(os.path.join(OUTPUTDIR,'1.txt')) or not os.path.exists(os.path.join(OUTPUTDIR,'2.txt')):
        log.info( 'Split file in part 1 and 2...' )# Part 1 contains the interactions and part 2 the gene mapping
        cmdStr = "sed '/^$/q' %s >%s/1.txt" % (data_file,OUTPUTDIR)
        config.execAndCheck( cmdStr )
        cmdStr = "sed '1,/^$/d' %s >%s/2.txt" % (data_file,OUTPUTDIR)
        config.execAndCheck( cmdStr )

####


def read_mapping(dbname):
    Map = collections.defaultdict(list)
    
    R = Psql.qstring("select uniprot_ac, genename From uniprotkb_protein where taxid = '9606' and genename != '' and complete = 'Complete proteome'", dbname)


    for r in R: class_prot[r[0]] += [r[1]]

    for l in R: 
        Map[l[0]].append(l[1])
    return Map

def read_pathwaycommons(Map):
    notmap = []
    log.info( 'Reading Pathway Commons...')
    log.info( '...from Part 1')
    pathwaycommon = []
    i = 0
    lines = open(os.path.join(OUTPUTDIR,'1.txt')).readlines()[1:]
    length = len(lines)
    #print length
    for l in lines:
        l = l.split('\t')
        #print l
        i+=1
        if i % 1000 == 0:
            log.info( '...',round(float(i)/length * 100,2),'%')
        if len(l) > 2:
            dbs = l[3].split(';')
            #print dbs
            if len(set(dbs) & DBS) > 0:
                if l[0] in Map and l[2] in Map:
                    pathwaycommon.append((Map[l[0]][0],Map[l[2]][0],{'database':l[3]}))
                else:
                    if l[0] not in Map and l[0].find('CHEBI')<0:
                        notmap.append(l[0])
                    if l[2] not in Map and l[2].find('CHEBI')<0:
                        notmap.append(l[2])
   
    return pathwaycommon


def write_network(pathwaycommon):
    ppis = set()
    for k in pathwaycommon.keys():
        ppis.update([tuple(sorted([k[0], k[1]]))])
    with open(OUTPUTDIR + "/interactions.tsv") as f:
        for ppi in ppis:
            f.write("%s\t%s\n" % (ppi[0], ppi[1]))


# Main

def main():
    
    if len(sys.argv) != 3:
        usage(sys.argv[0])
        sys.exit(1)
  
    configFilename = sys.argv[2]
    
    global log

    checkercfg = checkerconfig.checkerConf(configFilename )  

    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)

    downloadsdir = checkercfg.getDirectory( "downloads" )
    
    global DATA,OUTPUTDIR
    
    DATA =  checkerconfig.pathway_data
    
    networksdir = checkercfg.getDirectory( "networks" )
    
    dirname = "pathwaycommons"
    OUTPUTDIR = os.path.join(networksdir,dirname)
    check_dir = os.path.exists(OUTPUTDIR)


    if check_dir == False:
        c = os.makedirs(OUTPUTDIR)
    
    UNIPROTKB_SRCDBNAME = checkercfg.getVariable('UniprotKB', 'dbname'     )
    Map = read_mapping(UNIPROTKB_SRCDBNAME)
    pathwaycommon = read_pathwaycommons(Map)
    write_network(pathwaycommon)

    
if __name__ == '__main__':
    main()