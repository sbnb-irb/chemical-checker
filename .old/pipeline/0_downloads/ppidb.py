#!/miniconda/bin/python


'''

Fetch our local ppidb.

This script is adapted from a query by Teresa Juan-Blanco.

'''

# Imports

import csv
import os, sys
import numpy as np
import glob
import collections
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck
sys.path.append(os.path.join(sys.path[0],"../"))
import Psql

import checkerconfig

# Variables


output_folder = "XXXX/ppidb" # I recommend that this folder is named ppidb.

# Functions

def fetch_data(ppidb_dbname):

    PPIdbQuery  = """
                    SELECT uniref_canonical1, uniref_canonical2
                    FROM PPIDB_INTERACTIONS
                    WHERE uniprot_taxid1 = '9606' AND
                    uniprot_taxid2 = '9606' AND
                    active_uniprot_proteins AND
                    NOT ambiguous_mapping AND
                    NOT duplicated_in_author_inferences AND
                    uniref_canonical1 != uniref_canonical2
                    AND (method_binary OR curation_binary);"""

    R = set()
    for r in Psql.qstring(PPIdbQuery, ppidb_dbname):
        if r[0] == r[1]: continue
        R.update([tuple(sorted([r[0], r[1]]))])

    if not os.path.exists(output_folder): os.mkdir(output_folder)
    
    with open(output_folder + "/interactions.tsv", "w") as f:
        for r in R:
            f.write("%s\t%s\n" % (r[0], r[1]))


# Main

def main():
    
    if len(sys.argv) != 3:
        usage(sys.argv[0])
        sys.exit(1)
  
    configFilename = sys.argv[2]
    

    checkercfg = checkerconfig.checkerConf(configFilename )  

    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)

    downloadsdir = checkercfg.getDirectory( "downloads" )
    
    global output_folder
    
    networksdir = checkercfg.getDirectory( "networks" )
    
    dirname = "ppidb"
    output_folder = os.path.join(networksdir,dirname)
    check_dir = os.path.exists(output_folder)


    if check_dir == False:
        c = os.makedirs(output_folder)
    
    ppidb_dbname = checkercfg.getVariable('PpiDB', 'dbname'     )
    
    log.info( "Fetching data...")
    fetch_data(ppidb_dbname)

if __name__ == '__main__':
    main()

