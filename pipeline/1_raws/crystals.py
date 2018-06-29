# Imports

import csv
import numpy as np
import collections

import sys, os
sys.path.append(os.path.join(sys.path[0], "../../chemutils/"))
sys.path.append(os.path.join(sys.path[0],"../config"))

import Psql

# Variables

pdb_molrepo  = "XXXX"
ecod_domains = "XXXX" # ecod.latest.domains.txt
dbname           = ''

table = "crystals"

# Functions

def parse_ecod():

    # Read molrepo

    ligand_inchikey = {}
    with open(pdb_molrepo, "r") as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            if not l[2]: continue
            ligand_inchikey[l[0]] = l[2]

    # Parse ECOD
    # [X-group].[H-group].[T-group].[F-group]

    inchikey_ecod = collections.defaultdict(set)

    f = open(ecod_domains, "r")
    for l in f:
        if l[0] == "#": continue
        l = l.rstrip("\n").split("\t")
        s = "E:" + l[1]
        f_id = l[3].split(".")
        s += ",X:" + f_id[0]
        s += ",H:" + f_id[1]
        s += ",T:" + f_id[2]
        if len(f_id) == 4:
            s += ",F:" + f_id[3]
        lig_ids = l[-1].split(",")
        for lig_id in lig_ids:
            if lig_id not in ligand_inchikey: continue
            inchikey_ecod[ligand_inchikey[lig_id]].update([s])
    f.close()


def insert_to_database(inchikey_ecod):

    inchikey_ecod = dict((k,",".join(v)) for k,v in inchikey_ecod.iteritems())

    Psql.insert_raw(table, inchikey_ecod, dbname)


# Main

def main():
    
    import argparse
    
    if len(sys.argv) != 2:
        sys.exit(1)
  
    configFilename = sys.argv[1]

    checkercfg = checkerconfig.checkerConf( configFilename)  
    chembl_dbname = checkerconfig.chembl

    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
   
    downloadsdir = checkercfg.getDirectory( "downloads" )
    moldir = checkercfg.getDirectory( "molRepo" )
    ecod_domains = os.path.join(downloadsdir,checkerconfig.eco_domains)
    chembl_molrepo = checkercfg.getDirectory( "molRepo" )
    
    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)

    log.info(  "Reading ECOD")
    inchikey_ecod = parse_ecod()

    log.info(  "Inserting to database")
    insert_to_database(inchikey_ecod)
    

if __name__ == '__main__':
	main()