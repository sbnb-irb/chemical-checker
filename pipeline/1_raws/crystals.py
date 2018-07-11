#!/miniconda/bin/python


# Imports

import csv
import numpy as np
import collections

import sys, os
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck,draw

import Psql
import checkerconfig

# Variables

pdb_molrepo  = "XXXX"
ecod_domains = "XXXX" # ecod.latest.domains.txt
dbname           = ''

table = "crystals"

# Functions

def parse_ecod():

    # Read molrepo

    ligand_inchikey = {}
    inchikey_inchi = {}
    with open(pdb_molrepo, "r") as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            if not l[2]: continue
            ligand_inchikey[l[0]] = l[2]
            inchikey_inchi[l[2]] = l[3]

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
    
    return inchikey_ecod,inchikey_inchi


def insert_to_database(inchikey_ecod,inchikey_inchi):

    inchikey_ecod = dict((k,",".join(v)) for k,v in inchikey_ecod.iteritems())

    todos = Psql.insert_structures(inchikey_inchi, dbname)
    for ik in todos:
        draw(ik,inchikey_inchi[ik])
    Psql.insert_raw(table, inchikey_ecod, dbname)


# Main

def main():
    
    import argparse
    
    if len(sys.argv) != 2:
        sys.exit(1)
  
    configFilename = sys.argv[1]

    global dbname,ecod_domains,pdb_molrepo
    checkercfg = checkerconfig.checkerConf( configFilename)  

    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
   
    downloadsdir = checkercfg.getDirectory( "downloads" )
    ecod_domains = os.path.join(downloadsdir,checkerconfig.eco_domains)
    pdb_molrepo = os.path.join(checkercfg.getDirectory( "molRepo" ),"chembl.tsv")
    
    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)

    log.info(  "Reading ECOD")
    inchikey_ecod,inchikey_inchi = parse_ecod()

    log.info(  "Inserting to database")
    insert_to_database(inchikey_ecod,inchikey_inchi)
    

if __name__ == '__main__':
	main()