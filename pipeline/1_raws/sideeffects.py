#!/miniconda/bin/python


# Imports

import collections
import sys, os
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, draw
import Psql
import networkx as nx
import csv


import checkerconfig


# Variables

sider_molrepo = "XXX"
sider_file = "XXX" # db/meddra_all_se.tsv
table = "sideeffects"


# Functions

def parse_sider():

    cid_inchikey = {}
    inchikey_inchi = {}
    f = open(sider_molrepo, "r")
    for l in f:
        l = l.rstrip("\n").split("\t")
        if not l[2]: continue
        cid_inchikey[l[0]] = l[2]
        inchikey_inchi[l[2]] = l[3]
    f.close()

    inchikey_raw = collections.defaultdict(set) 
    with open(sider_file, "r") as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            cid = l[1]
            if cid not in cid_inchikey: continue
            inchikey_raw[cid_inchikey[cid]].update([l[2]])

    return inchikey_raw,inchikey_inchi


def insert_to_database(inchikey_raw,inchikey_inchi):

    inchikey_raw = dict((k, ",".join(sorted(v))) for k,v in inchikey_raw.iteritems())          
    
    todos = Psql.insert_structures(inchikey_inchi, dbname)
    for ik in todos:
        draw(ik,inchikey_inchi[ik])

    Psql.insert_raw(table, inchikey_raw,dbname)


# Main

def main():
    
    import argparse
    
    if len(sys.argv) != 2:
        sys.exit(1)
  
    configFilename = sys.argv[1]

    checkercfg = checkerconfig.checkerConf( configFilename)  
    global dbname,chembl_dbname
    chembl_dbname = checkerconfig.chembl

    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    global sider_file,sider_molrepo
    
    downloadsdir = checkercfg.getDirectory( "downloads" )
    sider_file = os.path.join(downloadsdir,checkerconfig.sider_file)
    
    sider_molrepo = os.path.join(checkercfg.getDirectory( "molRepo" ),"sider.tsv")
    logsFiledir = checkercfg.getDirectory( "logs" )
    
    log = logSystem(sys.stdout)

    log.info( "Parsing SIDER")
    inchikey_raw,inchikey_inchi = parse_sider()

    log.info( "Inserting to database")
    insert_to_database(inchikey_raw,inchikey_inchi)


if __name__ == "__main__":

    main()