#!/miniconda/bin/python


# Imports

import sys, os
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, draw
import Psql
from gaussian_scale_impute import scaleimpute
import numpy as np
import collections

import checkerconfig

# Variables

dbname = ''

mosaic_molrepo = "XXX" # mosaic.tsv"
all_conditions = "XXX" # data/All_conditions.txt
comb_gt_preds  = "XXX" # data/combined_gene-target-predictions.txt
table = "chemgenet"

pval_01  = 3.37
pval_001 = 7.12

# Functions

def read_mosaic_predictions():

    with open(mosaic_molrepo, "r") as f:
        cgid_inchikey = {}
        inchikey_inchi = {}
        for l in f:
            l = l.rstrip("\n").split("\t")
            if not l[2]: continue
            cgid_inchikey[l[0]] = l[2]
            inchikey_inchi[l[2]] = l[3]

    conds = {}
    f = open(all_conditions, "r")
    H = f.next().split("\t")[1:]
    f.close()

    for h in H:
        cond, Id = h.split("_")
        if Id not in cgid_inchikey: continue
        conds[cond] = cgid_inchikey[Id]

    sig = collections.defaultdict(set)
    with open(comb_gt_preds, "r") as f:
        f.next()
        for l in f:
            l = l.rstrip("\n").split("\t")
            if l[0] not in conds: continue
            ik = conds[l[0]]
            s = float(l[2])
            if s > pval_01: # P-value 0.01
                if s > pval_001: # P-value 0.001
                    d = 2
                else:
                    d = 1
                sig[ik].update(["%s(%d)" % (l[1], d)])

    return sig,inchikey_inchi


def insert_to_database(sig,inchikey_inchi):
    
    inchikey_raw = dict((k, ",".join(sorted(v))) for k,v in sig.iteritems())

    todos = Psql.insert_structures(inchikey_inchi, dbname)
    for ik in todos:
        draw(ik,inchikey_inchi[ik])
    Psql.insert_to_database(table, inchikey_raw,dbname)


# Main

def main():

    import argparse
    
    if len(sys.argv) != 2:
        sys.exit(1)
  
    configFilename = sys.argv[1]

    checkercfg = checkerconfig.checkerConf( configFilename)  
    global dbname,mosaic_molrepo,all_conditions,comb_gt_preds
    
    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    chembl_dbname = checkerconfig.chembl
    
    all_conditions = os.path.join(checkercfg.getDirectory( "downloads" ),checkerconfig.all_conditions)
    comb_gt_preds = os.path.join(checkercfg.getDirectory( "downloads" ),checkerconfig.comb_gt_preds)

    mosaic_molrepo = os.path.join(checkercfg.getDirectory( "molRepo" ),"mosaic.tsv")
    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)
    
    log.info( "Parsing Chad Myers' Mosaic")
    sig,inchikey_inchi = read_mosaic_predictions()

    log.info( "Inserting to database")
    insert_to_database(sig,inchikey_inchi)

if __name__ == '__main__':
    main()
