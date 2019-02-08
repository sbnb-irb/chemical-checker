#!/miniconda/bin/python


# Imports

import sys, os
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, draw
import Psql
import collections


import checkerconfig

# Variables

dbname = "XXX" # Psql.mosaic
ctd_molrepo = "XXX"
chemdis_file = "XXX" # db/CTD_chemicals_diseases.tsv
disfile = "XXX" # db/CTD_diseases.tsv

table = "phenotypes"

# Functions

def parse_ctd():

    ctd_inchikey = {}
    inchikey_inchi = {}
    with open(ctd_molrepo) as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            if not l[2]: continue
            ctd_inchikey[l[0]] = l[2] 
            inchikey_inchi[l[2]] = l[3]   

    dis_tree = collections.defaultdict(list)
    f = open(disfile, "r")
    for l in f:
        if l[0] == "#": continue
        l = l.rstrip("\n").split("\t")
        dis_tree[l[1]] = l[5].split("|")
    f.close()

    tree_dis = collections.defaultdict(list)
    for k,v in dis_tree.iteritems():
        for x in v:
            tree_dis[x] += [k]

    def expand_tree(tn):
        tns = []
        x = tn.split("/")[0].split(".")
        for i in xrange(len(x)):
            tns += [".".join(x[:i+1])]
        tns += [tn]
        return tns

    f = open(chemdis_file, "r")
    inchikey_raw = collections.defaultdict(set)

    for l in f:    
        if l[0] == "#": continue
        l = l.rstrip("\n").split("\t")
        if l[5] == "": continue
        dis = l[4]
        cid = l[1]
        if cid not in ctd_inchikey: continue
        inchikey = ctd_inchikey[cid]
        ev = l[5]
        for tn in dis_tree[dis]:
            exp_tns = expand_tree(tn)
            exp_dis = set()
            for exp_tn in exp_tns:
                exp_dis.update(tree_dis[exp_tn])
        exp_dis = sorted(exp_dis)
        for d in exp_dis:
            if not "MESH" in d: continue
            x = []
            if ev == "marker/mechanism":
                x += [d + "(M)"]
            if ev == "therapeutic":
                x += [d + "(T)"]
            for y in x:
                inchikey_raw[inchikey].update([y.split("MESH:")[1]])
    f.close()

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
    global dbname

    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    global chemdis_file,ctd_molrepo,disfile
    
    downloadsdir = checkercfg.getDirectory( "downloads" )
    chemdis_file = os.path.join(downloadsdir,checkerconfig.chemdis_file)
    disfile = os.path.join(downloadsdir,checkerconfig.ctd_diseases)
    
    ctd_molrepo = os.path.join(checkercfg.getDirectory( "molRepo" ),"ctd.tsv")
    logsFiledir = checkercfg.getDirectory( "logs" )
    
    log = logSystem(sys.stdout)

    log.info( "Parsing CTD...")
    inchikey_raw,inchikey_inchi = parse_ctd()

    log.info( "Inserting to database...")
    insert_to_database(inchikey_raw,inchikey_inchi)


if __name__ == "__main__":

    main()