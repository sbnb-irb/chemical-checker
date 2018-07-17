#!/miniconda/bin/python

# Imports

import sys, os
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, draw
import Psql
import csv
import collections
import numpy as np
import networkx as nx

import checkerconfig


# Variables

drugbank_molrepo = "XXX"
umls2mesh = "XXX" # db/map_umls_2_mesh.tsv
repodb = "XXX" # db/repodb.csv
chembl_molrepo = "XXX"
ctd_diseases = "XXX" # db/CTD_diseases.tsv

chembl_dbname = "XXX"

table = "indications"

# Functions

def parse_repodb(IND = None):

    if IND is None:
        IND = collections.defaultdict(list)

    # Parse DrugBank
    dbid_inchikey = {}
    inchikey_inchi = {}
    f = open(drugbank_molrepo, "r")
    for l in f:
        l = l.rstrip("\n").split("\t")
        if not l[2]: continue
        dbid_inchikey[l[0]] = l[2]
        inchikey_inchi[l[2]] = l[3]
    f.close()

    # Read UMLS to MESH
    umls_mesh = collections.defaultdict(set)
    f = open(umls2mesh, "r")
    for l in f:
        if l[0] == "#": continue
        l = l.rstrip("\n").split("\t")
        if l[0] == "diseaseId": continue
        if l[2] == "MSH":
            umls_mesh[l[0]].update([l[3]])
    f.close()

    # Parse RepoDB
    f = open(repodb, "r")
    f.next()
    for l in csv.reader(f):
        if l[1] not in dbid_inchikey: continue
        if l[3] not in umls_mesh: continue
        for meshid in umls_mesh[l[3]]:
            if l[4] == "Withdrawn" or l[4] == "NA" or l[4] == "Suspended": continue
            if l[4] == "Approved":
                phase = 4
            else:
                if "Phase 3" in l[5]:
                    phase = 3
                else:
                    if "Phase 2" in l[5]:
                        phase = 2
                    else:
                        if "Phase 1" in l[5]:
                            phase = 1
                        else:
                            if "Phase 0" in l[5]:
                                phase = 0
                            else:
                                continue
            IND[(dbid_inchikey[l[1]], meshid)] += [phase]
    f.close()

    return IND,inchikey_inchi


def parse_chembl(inchikey_inchi,IND = None):

    if IND is None:
        IND = collections.defaultdict(list)

    # Parse ChEMBL molrepo
    chemblid_inchikey = {}
    f = open(chembl_molrepo, "r")
    for l in f:
        l = l.rstrip("\n").split("\t")
        if not l[2]: continue
        chemblid_inchikey[l[0]] = l[2]
        inchikey_inchi[l[2]] = l[3]
    
    # Query ChEMBL
    R = Psql.qstring('''
    SELECT md.chembl_id, di.mesh_id, di.max_phase_for_ind

    FROM molecule_dictionary md, drug_indication di

    WHERE

    md.molregno = di.molregno''', chembl_dbname)

    for r in R:
        if r[0] not in chemblid_inchikey: continue
        IND[(chemblid_inchikey[r[0]], r[1])] += [r[2]]

    IND = dict((k, np.max(v)) for k,v in IND.iteritems())

    return IND


def include_mesh(IND):

    G = nx.DiGraph()
    with open(ctd_diseases, "r") as f:
        for l in f:
            if l[0] == "#": continue
            l = l.rstrip("\n").split("\t")
            disid = l[1]
            pardisids = l[4].split("|")
            if pardisids == [""]:
                pardisids = ["ROOT"]
            for pardisid in pardisids:
                G.add_edge(pardisid, disid)

    classIND = collections.defaultdict(list)
    for k,v in IND.iteritems():
        classIND[k] = [v]
        node = "MESH:"+k[1]
        if node not in G: continue
        path = nx.all_simple_paths(G, "ROOT", node)
        dis = [d.split("MESH:")[1] for p in path for d in p if "MESH:" in d]
        for d in dis:
            classIND[(k[0], d)] += [v]

    classIND = dict((k, np.max(v)) for k,v in classIND.iteritems())

    return classIND


def insert_to_database(classIND,inchikey_inchi):

    inchikey_raw = collections.defaultdict(list)
    for k,v in classIND.iteritems():
        inchikey_raw[k[0]] += [k[1] + "(%d)" % v]
    inchikey_raw = dict((k, ",".join(v)) for k,v in inchikey_raw.iteritems())
    
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
    global ctd_diseases,chembl_molrepo,drugbank_molrepo,repodb,umls2mesh
    
    downloadsdir = checkercfg.getDirectory( "downloads" )
    ctd_diseases = os.path.join(downloadsdir,checkerconfig.ctd_diseases)
    repodb = os.path.join(downloadsdir,checkerconfig.repodb)
    umls2mesh = os.path.join(downloadsdir,checkerconfig.umls_disease_mappings)
    chembl_molrepo = os.path.join(checkercfg.getDirectory( "molRepo" ),"chembl.tsv")
    drugbank_molrepo = os.path.join(checkercfg.getDirectory( "molRepo" ),"drugbank.tsv")
    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)

    log.info( "Parsing RepoDB")
    IND,inchikey_inchi = parse_repodb()

    log.info( "Parsing ChEMBL")
    IND = parse_chembl(inchikey_inchi,IND)

    log.info( "Including MeSH hierarchy")
    classIND = include_mesh(IND)

    log.info( "Inserting to database")
    insert_to_database(classIND,inchikey_inchi)


if __name__ == "__main__":
    main()