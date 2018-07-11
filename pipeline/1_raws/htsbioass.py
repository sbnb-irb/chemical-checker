#!/miniconda/bin/python

'''
# ChEMBL HTS data (mainly PubChem functional assays)

Get HTS data from ChEMBL, mainly the PubChem section.

Data are divided into active/inactive.
'''

# Imports

import numpy as np
import collections
import networkx as nx

import sys, os
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, draw

import Psql
import math

import checkerconfig


# Variables

pchembl_cutoff = 5

chembl_dbname  = "XXXX"
chembl_molrepo = "XXXX"
dbname         = ''
# Functions

def parse_chembl():

    # Read molrepo

    f = open(chembl_molrepo, "r")
    chemblid_inchikey = {}
    inchikey_inchi = {}
    for l in f:
        l = l.rstrip("\n").split("\t")
        if not l[2]: continue
        chemblid_inchikey[l[0]] = l[2]
        inchikey_inchi[l[2]] = l[3]
    f.close()

    def is_active(r):
        if r[3] >= 5: return True
        if r[2] == "Active" or r[2] == "active": return True
        return False

    # Query

    con = Psql.connect(Psql.chembl)
    con.set_isolation_level(0)
    cur = con.cursor()
    cur.execute('''
    SELECT md.chembl_id, cseq.accession, act.activity_comment, act.pchembl_value

    FROM molecule_dictionary md, activities act, assays ass, component_sequences cseq, target_components t

    WHERE

    (ass.src_id != 1 OR (ass.src_id = 1 AND ass.assay_type = 'F')) AND
    md.molregno = act.molregno AND
    act.assay_id = ass.assay_id AND
    ass.tid = t.tid AND
    t.component_id = cseq.component_id AND
    cseq.accession IS NOT NULL
    ''')

    R = []
    for r in cur:
        if r[0] not in chemblid_inchikey: continue
        if not is_active(r): continue
        R += [(chemblid_inchikey[r[0]], r[1])]

    # Use ChEMBL hierarchy

    S = Psql.qstring("SELECT protein_class_id, parent_id, pref_name FROM protein_classification", chembl_dbname)

    G = nx.DiGraph()

    for s in S: G.add_edge(s[1], s[0]) # The tree
        
    S = Psql.qstring("SELECT cs.accession, cc.protein_class_id FROM component_sequences cs, component_class cc WHERE cs.component_id = cc.component_id AND cs.accession IS NOT NULL", chembl_dbname)

    class_prot = collections.defaultdict(list)

    for s in S: class_prot[s[0]] += [s[1]]
        
    classes = set([x for k,v in class_prot.iteritems() for x in v])
    class_paths = collections.defaultdict(set)
    for c in classes:
        path = set()
        p = nx.all_simple_paths(G, 0, c)
        for sp in p:
            path.update(sp)
        class_paths[c] = path
            
    T = set()
    for r in R:
        T.update([r])
        if r[1] not in class_prot: continue
        path = set()
        for c in class_prot[r[1]]:
            path.update(class_paths[c])
        for p in path:
            T.update([(r[0], "Class:%d" % p,inchikey_inchi[r[0]])])

    return T


def insert_to_database(T):

    inchikey_raw = collections.defaultdict(list)
    inchikey_inchi = {}
    for t in T:
        inchikey_raw[t[0]] += [t[1]]
        inchikey_inchi[t[0]] = t[2]

    inchikey_raw = dict((k, ",".join([x for x in v])) for k,v in inchikey_raw.iteritems())

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

    global chembl_dbname, dbname,chembl_molrepo
    checkercfg = checkerconfig.checkerConf( configFilename)  
    chembl_dbname = checkerconfig.chembl

    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
   
    chembl_molrepo = os.path.join(checkercfg.getDirectory( "molRepo" ),"chembl.tsv")
    
    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)

    log.info(   "Parsing ChEMBL")

    T = parse_chembl()

    log.info(   "Insert to database")

    inserting_to_database(T)


if __name__ == '__main__':

    main()