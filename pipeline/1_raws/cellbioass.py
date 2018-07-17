#!/miniconda/bin/python


# Imports

import sys, os
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, draw
import Psql
import collections
import networkx as nx

import checkerconfig


# Variables

table = "cellbioass"
dbname = "XXX" # Psql.mosaic
chembl_dbname = "XXX"
cellosaurus_obo = "XXX" # data/cellosaurus.obo

chembl_molrepo = "XXX"


# Functions

def fetch_chembl():

    # ChEMBL - InChIKey

    f = open(chembl_molrepo, "r")
    chemblid_inchikey = {}
    inchikey_inchi = {}
    for l in f:
        l = l.rstrip("\n").split("\t")
        if not l[2]: continue
        chemblid_inchikey[l[0]] = l[2]
        inchikey_inchi[l[2]] = l[3]

    # Query

    con = Psql.connect(chembl_dbname)
    con.set_isolation_level(0)
    cur = con.cursor()
    cur.execute('''
    SELECT md.chembl_id, ass.assay_id, ass.src_id, act.standard_relation, act.standard_type, act.standard_value, act.standard_units, cd.cellosaurus_id 

    FROM molecule_dictionary md, activities act, assays ass, target_dictionary td, cell_dictionary cd

    WHERE

    act.molregno = md.molregno AND
    act.assay_id = ass.assay_id AND
    ass.tid = td.tid AND
    td.target_type = 'CELL-LINE' AND
    ass.cell_id IS NOT NULL AND
    cd.cell_id = ass.cell_id AND
    cd.cellosaurus_id IS NOT NULL AND
    ass.src_id = 1 AND act.standard_flag = 1
    ''')
    #((ass.src_id = 1 AND act.standard_flag = 1) OR (ass.src_id = 7 AND (act.activity_comment = 'Active' OR act.activity_comment = 'active')))
    #''')
    R = []
    for r in cur:
        if r[0] not in chemblid_inchikey: continue
        if r[2] == 1: # Literature
            if r[3] != '=': continue
            if (r[4], r[6]) in [("IC50", "nM"), ("GI50", "nM"), ("LC50", "nM"), ("LD50", "nM"), ("CC50", "nM"), ("EC50", "nM")]:
                if r[5] < 1000:
                    R += [(chemblid_inchikey[r[0]], r[1], r[-1])]
            elif (r[4], r[6]) == ("Activity", "%"):
                if r[5] <= 50:
                    R += [(chemblid_inchikey[r[0]], r[1], r[-1])]
            elif (r[4], r[6]) == ("GI", "%"):
                if r[5] >= 50:
                    R += [(chemblid_inchikey[r[0]], r[1], r[-1])]
        elif r[2] == 7: # PubChem
            R += [(chemblid_inchikey[r[0]], r[1], r[-1])]
        else:
            continue
    con.close()

    return R,inchikey_inchi


def parse_cellosaurus(R):

    f = open(cellosaurus_obo, "r")
    O = f.read().split("[Term]\n")
    f.close()

    G = nx.DiGraph()

    for term in O:
        term = term.split("\n")
        for l in term:
            if l[:4] == "id: ":
                child = l.split("id: ")[1]
                G.add_node(child)
            if "relationship: derived_from " in l:
                parent = l.split("derived_from ")[1].split(" !")[0]
                G.add_edge(parent, child)
            #if "relationship: originate_from_same_individual_as" in l:
            #    parent = l.split("originate_from_same_individual_as ")[1].split(" !")[0]
            #    G.add_edge(parent, child)

    # Add a root *

    for n in list(G.nodes()):
        if not nx.ancestors(G, n):
            G.add_edge(n, "*")

    # Cell hierarchy

    cells = set([r[-1] for r in R])
    cell_hier = collections.defaultdict(set)

    for cell in cells:
        for c in nx.all_simple_paths(G, cell, "*"):
            if c == "*": continue
            for x in c:
                cell_hier[cell].update([x])
    
    return cell_hier


def insert_to_database(R,cell_hier,inchikey_inchi):

    inchikey_raw = collections.defaultdict(set)
    for r in R:
        for c in cell_hier[r[-1]]:
            inchikey_raw[r[0]].update([r[-1]])
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
    
    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    chembl_dbname = checkerconfig.chembl
    global cellosaurus_obo,chembl_molrepo
    
    downloadsdir = checkercfg.getDirectory( "downloads" )
    cellosaurus_obo = os.path.join(downloadsdir,checkerconfig.cellosaurus_obo)
    chembl_molrepo = os.path.join(checkercfg.getDirectory( "molRepo" ),"chembl.tsv")
    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)

    log.info( "Fetch from ChEMBL")
    R,inchikey_inchi = fetch_chembl()

    log.info( "Reading Cellosaurus")
    cell_hier = parse_cellosaurus(R)

    log.info( "Inserting to database")
    insert_to_database(R,cell_hier,inchikey_inchi)


if __name__ == '__main__':
    main()