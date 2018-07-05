#!/miniconda/bin/python

'''

Molecular roles extracted from ChEBI.

'''

# Imports

import sys, os
import networkx as nx
import collections
sys.path.append(os.path.join(sys.path[0], "../../dbutils/"))
sys.path.append(os.path.join(sys.path[0],"../config"))

import Psql
import checkerconfig

# Variables

chebi_molrepo = "XXXX"
chebi_obo     = "XXXX" # chebi.obo
table = "molroles"
dbname = ''

Role = "CHEBI:50906"

# Functions

def parse_chebi():

    # Get molecules

    inchikey_chebi = collections.defaultdict(set)
    f = open(chebi_molrepo, "r")
    for l in f:
        l = l.rstrip("\n").split("\t")
        if not l[2]: continue
        inchikey_chebi[l[2]].update([l[0]])
    f.close()

    # Parse ChEBI graph

    CHEBI = nx.DiGraph()

    cuts = ["is_a: ",
            "relationship: is_conjugate_acid_of ",
            "relationship: is_conjugate_base_of ",
            "relationship: is_tautomer_of ",
            "relationship: is_enantiomer_of ",
            "relationship: has_role "]

    f = open(chebi_obo, "r")
    terms = f.read().split("[Term]\n")
    for term in terms[1:]:
        term = term.split("\n")
        chebi_id = term[0].split("id: ")[1]
        CHEBI.add_node(chebi_id)
        parents = []
        for l in term[1:]:
            for cut in cuts:
                if cut in l:
                    parent_chebi_id = l.split(cut)[1]
                    parents += [parent_chebi_id]
        parents = sorted(set(parents))
        for p in parents: CHEBI.add_edge(chebi_id, p)
    f.close()

    # Find paths

    inchikey_paths = {}
    for k,v in inchikey_chebi.iteritems():
        path = set()
        for chebi_id in v:
            if chebi_id not in CHEBI.nodes(): continue
            path.update([n for p in nx.all_simple_paths(CHEBI, chebi_id, Role) for n in p])
        path = sorted(path)
        inchikey_paths[k] = path

    return inchikey_paths


def insert_to_database(inchikey_paths):

    inchikey_raw = {}
    for k,v in inchikey_paths.iteritems():
        inchikey = k
        if not v: continue
        raw = ",".join(v)
        inchikey_raw[inchikey] = raw

    Psql.insert_raw(table, inchikey_raw,dbname)


# Main

def main():
    
    import argparse
    
    if len(sys.argv) != 2:
        sys.exit(1)
  
    configFilename = sys.argv[1]

    checkercfg = checkerconfig.checkerConf( configFilename)  
    chembl_dbname = checkerconfig.chembl
    chebi_obo = checkerconfig.chebi_obo

    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
   
    downloadsdir = checkercfg.getDirectory( "downloads" )
    moldir = checkercfg.getDirectory( "molRepo" )
    chembl_molrepo = checkercfg.getDirectory( "molRepo" )
    
    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)



    log.info( "Parsing the ChEBI ontology...")
    inchikey_paths = parse_chebi()

    log.info( "Inserting to database")
    insert_to_database(inchikey_paths)


if __name__ == '__main__':
    main()
