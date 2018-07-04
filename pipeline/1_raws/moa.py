#!/miniconda/bin/python


# Imports

import collections
import shelve

import sys, os
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
import Psql

import numpy as np
import networkx as nx
import subprocess

import xml.etree.ElementTree as ET

# Variables

chembl_molrepo   = "XXXX"
drugbank_molrepo = "XXXX"
drugbank_xml     = "XXXX" # db/drugbank.xml
chembl_dbname    = "chembl" # chembl
table            = "moa"
dbname           = ''
# Functions

def parse_chembl(ACTS = None):

    if ACTS is None: ACTS = collections.defaultdict(list)

    dirs = {
        'DEGRADER': -1,
        'CROSS-LINKING AGENT': -1,
        'ANTISENSE INHIBITOR': -1,
        'SEQUESTRING AGENT': -1,
        'DISRUPTING AGENT': -1,
        'CHELATING AGENT': -1,
        'SUBSTRATE': 1,
        'AGONIST': 1,
        'STABILISER': 1,
        'BLOCKER': -1,
        'POSITIVE MODULATOR': 1,
        'PARTIAL AGONIST': 1,
        'NEGATIVE ALLOSTERIC MODULATOR': -1,
        'ACTIVATOR': 1,
        'INVERSE AGONIST': -1,
        'INHIBITOR': -1,
        'ANTAGONIST': -1,
        'POSITIVE ALLOSTERIC MODULATOR': 1
    }

    # Query ChEMBL
    
    R = Psql.qstring('''

    SELECT md.chembl_id, cs.canonical_smiles, ts.accession, m.action_type

    FROM molecule_dictionary md, drug_mechanism m, target_components tc, component_sequences ts, compound_structures cs

    WHERE

    md.molregno = m.molregno AND
    m.tid = tc.tid AND
    ts.component_id = tc.component_id AND
    m.molregno = cs.molregno AND

    ts.accession IS NOT NULL AND
    m.action_type IS NOT NULL AND
    cs.canonical_smiles IS NOT NULL

    ''', chembl_dbname)

    # Read molrepo file

    chemblid_inchikey = {}
    with open(chembl_molrepo, "r") as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            if not l[2]: continue
            chemblid_inchikey[l[0]] = l[2]

    # Iterate over results

    for r in R:
        if r[0] not in chemblid_inchikey: continue
        inchikey = chemblid_inchikey[r[0]]
        uniprot_ac = r[2]
        if r[3] not in dirs: continue
        act = dirs[r[3]]
        ACTS[(inchikey, uniprot_ac)] += [act]

    def decide(acts):
        m = np.mean(acts)
        if m > 0:
            return 1
        else:
            return -1

    ACTS = dict((k, decide(v)) for k,v in ACTS.iteritems())

    return ACTS


def parse_drugbank(ACTS = None):

    if ACTS is None: ACTS = collections.defaultdict(list)

    dirs = {
     'Inhibitor': -1,
     'acetylation': -1,
     'activator': +1,
     'agonist': +1,
     'antagonist': -1,
     'binder': -1,
     'binding': -1,
     'blocker': -1,
     'cofactor': +1,
     'inducer': +1,
     'inhibitor': -1,
     'inhibitor, competitive': -1,
     'inhibitory allosteric modulator': -1,
     'intercalation': -1,
     'inverse agonist': +1,
     'ligand': -1,
     'negative modulator': -1,
     'partial agonist': +1,
     'partial antagonist': -1,
     'positive allosteric modulator': +1,
     'positive modulator': +1,
     'potentiator': +1,
     'stimulator': -1,
     'suppressor': -1}

    # Parse the molrepo

    dbid_inchikey = {}
    for l in open(drugbank_molrepo, "r"):
        l = l.rstrip("\n").split("\t")
        if not l[2]: continue
        dbid_inchikey[l[0]] = l[2]
    
    # Parse DrugBank

    prefix = "{http://www.drugbank.ca}"    

    tree = ET.parse(drugbank_xml)

    root = tree.getroot()

    DB = {}

    for drug in root:

        # Drugbank ID
        
        db_id = None
        for child in drug.findall(prefix + "drugbank-id"):
            if "primary" in child.attrib:
                if child.attrib["primary"] == "true":
                    db_id = child.text

        if db_id not in dbid_inchikey: continue
        inchikey = dbid_inchikey[db_id]
        
        # Targets
        
        targets = collections.defaultdict(list)
        
        for targs in drug.findall(prefix + "targets"):
            for targ in targs.findall(prefix + "target"):
                
                # Actions
                
                actions = []
                for action in targ.findall(prefix + "actions"):
                    for child in action:
                        actions += [child.text]
        
                # Uniprot AC
            
                uniprot_ac = None
                prot = targ.find(prefix + "polypeptide")
                if not prot: continue
                if "source" in prot.attrib:
                    if prot.attrib["source"] == "Swiss-Prot":
                        uniprot_ac = prot.attrib["id"]
                if not uniprot_ac: continue
                
                targets[uniprot_ac] = actions
                
        if not targets:
            continue
            
        DB[inchikey] = targets

    # Save activities

    for inchikey, targs in DB.iteritems():
        for uniprot_ac, actions in targs.iteritems():
            if (inchikey, uniprot_ac) in ACTS: continue
            d = []
            for action in actions:
                if action in dirs:
                    d += [dirs[action]]
            if not d: continue
            act = decide(d)
            ACTS[(inchikey, uniprot_ac)] = act

    return ACTS


def put_hierarchy(ACTS):

    R = Psql.qstring("SELECT protein_class_id, parent_id, pref_name FROM protein_classification", chembl_dbname)

    G = nx.DiGraph()

    for r in R: G.add_edge(r[1], r[0]) # The tree
        
    R = Psql.qstring("SELECT cs.accession, cc.protein_class_id FROM component_sequences cs, component_class cc WHERE cs.component_id = cc.component_id AND cs.accession IS NOT NULL", chembl_dbname)

    class_prot = collections.defaultdict(list)

    for r in R: class_prot[r[0]] += [r[1]]

    classACTS = {}

    for k,v in ACTS.iteritems():
        classACTS[k] = v
        if k[1] not in class_prot: continue
        path = set()
        for x in class_prot[k[1]]:
            p = nx.all_simple_paths(G, 0, x)
            for sp in p:
                path.update(sp)
        for p in path:
            classACTS[(k[0], "Class:%d" % p)] = v

    return classACTS


def insert_to_database(ACTS):

    RAW = collections.defaultdict(list)
    for k,v in ACTS.iteritems():
        RAW[k[0]] += [k[1] + "(%s)" % v]

    inchikey_raw = {}
    for k,v in RAW.iteritems():
        inchikey_raw[k] = ",".join(v)

    Psql.insert_raw(table, inchikey_raw, dbname)


# Main

def main():
    
    import argparse
    
    if len(sys.argv) != 2:
        sys.exit(1)
  
    configFilename = sys.argv[1]

    checkercfg = checkerconfig.checkerConf( configFilename)  
    
    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    chembl_dbname = checkerconfig.chembl
    chembl_molrepo   = "XXXX"
    drugbank_molrepo = "XXXX"
    
    downloadsdir = checkercfg.getDirectory( "downloads" )
    moldir = checkercfg.getDirectory( "molRepo" )
    drugbank_xml = os.path.join(downloadsdir,checkerconfig.drugbank_download)
    chembl_molrepo = checkercfg.getDirectory( "molRepo" )
    
    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)


    log.info(  "Parsing ChEMBL")
    ACTS = parse_chembl()
    
    log.info( "Parsing DrugBank")
    ACTS = parse_drugbank(ACTS)
    
    log.info( "Putting target hierarchy")
    ACTS = put_hierarchy(ACTS)

    log.info( "Inserting to database")
    insert_to_database(ACTS)


if __name__ == "__main__":
	main()