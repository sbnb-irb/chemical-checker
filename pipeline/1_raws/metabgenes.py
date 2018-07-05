#!/miniconda/bin/python

'''

Metabolic genes.

'''

# Imports

import collections
import shelve

import sys, os
sys.path.append(os.path.join(sys.path[0], "../../dbutils/"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck

import Psql

import numpy as np
import networkx as nx
import subprocess

import xml.etree.ElementTree as ET

import checkerconfig


# Variables

chembl_molrepo   = "XXXX"
drugbank_molrepo = "XXXX"
drugbank_xml     = "XXXX" # db/drugbank.xml
chembl_dbname    = "XXXX" # chembl
table            = "metabgenes"
dbname           = ''


# Functions

def parse_chembl(ACTS = None):

    if ACTS is None: ACTS = collections.defaultdict(set)

    # Query ChEMBL
    
    R = Psql.qstring('''

    SELECT md.chembl_id, cs.accession

    FROM molecule_dictionary md, compound_records cr, metabolism m, target_components tc, component_sequences cs

    WHERE

    md.molregno = cr.molregno AND
    cr.record_id = m.drug_record_id AND
    m.enzyme_tid = tc.tid AND
    tc.component_id = cs.component_id AND
    cs.accession IS NOT NULL

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
        uniprot_ac = r[1]
        ACTS.update([(inchikey, uniprot_ac)])

    return ACTS


def parse_drugbank(ACTS = None):

    if ACTS is None: ACTS = collections.defaultdict(set)    

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

    for drug in root:

        # Drugbank ID
        
        db_id = None
        for child in drug.findall(prefix + "drugbank-id"):
            if "primary" in child.attrib:
                if child.attrib["primary"] == "true":
                    db_id = child.text

        if db_id not in dbid_inchikey: continue
        inchikey = dbid_inchikey[db_id]
        
        # Enzymes, transporters, carriers
        
        targets = collections.defaultdict(list)
        
        for ps in drug.findall(prefix + "enzymes"):
            for p in ps.findall(prefix + "enzyme"):
                    
                # Uniprot AC
            
                uniprot_ac = None
                prot = p.find(prefix + "polypeptide")
                if not prot: continue
                if "source" in prot.attrib:
                    if prot.attrib["source"] == "Swiss-Prot":
                        uniprot_ac = prot.attrib["id"]
                if not uniprot_ac: continue
                
                ACTS.update([(inchikey, uniprot_ac)])
        
        for ps in drug.findall(prefix + "transporters"):
            for p in ps.findall(prefix + "transporter"):
                    
                # Uniprot AC
            
                uniprot_ac = None
                prot = p.find(prefix + "polypeptide")
                if not prot: continue
                if "source" in prot.attrib:
                    if prot.attrib["source"] == "Swiss-Prot":
                        uniprot_ac = prot.attrib["id"]
                if not uniprot_ac: continue
                
                ACTS.update([(inchikey, uniprot_ac)])

        for ps in drug.findall(prefix + "carriers"):
            for p in ps.findall(prefix + "carrier"):
                    
                # Uniprot AC
            
                uniprot_ac = None
                prot = p.find(prefix + "polypeptide")
                if not prot: continue
                if "source" in prot.attrib:
                    if prot.attrib["source"] == "Swiss-Prot":
                        uniprot_ac = prot.attrib["id"]
                if not uniprot_ac: continue
                
                ACTS.update([(inchikey, uniprot_ac)])

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

    log.info( "Parsing ChEMBL")
    ACTS = parse_chembl()
    
    log.info( "Parsing DrugBank")
    ACTS = parse_drugbank(ACTS)
    
    log.info( "Putting target hierarchy")
    ACTS = put_hierarchy(ACTS)

    log.info( "Inserting to database")
    insert_to_database(ACTS)


if __name__ == "__main__":
    main()