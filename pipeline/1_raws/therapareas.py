#!/miniconda/bin/python

# Imports

import collections
import pybel
import subprocess
import os, sys
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, draw
import Psql

import checkerconfig

# Variables

kegg_molrepo = "XXX"
drugbank_molrepo = "XXX"
br_file = "XXX" # db/br08303.keg
dbname = ''

table = "therapareas"

# Functions

def parse_kegg(inchikey_atc = None):

    if not inchikey_atc:
        inchikey_atc = collections.defaultdict(set)

    # Read molrepo
    kegg_inchikey = {}
    inchikey_inchi = {}
    f = open(molrepo_file, "r")
    for l in f:
        l = l.rstrip("\n").split("\t")
        if not l[2]: continue
        kegg_inchikey[l[0]] = l[2]
        inchikey_inchi[l[2]] = l[3]

    # Read drug KEGG branch
    with open(br_file, "r") as f:
        for l in f:
            if l[0] == "E":
                atc = l.split()[1]
            if l[0] == "F":
                drug = l.split()[1]
                if drug not in kegg_inchikey: continue
                inchikey_atc[kegg_inchikey[drug]].update([atc])

    return inchikey_atc,inchikey_inchi


def parse_drugbank(inchikey_inchi,inhchikey_atc = None):

    if not inchikey_atc:
        inchikey_atc = collections.defaultdict(set)

    # DrugBank molrepo
    dbid_inchikey = {}
    f = open(drugbank_molrepo)
    for l in f:
        l = l.rstrip("\n").split("\t")
        if not l[2]: continue
        dbid_inchikey[l[0]] = l[2]
        inchikey_inchi[l[2]] = l[3]

    # Read DrugBank

    import xml.etree.ElementTree as ET

    xmlfile = "db/drugbank.xml"

    prefix = "{http://www.drugbank.ca}"

    tree = ET.parse(xmlfile)

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
        
        # ATCs
                
        for atcs in drug.findall(prefix + "atc-codes"):
            for atc in atcs:
                inchikey_atc[inchikey].update([atc.attrib["code"]])


def break_atcs(inchikey_atc):

    def break_atc(atc):
        A = "A:%s" % atc[0]
        B = "B:%s" % atc[:3]
        C = "C:%s" % atc[:4]
        D = "D:%s" % atc[:5]
        E = "E:%s" % atc
        return [A, B, C, D, E]

    inchikey_raw = collections.defaultdict(set)
    for k,v in inchikey_atc.iteritems():
        for x in v:
            inchikey_raw[k].update(break_atc(x))

    return inchikey_raw


def insert_to_database(inchikey_raw,inchikey_inchi):

    inchikey_raw = dict((k,",".join(sorted(v))) for k,v in inchikey_raw.iteritems())
    
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
    global br_file,kegg_molrepo,drugbank_molrepo
    
    downloadsdir = checkercfg.getDirectory( "downloads" )
    br_file = os.path.join(downloadsdir,checkerconfig.kegg_atcs_download)
    kegg_molrepo = os.path.join(checkercfg.getDirectory( "molRepo" ),"kegg.tsv")
    drugbank_molrepo = os.path.join(checkercfg.getDirectory( "molRepo" ),"drugbank.tsv")
    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)


    log.info( "Parsing KEGG...")
    inchikey_atc,inchikey_inchi = parse_kegg()

    log.info( "Parsing DrugBank...")
    inchikey_atc = parse_drugbank(inchikey_inchi,inchikey_atc)

    log.info( "Breaking ATCs...")
    inchikey_raw = break_atcs(inchikey_atc)

    log.info( "Inserting to database...")
    insert_to_database(inchikey_raw,inchikey_inchi)


if __name__ == "__main__":
    main()