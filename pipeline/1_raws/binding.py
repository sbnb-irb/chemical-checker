#!/miniconda/bin/python

'''
# ChEMBL and BindingDB

Get binding data from ChEMBL and BindingDB.

Use the same restrictions that https://pharos.nih.gov/idg/index

ChEMBL ligand activities are selected for a given target are identified using the following criteria:

* must have a pchembl value (ie a -Log M value)
* must be from a binding assay
* must have a MOL structure type
* must have a target type of SINGLE_PROTEIN
* must have standard_flag = 1 and and exact standard_relation (ie. no <= 10uM type values)
* must be associated with a publication
* must pass the family-specific thresholds (Kinases: <= 30nM; GPCRs: <= 100nM; Nuclear Receptors: <= 100nM; Ion Channels: <= 10μM; Others: <= 1uM)

'''

# Imports

import numpy as np
import collections

import sys, os
sys.path.append(os.path.join(sys.path[0], "../../dbutils/"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck
import Psql
import math

import networkx as nx

import checkerconfig


# Variables

chembl_molrepo    = "XXXX"
bindingdb_molrepo = "XXXX"
chembl_dbname     = "XXXX"
bindingdb_file    = "XXXX" # BindingDB_All.tsv
dbname           = ''

table = "binding"

# Functions

def parse_chembl(ACTS = None):

    if ACTS is None: ACTS = collections.defaultdict(list)

    # Read molrepo file

    chemblid_inchikey = {}
    with open(chembl_molrepo, "r") as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            if not l[2]: continue
            chemblid_inchikey[l[0]] = l[2]

    # Query ChEMBL

    con = Psql.connect(chembl_dbname)
    con.set_isolation_level(0)
    cur = con.cursor()
    cur.execute('''
    SELECT md.chembl_id, cseq.accession, act.pchembl_value

    FROM molecule_dictionary md, activities act, assays ass, component_sequences cseq, target_components t, target_dictionary td

    WHERE

    act.molregno = md.molregno AND
    act.assay_id = ass.assay_id AND
    act.standard_relation = '=' AND
    act.standard_flag = 1 AND
    ass.assay_type = 'B' AND
    ass.tid = t.tid AND
    ass.tid = td.tid AND
    td.target_type = 'SINGLE PROTEIN' AND
    t.component_id = cseq.component_id AND
    cseq.accession IS NOT NULL AND
    act.pchembl_value >= 5''')
    ACTS = collections.defaultdict(list)
    for r in cur:
        chemblid = r[0]
        if chemblid not in chemblid_inchikey: continue
        ACTS[(chemblid_inchikey[chemblid], r[1])] += [r[2]]
    con.close()

    return ACTS


def parse_bindingdb(ACTS = None):

    if ACTS is None: ACTS = collections.defaultdict(list)

    def pchemblize(act):
        try:
            act = act / 1e9
            return -math.log10(act)
        except:
            return None

    def activity(ki, ic50, kd, ec50):
        def to_float(s):
            s = s.replace("<", "")
            if s == '' or ">" in s or ">" in s:
                return []
            else:
                return [float(s)]
        acts = []
        acts += to_float(ki)
        acts += to_float(ic50)
        acts += to_float(kd)
        acts += to_float(ec50)
        if acts:
            pchembl = pchemblize(np.min(acts))
            if pchembl < 5: return None
            return pchembl
        return None    

    # Molrepo

    bdlig_inchikey = {}
    f = open(bindingdb_molrepo, "r")
    for l in f:
        l = l.rstrip("\n").split("\t")
        if not l[2]: continue
        bdlig_inchikey[l[0]] = l[2]
    f.close()

    # Read header of BindingDB

    f = open(bindingdb_file, "r")

    header = f.next()
    header = header.rstrip("\n").split("\t")
    bdlig_idx = header.index("Ligand InChI Key")
    smiles_idx = header.index("Ligand SMILES")
    ki_idx = header.index("Ki (nM)")
    ic50_idx = header.index("IC50 (nM)")
    kd_idx = header.index("Kd (nM)")
    ec50_idx = header.index("EC50 (nM)")
    uniprot_ac_idx = header.index("UniProt (SwissProt) Primary ID of Target Chain")
    nchains_idx = header.index("Number of Protein Chains in Target (>1 implies a multichain complex)")

    f.close()

    # Now read activity

    # Now get the activity.

    f = open(bindingdb_file, "r")
    f.next()
    for l in f:
        l = l.rstrip("\n").split("\t")
        nchains = int(l[nchains_idx])
        if nchains != 1: continue
        bdlig = l[bdlig_idx]
        if bdlig not in bdlig_inchikey: continue
        inchikey = bdlig_inchikey[bdlig]
        ki = l[ki_idx]
        ic50 = l[ic50_idx]
        kd = l[kd_idx]
        ec50 = l[ec50_idx]
        act = activity(ki, ic50, kd, ec50)
        if not act: continue
        uniprot_ac = l[uniprot_ac_idx]
        if not uniprot_ac: continue
        for p in uniprot_ac.split(","):
            ACTS[(inchikey, p)] += [act]
    f.close()

    return ACTS


def process_activity_according_to_pharos(ACTS):

    R = Psql.qstring("SELECT protein_class_id, parent_id, pref_name FROM protein_classification", chembl_dbname)

    kinase_idx = [r[0] for r in R if r[2] == 'Kinase']
    gpcr_idx = [r[0] for r in R if r[2] == 'Family A G protein-coupled receptor'
                or r[2] == 'Family B G protein-coupled receptor'
                or r[2] == 'Family C G protein-coupled receptor'
                or r[2] == 'Frizzled family G protein-coupled receptor'
                or r[2] == 'Taste family G protein-coupled receptor']
    nuclear_idx = [r[0] for r in R if r[2] == 'Nuclear receptor']
    ionchannel_idx = [r[0] for r in R if r[2] == 'Ion channel']

    G = nx.DiGraph()

    for r in R: G.add_edge(r[1], r[0]) # The tree
        
    kinase_idx = set([x for w in kinase_idx for k,v in nx.dfs_successors(G, w).iteritems() for x in v] + kinase_idx)
    gpcr_idx = set([x for w in gpcr_idx for k,v in nx.dfs_successors(G, w).iteritems() for x in v] + gpcr_idx)
    nuclear_idx = set([x for w in nuclear_idx for k,v in nx.dfs_successors(G, w).iteritems() for x in v] + nuclear_idx)
    ionchannel_idx = set([x for w in ionchannel_idx for k,v in nx.dfs_successors(G, w).iteritems() for x in v] + ionchannel_idx)

    R = Psql.qstring("SELECT cs.accession, cc.protein_class_id FROM component_sequences cs, component_class cc WHERE cs.component_id = cc.component_id AND cs.accession IS NOT NULL", Psql.chembl)

    class_prot = collections.defaultdict(list)

    for r in R: class_prot[r[0]] += [r[1]]

    # According to Pharos

    cuts = {
        'kinase': -math.log10(30e-9),
        'gpcr': -math.log10(100e-9),
        'nuclear': -math.log10(100e-9),
        'ionchannel': -math.log10(10e-6),
        'other': -math.log10(1e-6)
    }

    protein_cutoffs = collections.defaultdict(list)

    for k,v in class_prot.iteritems():
        for idx in v:
            if idx in ionchannel_idx:
                protein_cutoffs[k] += [cuts['ionchannel']]
            elif idx in nuclear_idx:
                protein_cutoffs[k] += [cuts['nuclear']]
            elif idx in gpcr_idx:
                protein_cutoffs[k] += [cuts['gpcr']]
            elif idx in kinase_idx:
                protein_cutoffs[k] += [cuts['kinase']]
            else:
                protein_cutoffs[k] += [cuts['other']]

    protein_cutoffs = dict((k, np.min(v)) for k,v in protein_cutoffs.iteritems())

    ACTS = dict((k, np.max(v)) for k,v in ACTS.iteritems())

    R = Psql.qstring("SELECT protein_class_id, parent_id, pref_name FROM protein_classification", chembl_dbname)

    G = nx.DiGraph()

    for r in R: G.add_edge(r[1], r[0]) # The tree
        
    R = Psql.qstring("SELECT cs.accession, cc.protein_class_id FROM component_sequences cs, component_class cc WHERE cs.component_id = cc.component_id AND cs.accession IS NOT NULL", chembl_dbname)

    class_prot = collections.defaultdict(list)

    for r in R: class_prot[r[0]] += [r[1]]
        
    classes = set([c for k,v in class_prot.iteritems() for c in v])
    class_path = collections.defaultdict(set)
    for c in classes:
        path = set()
        for sp in nx.all_simple_paths(G, 0, c):
            path.update(sp)
        class_path[c] = path

    classACTS = collections.defaultdict(list)

    for k,v in ACTS.iteritems():
        if k[1] in protein_cutoffs:
            cut = protein_cutoffs[k[1]]
        else:
            cut = cuts['other']
        if v < cut:
            if v < (cut - 1): continue
            V = 1
        else:
            V = 2
        classACTS[k] += [V]
        if k[1] not in class_prot: continue
        for c in class_prot[k[1]]:
            for p in class_path[c]:
                classACTS[(k[0], "Class:%d" % p)] += [V]

    classACTS = dict((k, np.max(v)) for k,v in classACTS.iteritems())

    return classACTS


def insert_to_database(ACTS):

    inchikey_raw = collections.defaultdict(list)
    for k,v in ACTS.iteritems():
        inchikey_raw[k[0]] += [(k[1], v)]

    inchikey_raw = dict((k, ",".join(["%s(%d)" % (x[0], x[1]) for x in v])) for k,v in inchikey_raw.iteritems())

    Psql.insert_raw(table, inchikey_raw,dbname)    


# Main

def main():
    
    import argparse
    
    if len(sys.argv) != 2:
        sys.exit(1)
  
    configFilename = sys.argv[1]

    checkercfg = checkerconfig.checkerConf( configFilename)  
    
    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    chembl_dbname = checkerconfig.chembl

    downloadsdir = checkercfg.getDirectory( "downloads" )
    moldir = checkercfg.getDirectory( "molRepo" )
    ecod_domains = os.path.join(downloadsdir,checkerconfig.eco_domains)
    bindingdb_file =os.path.join(downloadsdir,checkerconfig.bindingdb_download)
    chembl_molrepo = checkercfg.getDirectory( "molRepo" )
    
    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)


    log.info(  " Parsing ChEMBL")

    ACTS = parse_chembl()

    log.info(  " Parsing DrugBank")

    ACTS = parse_drugbank(ACTS)

    log.info(  " Processing activity and assigning target classes")

    ACTS = process_activity_according_to_pharos(ACTS)

    log.info(  " Inserting to database")

    insert_to_database(ACTS)


if __name__ == '__main__':

    main()