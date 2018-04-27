#!/usr/bin/python

# Do the essential molrepos (DrugBank, ChEMBL, etc.)

# Imports

import sys, os
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
import htstandardizer as hts
import Psql
import csv
import pybel
import urllib2

import checkerconfig


downloadsdir = ""
moldir = ""
# Filenames are the same as the function names (e.g. drugbank() -> drugbank.tsv)

# Molrepo functions (ordered alphabetically)

def bindingdb():

    f = open(os.path.join(downloadsdir,checkerconfig.bindingdb_download), "r")
    g = open(os.path.join(moldir,"bindingdb.tsv"), "w")
    header = f.next()
    header = header.rstrip("\n").split("\t")
    bdlig_idx = header.index("Ligand InChI Key")
    smiles_idx = header.index("Ligand SMILES")
    dones = set()
    for l in f:
        l = l.rstrip("\n").split("\t")
        Id = l[bdlig_idx]
        smi   = l[smiles_idx]
        if Id in dones or not smi: continue
        mol = hts.apply(smi)
        if not mol:
            inchikey = ""
            inchi = ""
        else:
            inchikey = mol[0]
            inchi = mol[1]
        g.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))
        dones.update([Id])


def chebi():
    import rdkit.Chem as Chem
    print os.path.join(downloadsdir,checkerconfig.chebi_lite_download)
    with open(os.path.join(moldir,"chebi.tsv"), "w") as f:
        suppl = Chem.SDMolSupplier(os.path.join(downloadsdir,checkerconfig.chebi_lite_download))
        for m in suppl:
            if not m: continue
            Id = m.GetPropsAsDict()['ChEBI ID']
            smi = Chem.MolToSmiles(m)
            mol = hts.apply(smi)
            if not mol:
                inchikey = ""
                inchi = ""
            else:
                inchikey = mol[0]
                inchi = mol[1]
            f.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))            


def chembl():
    
    with open(moldir+"/chembl.tsv", "w") as f:

        query = "SELECT md.chembl_id, cs.canonical_smiles FROM molecule_dictionary md, compound_structures cs WHERE md.molregno = cs.molregno AND cs.canonical_smiles IS NOT NULL"

        con = Psql.connect(checkerconfig.chembl)
        con.set_isolation_level(0)
        cur = con.cursor()
        cur.execute(query)
        for r in cur:
            Id = r[0]
            smi = r[1]
            mol = hts.apply(smi)
            if not mol:
                inchikey = ""
                inchi = ""
            else:
                inchikey = mol[0]
                inchi = mol[1]
            f.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))


def ctd():
    f = open(os.path.join(downloadsdir,checkerconfig.ctd_molecules_download), "r")
    g = open(moldir + "/ctd.tsv", "w")
    for l in csv.reader(f, delimiter = "\t"):
        if len(l) < 2: continue
        Id = l[0]
        smi = l[1]
        mol = hts.apply(smi)
        if not mol:
            inchikey = ""
            inchi = ""
        else:
            inchikey = mol[0]
            inchi = mol[1]
        g.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))
    g.close()
    f.close()


def drugbank():

    # Parse Drugbank and convert to inchikeys.

    import xml.etree.ElementTree as ET

    xmlfile = os.path.join(downloadsdir,checkerconfig.drugbank_download)

    prefix = "{http://www.drugbank.ca}"

    tree = ET.parse(xmlfile)

    root = tree.getroot()
    
    with open(moldir+"/drugbank.tsv", "w") as f:
    
        for drug in root:

            # Drugbank ID
        
            db_id = None
            for child in drug.findall(prefix + "drugbank-id"):
                if "primary" in child.attrib:
                    if child.attrib["primary"] == "true":
                        db_id = child.text

            if not db_id: continue

            # Smiles
        
            smiles = None
            for props in drug.findall(prefix + "calculated-properties"):
                for prop in props:
                    if prop.find(prefix + "kind").text == "SMILES":
                        smiles = prop.find(prefix + "value").text
            if not smiles: continue
      
            smi = smiles
            Id  = db_id

            mol = hts.apply(smi)
            if not mol:
                inchikey = ""
                inchi = ""
            else:
                inchikey = mol[0]
                inchi = mol[1]
            f.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))


def kegg():
    with open(moldir + "/kegg.tsv", "w") as f:
        L = os.listdir(os.path.join(downloadsdir,checkerconfig.kegg_mol_folder_download))
        for l in L:
            mol = pybel.readfile("mol", os.path.join(downloadsdir,checkerconfig.kegg_mol_folder_download) + "/" + l)
            for m in mol:
                smi = m.write("smi").rstrip("\n")
                if ".mol" not in l: continue
                Id = l.split(".")[0]
                if not smi: continue
                mol = hts.apply(smi)
                if not mol:
                    inchikey = ""
                    inchi = ""
                else:
                    inchikey = mol[0]
                    inchi = mol[1]
                f.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))

                
def lincs():

    S = set()

    with open(os.path.join(downloadsdir,checkerconfig.lincs_GSE92742_pert_info_download), "r") as f:
        f.next()
        for r in csv.reader(f, delimiter = "\t"):
            if not r[1] or r[1] == "-666": continue
            S.update([(r[0], r[1])])

    with open(os.path.join(downloadsdir,checkerconfig.lincs_GSE70138_pert_info_download), "r") as f:
        f.next()
        for r in csv.reader(f, delimiter = "\t"):
            if not r[6] or r[6] == "-666": continue
            S.update([(r[0], r[6])])

    with open(moldir+"/lincs.tsv", "w") as f:
        for s in sorted(S):
            Id = s[0]
            smi = s[1]
            mol = hts.apply(smi)
            if not mol:
                inchikey = ""
                inchi = ""
            else:
                inchikey = mol[0]
                inchi = mol[1]
            f.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))


def mosaic():

    with open(moldir+"/mosaic.tsv", "w") as f:
        for mol in pybel.readfile("sdf", os.path.join(downloadsdir,checkerconfig.mosaic_all_collections_download)):
            if not mol: continue
            smi, Id = mol.write("can").rstrip("\n").split("\t")
            mol = hts.apply(smi)
            if not mol:
                inchikey = ""
                inchi = ""
            else:
                inchikey = mol[0]
                inchi = mol[1]
            f.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))


def morphlincs():
    f = open(moldir+"/morphlincs.tsv", "w")
    g = open(os.path.join(downloadsdir,checkerconfig.morphlincs_molecules_download), "r")
    g.next()
    for l in csv.reader(g, delimiter = "\t"):
        if not l[7]: continue
        Id = l[0]
        smi = l[7]
        if not smi: continue
        mol = hts.apply(smi)
        if not mol:
            inchikey = ""
            inchi = ""
        else:
            inchikey = mol[0]
            inchi = mol[1]
        f.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))
    g.close()
    f.close()


def nci60():
    f = open(os.path.join(downloadsdir,checkerconfig.nci60_download), "r")
    g = open(moldir+"/nci60.tsv", "w")
    f.next()
    for l in csv.reader(f):
        Id, smi = l[0], l[5]
        if not smi: continue
        mol = hts.apply(smi)
        if not mol:
            inchikey = ""
            inchi = ""
        else:
            inchikey = mol[0]
            inchi = mol[1]
        g.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))
    g.close()
    f.close()


def pdb():

    ligand_inchikey = {}
    inchikey_inchi = {}
    f = open(os.path.join(downloadsdir,checkerconfig.pdb_components_smiles_download), "r")
    g = open(moldir+"/pdb.tsv", "w")
    for l in f:
        l = l.rstrip("\n").split("\t")
        if len(l) < 2: continue
        lig_id = l[1]
        mol = hts.apply(l[0])
        if not mol:
            g.write("%s\t%s\t\t\n" % (lig_id, l[0]))
            continue
        ligand_inchikey[lig_id] = mol[0]
        inchikey_inchi[mol[0]] = mol[1]
        g.write("%s\t%s\t%s\t%s\n" % (lig_id, l[0], mol[0], mol[1]))
    f.close()
    g.close()


def sider():

    with open(os.path.join(downloadsdir,checkerconfig.sider_download), "r") as f:
        S = set()
        for l in f:
            l = l.split("\t")
            S.update([l[1]])

    with open(os.path.join(downloadsdir,checkerconfig.stitch_molecules_download), "r") as f:
        stitch = {}
        f.next()
        for r in csv.reader(f, delimiter = "\t"):
            if r[0] not in S: continue
            stitch[r[0]] = r[-1]

    with  open(moldir + "/sider.tsv", "w") as f:
        for s in list(S):
            Id = s
            smi = stitch[s]
            if not smi: continue
            mol = hts.apply(smi)
            if not mol:
                inchikey = ""
                inchi = ""
            else:
                inchikey = mol[0]
                inchi = mol[1]
            f.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))


def smpdb():

    f = open(os.path.join(downloadsdir,checkerconfig.smpdb_metabolites_download), "r")
    f.next()
    g = open(moldir+"/smpdb.tsv", "w")
    S = set()
    for r in csv.reader(f):
        if not r[12]: continue
        S.update([(r[5], r[12])])
    for s in sorted(S):
        Id = s[0]
        smi = s[1]
        mol = hts.apply(smi)
        if not mol:
            inchikey = ""
            inchi = ""
        else:
            inchikey = mol[0]
            inchi = mol[1]
        g.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))

    f.close()
    g.close()


if __name__ == '__main__':
    
    import argparse
    
    if len(sys.argv) != 3:
        sys.exit(1)
  
    configFilename = sys.argv[2]

    checkercfg = checkerconfig.checkerConf( configFilename)  
    
    downloadsdir = checkercfg.getDirectory( "downloads" )
    moldir = checkercfg.getDirectory( "molRepo" )

    args = dict()
    args["do"] = sys.argv[1]

    if args["do"] == "bindingdb" or args["do"] == "all":
        bindingdb()
    if args["do"] == "chebi" or args["do"] == "all":
        chebi()
    if args["do"] == "chembl" or args["do"] == "all":
        chembl()
    if args["do"] == "ctd" or args["do"] == "all":
        ctd()
    if args["do"] == "drugbank" or args["do"] == "all":
        drugbank()
    if args["do"] == "kegg" or args["do"] == "all":
        kegg()
    if args["do"] == "lincs" or args["do"] == "all":
        lincs()
    if args["do"] == "morphlincs" or args["do"] == "all":
        morphlincs()
    if args["do"] == "mosaic" or args["do"] == "all":
        mosaic()
    if args["do"] == "nci60" or args["do"] == "all":
        nci60()
    if args["do"] == "pdb" or args["do"] == "all":
        pdb()
    if args["do"] == "sider" or args["do"] == "all":
        sider()
    if args["do"] == "smpdb" or args["do"] == "all":
        smpdb()
