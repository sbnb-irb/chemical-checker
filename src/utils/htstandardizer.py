# -*- coding: utf-8 -*-

from standardiser import standardise
from rdkit.Chem import AllChem as Chem
from rdkit import RDLogger

import logging
# Iterate

def apply(smi):
    
    lg = RDLogger.logger()

    lg.setLevel(RDLogger.CRITICAL)
    
    logging.getLogger(standardise.__name__).setLevel(logging.ERROR)


    mol = standardise.Chem.MolFromSmiles(smi)
    if not mol: return None
    try:
        mol = standardise.run(mol)
    except:
        return None
    inchi = Chem.rdinchi.MolToInchi(mol)[0]
    inchikey = Chem.rdinchi.InchiToInchiKey(inchi)
    if not inchi or not inchikey:
        return None
    try:
        mol = Chem.rdinchi.InchiToMol(inchi)[0]
    except:
        return None
    return inchikey, inchi


if __name__ == "__main__":

    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type = str, help = "Input file")
    parser.add_argument("--outfile", type = str, default = None, help = "Output file")
    parser.add_argument("--idcol", type = int, default = 0, help = "Id column")
    parser.add_argument("--smicol", type = int, default = 1, help = "Smiles column")
    parser.add_argument("--delimiter", type = str, default = "\t", help = "Delimiter")
    parser.add_argument("--has_header", default = False, action = "store_true", help = "Has header?")

    args = parser.parse_args()

    if not args.outfile: args.outfile = args.infile + ".std"

    f = open(args.infile, "r")
    g = open(args.outfile, "w")
    if args.has_header: f.next()
    for l in f:
        l = l.rstrip("\n").split(args.delimiter)
        Id = l[args.idcol]
        smi = l[args.smicol]
        mol = apply(smi)
        if not mol:
            inchikey = ""
            inchi = ""
        else:
            inchikey = mol[0]
            inchi = mol[1]
        g.write("%s\t%s\t%s\t%s\n" % (Id, smi, inchikey, inchi))
    g.close()
    f.close()
