'''

Chad Myers Chemical Genetics data.

'''

# Imports

import sys, os
sys.path.append(os.path.join(sys.path[0], "../../dbutils/"))
sys.path.append(os.path.join(sys.path[0], "../../mlutils/"))
from gaussian_scale_impute import scaleimpute
import Psql
import numpy as np
import collections

# Variables

db = Psql.mosaic

mosaic_molrepo = "XXX" # mosaic.tsv"
all_conditions = "XXX" # data/All_conditions.txt
comb_gt_preds  = "XXX" # data/combined_gene-target-predictions.txt
table = "chemgenet"

pval_01  = 3.37
pval_001 = 7.12

# Functions

def read_mosaic_predictions():

    with open(mosaic_molrepo, "r") as f:
        cgid_inchikey = {}
        for l in f:
            l = l.rstrip("\n").split("\t")
            if not l[2]: continue
            cgid_inchikey[l[0]] = l[2]

    conds = {}
    f = open(all_conditions, "r")
    H = f.next().split("\t")[1:]
    f.close()

    for h in H:
        cond, Id = h.split("_")
        if Id not in cgid_inchikey: continue
        conds[cond] = cgid_inchikey[Id]

    sig = collections.defaultdict(set)
    with open(comb_gt_preds, "r") as f:
        f.next()
        for l in f:
            l = l.rstrip("\n").split("\t")
            if l[0] not in conds: continue
            ik = conds[l[0]]
            s = float(l[2])
            if s > pval_01: # P-value 0.01
                if s > pval_001: # P-value 0.001
                    d = 2
                else:
                    d = 1
                sig[ik].update(["%s(%d)" % (l[1], d)])

    return sig


def insert_to_database(sig):
    
    inchikey_raw = dict((k, ",".join(sorted(v))) for k,v in sig.iteritems())

    Psql.insert_to_database(table, inchikey_raw)


# Main

def main():

    print "Parsing Chad Myers' Mosaic"
    sig = read_mosaic_predictions()

    print "Inserting to database"
    insert_to_database(sig)

if __name__ == '__main__':
    main()
