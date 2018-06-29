'''
NCI60 data
'''

# Imports

import sys, os
sys.path.append(os.path.join(sys.path[0], "../../dbutils/"))
sys.path.append(os.path.join(sys.path[0], "../../mlutils/"))
from gaussian_scale_impute import scaleimpute
import Psql
import numpy as np
import pybel
import collections
import subprocess
import csv

# Variables

db = Psql.mosaic
nci60_molrepo = "XXXX" # nci60.tsv
dtp_data = "XXXX" # "data/DTP_NCI60_ZSCORE.csv"
table = "cellpanel"

# Functions

def parse_nci60():

    with open(nci60_molrepo, "r") as f:
        nci_inchikey = {}
        for l in f:
            l = l.rstrip("\n").split("\t")
            if not l[2]: continue
            nci_inchikey[l[0]] = l[2]

    # Read the NCI60 data

    def Float(x):
        try:
            return float(x)
        except:
            return np.nan

    def count_nans(v):
        return len([1 for x in v if np.isnan(x)])

    def find_strongest_signature(v):
        if len(v) == 1: return v[0]
        my_i = 0
        my_nonan = 0
        for i in xrange(len(v)):
            nonan = len([1 for x in v[i] if not np.isnan(x)])
            if nonan > my_nonan:
                my_i = i
                my_nonan = nonan
        return v[my_i]

    sigs = collections.defaultdict(list)

    with open(dtp_data, "r") as f:
        f.next()
        for l in csv.reader(f):
            if l[0] not in nci_inchikey: continue
            inchikey = nci_inchikey[l[0]]
            v = [Float(x) for x in l[6:-2]]
            sigs[inchikey] += [v]

    sigs = dict((k, find_strongest_signature(v)) for k,v in sigs.iteritems())
    sigs = dict((k, v) for k,v in sigs.iteritems() if count_nans(v) < 10)

    # Scale the signatures, and impute

    rowNames = []
    X_incomplete = []
    for k,v in sigs.iteritems():
        rowNames += [k]
        X_incomplete += [v]
    X_incomplete = np.array(X_incomplete)

    # Scale and impute

    X = scaleimpute(X_incomplete)

    return X, rowNames


def insert_to_database(X, rowNames):

    inchikey_raw = {}
    for i in xrange(len(rowNames)):
        inchikey_raw[rowNames[i]] = ",".join(["%.5f" % x for x in X[i]])

    Psql.insert_raw(table, inchikey_raw)


# Main

def main():

    print "Parsing NCI-60"
    X, rowNames = parse_nci60()

    print "Inserting to database"
    insert_to_database(X, rowNames)


if __name__ == '__main__':
    main()