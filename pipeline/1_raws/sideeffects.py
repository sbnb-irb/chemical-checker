'''

Side effects from SIDER.

'''

# Imports

import collections
import sys, os
sys.path.append(os.path.join(sys.path[0], "../../dbutils/"))
import Psql
import networkx as nx
import csv


# Variables

sider_molrepo = "XXX"
sider_file = "XXX" # db/meddra_all_se.tsv
table = "sider"


# Functions

def parse_sider():

    cid_inchikey = {}
    f = open(molrepo_file, "r")
    for l in f:
        l = l.rstrip("\n").split("\t")
        if not l[2]: continue
        cid_inchikey[l[0]] = l[2]
    f.close()

    inchikey_raw = collections.defaultdict(set) 
    with open(sider_file, "r") as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            cid = l[1]
            if cid not in cid_inchikey: continue
            inchikey_raw[cid_inchikey[cid]].update([l[2]])

    return inchikey_raw


def insert_to_database(inchikey_raw):

    inchikey_raw = dict((k, ",".join(sorted(v))) for k,v in inchikey_raw.iteritems())          

    Psql.insert_raw(table, inchikey_raw)


# Main

def main():

    print "Parsing SIDER"
    inchikey_raw = parse_sider()

    print "Inserting to database"
    insert_to_database(inchikey_raw)


if __name__ == "__main__":

    main()