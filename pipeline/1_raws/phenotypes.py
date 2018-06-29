'''

Comparative Toxicogenomics Database

'''

# Imports

import sys, os
sys.path.append(os.path.join(sys.path[0], "../../dbutils/"))
import Psql
import collections

# Variables

db = "XXX" # Psql.mosaic
ctd_molrepo = "XXX"
chemdis_file = "XXX" # db/CTD_chemicals_diseases.tsv
disfile = "XXX" # db/CTD_diseases.tsv

table = "phenotypes"

# Functions

def parse_ctd():

    ctd_inchikey = {}
    with open(ctd_molrepo) as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            if not l[2]: continue
            ctd_inchikey[l[0]] = l[2]    

    dis_tree = collections.defaultdict(list)
    f = open(disfile, "r")
    for l in f:
        if l[0] == "#": continue
        l = l.rstrip("\n").split("\t")
        dis_tree[l[1]] = l[5].split("|")
    f.close()

    tree_dis = collections.defaultdict(list)
    for k,v in dis_tree.iteritems():
        for x in v:
            tree_dis[x] += [k]

    def expand_tree(tn):
        tns = []
        x = tn.split("/")[0].split(".")
        for i in xrange(len(x)):
            tns += [".".join(x[:i+1])]
        tns += [tn]
        return tns

    f = open(chemdis_file, "r")
    inchikey_raw = collections.defaultdict(set)

    for l in f:    
        if l[0] == "#": continue
        l = l.rstrip("\n").split("\t")
        if l[5] == "": continue
        dis = l[4]
        cid = l[1]
        if cid not in ctd_inchikey: continue
        inchikey = ctd_inchikey[cid]
        ev = l[5]
        for tn in dis_tree[dis]:
            exp_tns = expand_tree(tn)
            exp_dis = set()
            for exp_tn in exp_tns:
                exp_dis.update(tree_dis[exp_tn])
        exp_dis = sorted(exp_dis)
        for d in exp_dis:
            if not "MESH" in d: continue
            x = []
            if ev == "marker/mechanism":
                x += [d + "(M)"]
            if ev == "therapeutic":
                x += [d + "(T)"]
            for y in x:
                inchikey_raw[inchikey].update([y.split("MESH:")[1]])
    f.close()

    return inchikey_raw


def insert_to_database(inchikey_raw):

    inchikey_raw = dict((k, ",".join(sorted(v))) for k,v in inchikey_raw.iteritems())

    Psql.insert_raw(table, inchikey_raw)
  

# Main

def main():

    print "Parsing CTD..."
    inchikey_raw = parse_ctd()

    print "Inserting to database..."
    insert_to_database(inchikey_raw)


if __name__ == "__main__":

    main()