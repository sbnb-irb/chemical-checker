'''

Pathways mapped from binding data.

'''

# Imports

import sys, os
sys.path.append(os.path.join(sys.path[0], "../../dbutils/"))
import Psql
import collections
import numpy as np
import csv

# Variables

id_conversion    = "XXXX" # Metaphors - id_conversion.txt
9606_file        = "XXXX" # Metaphors - 9606.txt
human_proteome   = "XXXX" # Human proteome - human_proteome.tab
uniprot2reactome = "XXXX" # Uniprot to reactome - UniProt2Reactome_All_Levels.txt"
db               = Psql.mosaic

table = "pathways"

# Functions

def human_metaphors():

    metaphorsid_uniprot = collections.defaultdict(set)
    f = open(id_conversion, "r")
    f.next()
    dbs = set()
    for l in f:
        l = l.rstrip("\n").split("\t")
        if l[1] == "SwissProt" or l[1] == "TrEMBL":
            metaphorsid_uniprot[l[2]].update([l[0]])
    f.close()

    any_human = collections.defaultdict(set)
    f = open(9606_file, "r")
    f.next()
    for l in f:
        l = l.rstrip("\n").split("\t")
        if l[3] not in metaphorsid_uniprot: continue
        if l[1] not in metaphorsid_uniprot: continue
        for po in metaphorsid_uniprot[l[3]]:
            for ph in metaphorsid_uniprot[l[1]]:
                any_human[po].update([ph])
                any_human[ph].update([ph])
    f.close()

    f = open(human_proteome, "r")
    f.next()
    for l in tqdm(f):
        p = l.split("\t")[0]
        any_human[p].update([p])
    f.close()

    return any_human


def fetch_binding(any_human):

    R = Psql.qstring("SELECT inchikey, raw FROM binding", db)

    ACTS = collections.defaultdict(list)
    for r in R:
        for x in r[1].split(","):
            uniprot_ac, act = x.split("(")
            act = int(act.split(")")[0])
            if uniprot_ac not in any_human: continue
            hps = any_human[uniprot_ac]
            
            for hp in hps:    
                ACTS[(r[0], hp)] += [act]
    ACTS = dict((k, np.max(v)) for k,v in ACTS.iteritems())

    PWYS = collections.defaultdict(list)
    for k,v in ACTS.iteritems():
        if k[1] not in uniprot_reactome: continue
        for pwy in uniprot_reactome[k[1]]:
            PWYS[(k[0], pwy)] += [v]
    PWYS = dict((k, np.max(v)) for k,v in PWYS.iteritems())

    return PWYS


def insert_to_database(PWYS):

    inchikey_raw = collections.defaultdict(list)
    for k,v in PWYS.iteritems():
        inchikey_raw[k[0]] += [(k[1], v)]

    inchikey_raw = dict((k, ",".join(["%s(%d)" % (x[0], x[1]) for x in v])) for k,v in inchikey_raw.iteritems())

    Psql.insert_raw(table, inchikey_raw)


# Main

def main()

    print "Reading human MetaPhors"
    any_human = human_metaphors()

    print "Fetching binding data"
    PWYS = fetch_binding(any_human)

    print "Inserting to database"
    insert_to_database(PWYS)


if __name__ == '__main__':
    main()
