'''

Drug-drug interactions from DrugBank.

'''

# Imports

import sys, os
import collections
sys.path.append(os.path.join(sys.path[0], "../../dbutils/"))
import Psql


# Variables

drugbank_molrepo = "XXX"
ddi_file = "XXX" # db/ddi.tsv I CANNOT RECALL WHERE DID I GET THIS FILE FROM! PROBABLY FROM DRUGBANK ITSELF (A PREVIOUS VERSION) IN THE NEW VERSION WE HAVE TO DO IT AGAIN. LET'S TALK ABOUT IT.

table = "ddis"

# Functions

def parse_ddis():

	f = open(drugbank_molrepo, "r")
	dbid_inchikey = {}
	for l in f:
	    l = l.rstrip("\n").split("\t")
	    if not l[2]: continue
	    dbid_inchikey[l[0]] = l[2]
	f.close()

	inchikey_ddi = collections.defaultdict(list)
	f = open(ddi_file, "r")
	for l in f:
	    l = l.split("\t")
	    if l[0] not in dbid_inchikey: continue
	    inchikey_ddi[dbid_inchikey[l[0]]] += [l[1]]
	f.close()

	return inchikey_ddi

def insert_to_database(inchikey_ddi):

	inchikey_raw = dict((k, ",".join(v)) for k,v in inchikey_ddi.iteritems())

	Psql.insert_raw(table, inchikey_raw)


# Main

def main():

    print "Parsing DDIs..."
    inchikey_ddi = parse_ddis()

    print "Inserting to database..."
    insert_to_database(inchikey_ddi)


if __name__ == "__main__":

	main()