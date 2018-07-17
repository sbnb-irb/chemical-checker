#!/miniconda/bin/python


# Imports

import sys, os
import collections
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, draw
import Psql

import checkerconfig

# Variables

dbname = ''
drugbank_molrepo = "XXX"
ddi_file = "XXX" # db/ddi.tsv I CANNOT RECALL WHERE DID I GET THIS FILE FROM! PROBABLY FROM DRUGBANK ITSELF (A PREVIOUS VERSION) IN THE NEW VERSION WE HAVE TO DO IT AGAIN. LET'S TALK ABOUT IT.

table = "ddis"

# Functions

def parse_ddis():

	f = open(drugbank_molrepo, "r")
	dbid_inchikey = {}
	inchikey_inchi = {}
	for l in f:
	    l = l.rstrip("\n").split("\t")
	    if not l[2]: continue
	    dbid_inchikey[l[0]] = l[2]
	    inchikey_inchi[l[2]] = l[3]   
	f.close()

	inchikey_ddi = collections.defaultdict(list)
	f = open(ddi_file, "r")
	for l in f:
	    l = l.split("\t")
	    if l[0] not in dbid_inchikey: continue
	    inchikey_ddi[dbid_inchikey[l[0]]] += [l[1]]
	f.close()

	return inchikey_ddi,inchikey_inchi

def insert_to_database(inchikey_ddi,inchikey_inchi):

	inchikey_raw = dict((k, ",".join(v)) for k,v in inchikey_ddi.iteritems())
	
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
    global ddi_file,drugbank_molrepo
    
    downloadsdir = checkercfg.getDirectory( "downloads" )
    ddi_file = os.path.join(downloadsdir,checkerconfig.drugbank_download)
    drugbank_molrepo = os.path.join(checkercfg.getDirectory( "molRepo" ),"drugbank.tsv")
    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)

    log.info(  "Parsing DDIs...")
    inchikey_ddi,inchikey_inchi = parse_ddis()

    log.info(  "Inserting to database...")
    insert_to_database(inchikey_ddi,inchikey_inchi)


if __name__ == "__main__":

	main()
