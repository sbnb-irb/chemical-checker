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



# Main

def main():
	
    import argparse
    
    if len(sys.argv) != 2:
    	sys.exit(1)
  
    configFilename = sys.argv[1]

    checkercfg = checkerconfig.checkerConf( configFilename)  
    
    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    
    filesdir = checkercfg.getDirectory( "files_validations" )
   
    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)
    
    if not os.path.exists(filesdir):
        os.makedirs(filesdir)

    log.info(  "Createing ATC validation file...")
    R = Psql.qstring("SELECT inchikey, raw FROM therapareas", dbname)

    d = {}
    for r in R:
        d[r[0]] = set([x.split(":")[1] for x in r[1].split(",") if "C:" in x])
    
    root = {}
    for r in R:
        root[r[0]] = set([x.split(":")[1] for x in r[1].split(",") if "A:" in x])
    
    keys = sorted(d.keys())
    
    f = open(os.path.join(filesdir,"atc_validation.tsv"), "w")
    for i in xrange(len(keys) - 1):
        for j in range(i+1, len(keys)):
            if keys[i].split("-")[0] == keys[j].split("-")[0]: continue
            com = len(d[keys[i]].intersection(d[keys[j]]))
            if com > 0:
                v = 1
            else:
                rootcom = len(root[keys[i]].intersection(root[keys[j]]))
                if rootcom == 0:
                    v = 0
                else:
                    continue
            f.write("%s\t%s\t%d\n" % (keys[i], keys[j], v))
    f.close()
    
    log.info(  "Creating MOA validation file...")
    R = Psql.qstring("SELECT inchikey, raw FROM moa", dbname)

    d = {}
    for r in R:
        d[r[0]] = set([x for x in r[1].split(",") if "Class" not in x])
    
    keys = sorted(d.keys())
    
    f = open(os.path.join(filesdir,"moa_validation.tsv"), "w")
    for i in xrange(len(keys) - 1):
        for j in range(i+1, len(keys)):
            if keys[i].split("-")[0] == keys[j].split("-")[0]: continue
            com = len(d[keys[i]].intersection(d[keys[j]]))
            if com > 0:
                v = 1
            else:
                v = 0
            f.write("%s\t%s\t%d\n" % (keys[i], keys[j], v))
    f.close()


if __name__ == "__main__":

	main()
