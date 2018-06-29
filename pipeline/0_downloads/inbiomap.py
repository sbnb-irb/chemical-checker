#!/miniconda/bin/python

'''

InBioMap network from InWeb.

'''

# Imports

import csv
import os, sys
import numpy as np
import glob
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck
sys.path.append(os.path.join(sys.path[0],"../"))

# Variables

inweb = "core.psimitab" # Downloaded from https://www.intomics.com/inbio/map/#downloads

output_folder = "inbiomap" # I recommend that this is called inbiomap

THRESHOLD = 0.2

# Functions

def read_data():
    with open(inweb, "r") as f:
        E = set()
        for r in csv.reader(f, delimiter = "\t"):
            if "uniprotkb:" not in r[0] or "uniprotkb:" not in r[1]: continue
            if "taxid:9606" not in r[9] or "taxid:9606" not in r[10]: continue
            score = np.mean([float(x) for x in r[-2].split("|") if x != "-"])
            if score < THRESHOLD: continue
            ps0 = r[0].split("uniprotkb:")[1].split("|")
            ps1 = r[1].split("uniprotkb:")[1].split("|")
            for p0 in ps0:
                for p1 in ps1:
                    E.update([tuple(sorted([p0, p1]))])
    return E

def write_network(E):
    with open(output_folder + "/interactions.tsv", "w") as f:
        for e in E:
            f.write("%s\t%s\n" % (e[0], e[1]))


# Main

def main():
    
    if len(sys.argv) != 3:
        usage(sys.argv[0])
        sys.exit(1)
  
    configFilename = sys.argv[2]

    checkercfg = checkerconfig.checkerConf(configFilename )  

    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)

    downloadsdir = checkercfg.getDirectory( "downloads" )
    
    inweb =  glob.glob(downloadsdir + '/InBio_Map_core_*/core.psimitab')[0] 
    
    networksdir = checkercfg.getDirectory( "networks" )
    
    output_folder = os.path.join(networksdir,output_folder)
    check_dir = os.path.exists(output_folder)


    if check_dir == False:
        c = os.makedirs(output_folder)
    
    E = read_data()
    write_network(E)


if __name__ == '__main__':
    main()
