
# Imports

import numpy as np
import sys, os
import collections

sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../../src/mlutils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck

import clust

# Main

if __name__ == '__main__':
    
    table = sys.argv[1]
    if sys.argv[2].strip()  == 'None':
        balance = None
    else:
        balance = float(sys.argv[2])
    

    tempdir = sys.argv[3]
    filesdir = sys.argv[4]
    log = logSystem(sys.stdout)
    log.debug(os.getcwd())
    
    clust.clustering(table = table,balance = balance,filesdir = filesdir,log = log,tmpDir = tempdir)