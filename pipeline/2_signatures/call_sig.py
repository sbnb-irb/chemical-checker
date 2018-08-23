
# Imports

import numpy as np
import sys, os
import collections

sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../../src/signatures"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck

import sig

# Main

if __name__ == '__main__':
    
    dbname = sys.argv[1]
    table = sys.argv[2]
    if sys.argv[3].strip()  == 'None':
        num_topics = None
    else:
        num_topics = int(sys.argv[3])
    if sys.argv[4].strip()  == 'None':
        max_freq = None
    else:
        max_freq = float(sys.argv[4])

    multipass = bool(sys.argv[5])
    tempdir = sys.argv[6]
    log = logSystem(sys.stdout)
    log.debug(os.getcwd())
    
    sig.generate_signatures(dbname = dbname,table = table,num_topics = num_topics,max_freq=max_freq,multipass = multipass,log = log,tmpDir = tempdir)