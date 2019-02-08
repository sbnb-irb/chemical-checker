#!/miniconda/bin/python

# Imports

import sys, os
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, draw,checkJobResultsForErrors,compressJobResults,all_coords
import Psql
from subprocess import call, Popen
import subprocess
import numpy as np
import collections
import h5py
import time
import uuid
import math


import checkerconfig


SUBMITJOBANDREADY = os.path.join(sys.path[0],'../../src/utils/submitJobOnClusterAndReady.py')

# Main

def main():
    
    import argparse
    
    if len(sys.argv) != 2:
        sys.exit(1)
  
    configFilename = sys.argv[1]

    checkercfg = checkerconfig.checkerConf( configFilename)  

    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    vnumber = checkercfg.getVariable("General",'release')
    log = logSystem(sys.stdout)
    log.debug(os.getcwd())
    

    
    inferfolder = checkerconfig.RELEASESPATH+"/"+vnumber+"/infer"


    try:
     
      scriptName = os.path.join(sys.path[0],"../../src/infer_similarity/inference_preparation.py") + " --dbname " + dbname + " --path " + inferfolder 
      print scriptName
     
      p = Popen(scriptName,shell=True)
      (pid,retcode) = os.waitpid(p.pid, 0)
      if retcode != 0:
        bOk = False
        if retcode > 0:
          log.error( "Script preparation inference  produced some ERROR, please check (exit code %d)!" % (retcode) )
        elif retcode < 0:
          log.error( "Script  was terminated by signal %d" % (-retcode))
        sys.exit(1)
      
    except OSError, e:
      log.critical( "Execution of Script %s failed: %s" % (task,e) )
      sys.exit(1)
  

if __name__ == '__main__':
    main()
