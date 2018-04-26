#!/usr/bin/env python
#
# Runs all the tasks of this step
#

# Imports
import os
import sys
import subprocess


sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck


import checkerconfig


# Constants
tasks = [
  ( 'Dowload all files',            'downloadFiles.py',                          'download_files' ),
  ( 'Mol Repos BindingDB',          'essential_molrepos.py bindingdb',      'mol_repos_bindingdb' ),
  ( 'Mol Repos Chebi',              'essential_molrepos.py chebi',          'mol_repos_chebi' ),
  ( 'Mol Repos Chembl',             'essential_molrepos.py chembl',         'mol_repos_chembl' ),
  ( 'Mol Repos CTD',                'essential_molrepos.py ctd',            'mol_repos_ctd' ),
  ( 'Mol Repos Drugbank',           'essential_molrepos.py drugbank',       'mol_repos_drugbank' ),
  ( 'Mol Repos Kegg',               'essential_molrepos.py kegg',           'mol_repos_kegg' ),
  ( 'Mol Repos Lincs',              'essential_molrepos.py lincs',          'mol_repos_lincs' ),
  ( 'Mol Repos Morphlincs',         'essential_molrepos.py morphlincs',     'mol_repos_morphlincs' ),
  ( 'Mol Repos Mosaic',             'essential_molrepos.py mosaic',         'mol_repos_mosaic' ),
  ( 'Mol Repos NCI60',              'essential_molrepos.py nci60',          'mol_repos_nci60' ),
  ( 'Mol Repos PDB',                'essential_molrepos.py pdb',            'mol_repos_pdb' ),
  ( 'Mol Repos Sider',              'essential_molrepos.py sider',          'mol_repos_sider' ),
  ( 'Mol Repos Smpdb',              'essential_molrepos.py smpdb',          'mol_repos_smpdb' )
]

# Functions
def usage(progName):
  print "Usage: "+progName+" <config_ini>"

def main():
 # Check arguments
  # Check arguments
  if len(sys.argv) != 2:
    usage(sys.argv[0])
    sys.exit(1)
  
  configFilename = sys.argv[1]

  checkercfg = checkerconfig.checkerConf(configFilename )  

  readyFiledir = checkercfg.getDirectory( "ready", configUpdate.UNIPROTKB_SUBDIR )
  

  log = logSystem(sys.stdout)
  log.debug(os.getcwd())

  dirName = os.path.abspath(sys.argv[0]).split("/")[-2]
  
  bOk = True
  for i in range(0,len(tasks)):
    log.info("====>>>> "+tasks[i][0]+" <<<<====")
    readyFilename = os.path.join(readyFiledir,dirName+"_"+tasks[i][2]+".ready")
    if os.path.exists(readyFilename):
      log.info( "Ready file for task %s does exist. Skipping this task..." % tasks[i][2] )
      continue
    # Then I execute the current task
    try:
      scriptName = os.path.join(sys.path[0],tasks[i][1])
      p = subprocess.Popen( [scriptName,configFilename], stderr=subprocess.STDOUT )
      (pid,retcode) = os.waitpid(p.pid, 0)
      if retcode != 0:
        bOk = False
        if retcode > 0:
          log.error( "Script %s produced some ERROR, please check (exit code %d)!" % (tasks[i][1],retcode) )
        elif retcode < 0:
          log.error( "Script %s was terminated by signal %d" % (tasks[i][1],-retcode))
        sys.exit(1)
      cmdStr = "touch "+readyFilename
      subprocess.call(cmdStr,shell=True)
    except OSError, e:
      log.critical( "Execution of Script %s failed: %s" % (tasks[i][1],e) )
      sys.exit(1)
    log.info("====>>>> "+tasks[i][0]+"...done! <<<<====")
  
  if bOk:
    readyFilename = os.path.join(readyFiledir,dirName+".ready")
    log.debug(readyFilename)
    cmdStr = "touch "+readyFilename
    subprocess.call(cmdStr,shell=True)
  
# Main
main()

