#
# Utility functions used in the interactome3D pipeline
#

# Imports
import os
import sys
import datetime
from tqdm import tqdm
from subprocess import call, Popen
from rdkit.Chem import AllChem
from rdkit import Chem
sys.path.append(os.path.join(sys.path[0],"../../pipeline/config"))
import checkerconfig

# Constants

SPATH = os.path.dirname(os.path.abspath(__file__))


# Classes
class logSystem:
  """Defines a simple logging system which works on a file stream provided from
     outside"""
  
  _fileHandler = None
  _minLevel    = 20
  
  DEBUG    = 10 
  INFO     = 20
  WARNING  = 30
  ERROR    = 40
  CRITICAL = 50
  
  def __init__( self, aFileHandler ):
    self._fileHandler = aFileHandler
  
  def setMinLevel( self, minLevel ):
    self._minLevel = minLevel

  def detach( self ):
    self._fileHandler = None
    
  def logMessage( self, type, message ):
    """Writes out a message to the log file. Includes information on the time and
       date of the message as well as the type of message. Type can be one of
       DEBUG, INFO, WARNING, ERROR, CRITICAL"""
  
    if self._fileHandler == None: return
    
    if type < self._minLevel: return
    
    now = datetime.datetime.now()
    
    if type == self.DEBUG:
      typeStr = "DEBUG"
    elif type == self.INFO:
      typeStr = "INFO"
    elif type == self.WARNING:
      typeStr = "WARNING"
    elif type == self.ERROR:
      typeStr = "ERROR"
    elif type == self.CRITICAL:
      typeStr = "ABORT"
    outText = "%s\t%s\t%s\n" % (now.strftime("%Y-%m-%d %H:%M.%S"), typeStr, message)
    self._fileHandler.write(outText)
    self._fileHandler.flush()

  def debug( self, message):
    """Writes out a DEBUG message to the log file. Includes information on the time and
       date of the message as well as the type of message."""
    self.logMessage( self.DEBUG, message )

  def info( self, message):
    """Writes out a INFO message to the log file. Includes information on the time and
       date of the message as well as the type of message."""
    self.logMessage( self.INFO, message )
    
  def warning( self, message):
    """Writes out a WARNING message to the log file. Includes information on the time and
       date of the message as well as the type of message."""
    self.logMessage( self.WARNING, message )

  def error( self, message):
    """Writes out a ERROR message to the log file. Includes information on the time and
       date of the message as well as the type of message."""
    self.logMessage( self.ERROR, message )
    
  def critical( self, message):
    """Writes out a CRITICAL message to the log file. Includes information on the time and
       date of the message as well as the type of message."""
    self.logMessage( self.CRITICAL, message )

# Functions
def execAndCheck(cmdStr,log,allowedReturnValues=set()):
  try:
    log.debug( cmdStr )
    retcode = call(cmdStr,shell=True)
    log.debug( "FINISHED! "+cmdStr+(" returned code %d" % retcode) )
    if retcode != 0:
      if retcode not in allowedReturnValues:
        if retcode > 0:
          log.error( "ERROR return code %d, please check!" % retcode )
        elif retcode < 0:
          log.error( "Command terminated by signal %d" % -retcode )
        sys.exit(1)
  except OSError as e:
      log.critical( "Execution failed: %s" % e )
      sys.exit(1)

  return retcode

def execInBackground(cmdStr,log):
  try:
    log.debug( cmdStr )
    proc = Popen(cmdStr,shell=True)
  except OSError as e:
      log.critical( "Launching the program in %s background failed: %s" % (cmdStr,e) )
      sys.exit(1)
  return proc

def inchikey2webrepo(inchikey):
    PATH = checkerconfig.WEBREPOMOLS + "/" + inchikey[:2]
    PATH = PATH.replace("//", "/")
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    PATH = PATH + "/" + inchikey[2:4]
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    PATH = PATH + "/" + inchikey
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    return PATH + "/"

def inchikey2webrepo_no_mkdir(inchikey):
    PATH = checkerconfig.WEBREPOMOLS + "/" + inchikey[:2] + "/" + inchikey[2:4] + "/" + inchikey + "/"
    PATH = PATH.replace("//", "/")
    return PATH



# TABLE COORDINATES

table_coordinates = {
'fp2d': 'A1',
'fp3d': 'A2',
'scaffolds': 'A3',
'subskeys': 'A4',
'physchem': 'A5',
'moa': 'B1',
'metabgenes': 'B2',
'crystals': 'B3',
'binding': 'B4',
'htsbioass': 'B5',
'molroles': 'C1',
'molpathways': 'C2',
'pathways': 'C3',
'bps': 'C4',
'networks': 'C5',
'transcript': 'D1',
'cellpanel': 'D2',
'chemgenet': 'D3',
'morphology': 'D4',
'cellbioass': 'D5',
'therapareas': 'E1',
'indications': 'E2',
'sideeffects': 'E3',
'phenotypes': 'E4',
'ddis': 'E5'
}

coordinate_tables = dict((v,k) for k,v in table_coordinates.iteritems())

# Table colors

def rgb2hex(r, g, b):
    return '#%02x%02x%02x' % (r, g, b)

def table_color(table):
    A = rgb2hex(250, 100, 80 )
    B = rgb2hex(200, 100, 225)
    C = rgb2hex(80,  120, 220)
    D = rgb2hex(120, 180, 60 )
    E = rgb2hex(250, 150, 50 )
    if table_coordinates[table][0] == 'A': return A
    if table_coordinates[table][0] == 'B': return B
    if table_coordinates[table][0] == 'C': return C
    if table_coordinates[table][0] == 'D': return D
    if table_coordinates[table][0] == 'E': return E

def coordinate_color(coord):
    table = coordinate_tables[coord]
    return table_color(table)


gray = rgb2hex(220, 218, 219)

def log_data(log_obj, data):

    if log_obj == None:
        print data
    else:
        log_obj.info(data)
        
        
def tqdm_local(log_obj,iter_obj):
    if log_obj == None:
        for i in tqdm(iter_obj):
            yield i
    else:
        for i in iter_obj:
            yield i

def getNumOfLines(aFilename):
  tmpFile = open(aFilename)
  numLines = 0
  for line in tmpFile:
    numLines += 1
  tmpFile.close()
  return numLines
  

def draw(inchikey, inchi):
    PATH = inchikey2webrepo(inchikey)
    if  os.path.exists(PATH + "/2d.svg"): return
    mol = Chem.rdinchi.InchiToMol(inchi)[0]
    AllChem.Compute2DCoords(mol)
    with open(PATH + "/2d.mol", "w") as f:
        try:
            f.write(Chem.MolToMolBlock(mol))
        except:
            f.write(Chem.MolToMolBlock(mol, kekulize = False))
    cmd = "%s/../../tools/mol2svg --bgcolor=220,218,219 --color=%s/black.conf %s/2d.mol > %s/2d.svg" % (SPATH, SPATH, PATH, PATH)
    Popen(cmd, shell = True).wait()
    os.remove("%s/2d.mol" % PATH)



def checkJobResultsForErrors(directory,jobName,log,maxNumOfErrors=0,errStrings=[]):
  # Error checking
  errorFilename = os.path.join(directory,jobName+".error")
  if os.path.exists(errorFilename):
    log.critical( "An ERROR occurred in the execution of job %s" % jobName )
    sys.exit(1)
  grepResultsFilename = os.path.join(directory,jobName+".errcheck")
  if len(errStrings) == 0:
    cmdStr = 'for i in '+os.path.join(directory,jobName+".o*.*")+'; do grep -i error $i; done > '+grepResultsFilename
    execAndCheck(cmdStr,log,set([1]))
    cmdStr = 'for i in '+os.path.join(directory,jobName+".o*.*")+'; do grep -i "Traceback (most recent call last)" $i; done >> '+grepResultsFilename
    execAndCheck(cmdStr,log,set([1]))
  else:
    for es in errStrings:
      cmdStr = 'for i in '+os.path.join(directory,jobName+".o*.*")+'; do grep "'+es+'" $i; done > '+grepResultsFilename
      execAndCheck(cmdStr,log,set([1]))
  numOfErrors = getNumOfLines(grepResultsFilename)
  if numOfErrors > maxNumOfErrors:
    log.critical( "Some ERRORs occurred in the execution of job %s, see the file %s" % (jobName,grepResultsFilename) )
    sys.exit(1)
  elif numOfErrors > 0:
    log.warning( "Some ERRORs occurred in the execution of job %s, see the file %s (%d errors, %d max. tolerated)" % (jobName,grepResultsFilename,numOfErrors,maxNumOfErrors) )

def compressJobResults(directory,jobName,additionalFileSpecs,log):
  currentPath = os.getcwd()
  os.chdir(directory)
  cmdStr = 'tar czf '+jobName+'.tgz'+' '+jobName+'.o*'
  for fs in additionalFileSpecs: cmdStr += ' '+fs
  execAndCheck(cmdStr,log)
    
  cmdStr = 'rm -rf '+jobName+'.o*'
  for fs in additionalFileSpecs: cmdStr += ' '+fs
  execAndCheck(cmdStr,log)
  os.chdir(currentPath)
