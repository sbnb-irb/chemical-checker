#
# Utility functions used in the interactome3D pipeline
#

# Imports
import os
import sys
import datetime
from subprocess import call, Popen

# Constants

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

def getNumOfLines(aFilename):
  tmpFile = open(aFilename)
  numLines = 0
  for line in tmpFile:
    numLines += 1
  tmpFile.close()
  return numLines

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
