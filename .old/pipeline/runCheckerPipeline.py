#!/usr/bin/env python
#
# Runs the Interactome3D and checks that every step is executed correctly
#

# Imports
import os
import sys
import logging
import socket
import datetime
import smtplib
import email.utils
from email.mime.text import MIMEText
from optparse import OptionParser
from subprocess import Popen, PIPE, STDOUT

sys.path.append(os.path.join(sys.path[0],"../src/utils"))
sys.path.append(os.path.join(sys.path[0],"config"))

from checkerUtils import logSystem, execAndCheck

import checkerconfig

# Constants
usageText = """%prog [options] <config_ini>

Runs the pipeline."""

FIRST_STEP = 1
LAST_STEP  = len(checkerconfig.steps)

CLEANUP_SCRIPT = "cleanup.py"
RUN_SCRIPT     = "run.py"

# Functions
def sendEmail(toAddress,subject,text,smtpServer,smtpUsername,smtpPassword):
  """Sends an e-mail"""  

  msg = MIMEText(text)
  msg['To'] = toAddress
  msg['From'] = email.utils.formataddr(checkerconfig.checkerEmailAddress)
  msg['Reply-To'] = email.utils.formataddr(checkerconfig.checkerEmailAddress)
  msg['Subject'] = subject
  
  
  server = smtplib.SMTP(smtpServer)
  try:
      # identify ourselves, prompting server for supported features
      server.ehlo()
  
      # If we can encrypt this session, do it
      if server.has_extn('STARTTLS'):
          server.starttls()
          server.ehlo() # re-identify ourselves over TLS connection
  
      if smtpUsername != "":
        server.login(smtpUsername, smtpPassword)
      server.sendmail(checkerconfig.checkerEmailAddress[1], [toAddress], msg.as_string())
  finally:
      server.quit()

def sendErrorMail(runName,logFilepath,emailAddress,step,smtpServer,smtpUsername,smtpPassword):
  cmdStr = 'tail -n 20 '+logFilepath
  textToSend = Popen(cmdStr, shell=True, stdout=PIPE).communicate()[0]
  sendEmail(emailAddress,
            "Error in step %d of the pipeline" % step,
            ("The Checker pipeline for job '%s' has been interrupted on step *%s*"+
             ", please check.\n\nFind below an extract of the pipeline log "+
             "file.\n\nChecker Pipeline\n\n"+
             "------------------------------------------------------------"+
             "--------------------\n%s\n----------------------------------"+
             "----------------------------------------------") % (runName,checkerconfig.steps[step-1],textToSend),
            smtpServer,smtpUsername,smtpPassword)

def sendSuccessMail(datasetName,resultsUrl,emailAddress,smtpServer,smtpUsername,smtpPassword):
  if resultsUrl != "":
    mailText = checkerconfig.resultsReadyUrlMailText % {'DATASET_NAME':datasetName,
                                                      'RESULTS_URL':resultsUrl,
                                                      'INT3D_EMAIL':checkerconfig.checkerEmailAddress[1]}
  else:
    mailText = checkerconfig.resultsReadyMailText % {'DATASET_NAME':datasetName,
                                                      'INT3D_EMAIL':checkerconfig.checkerEmailAddress[1]}
  sendEmail(emailAddress,
            "Checker Job FINISHED: %s" % datasetName,
            mailText,
            smtpServer,smtpUsername,smtpPassword)

def main():
  parser = OptionParser(usage=usageText)
  parser.disable_interspersed_args()

  parser.set_defaults(force=False,cleanup=False,disableModbase=False)

  parser.add_option("-f", "--force",
                    action="store_true", dest="force",
                    help="Prevents the script asking for confirmation")

  parser.add_option("-C", "--cleanup",
                    action="store_true", dest="cleanup",
                    help="Cleans up things")

 
  
  parser.add_option("-s", "--steps", dest="steps",
                    help="specifies the steps to be executed. Use the format "+
                         "start-end like 2-3 or 3- for all the steps starting "+
                         "from 3.", metavar="STEPS")
                    
  (options, args) = parser.parse_args()
  
  if len(args) != 1:
    print >>sys.stderr,"Error: No config file entered."
    parser.print_help()
    sys.exit(1)
  
  firstStep = FIRST_STEP
  lastStep  = LAST_STEP
  
  configFilename = os.path.abspath(args[0])

  checkerCfg = checkerconfig.checkerConf( configFilename )  
  
  if options.steps != None:
    stepsSpecification = options.steps
    limits = stepsSpecification.split('-')
    if len(limits) > 2:
      sys.stderr.write("ERROR! Wrong specification of steps. Use the format "+
                       "start-end like 2-3 or 3- for all the steps starting "+
                       "from 3.")
      sys.exit(1)
    else:
      start = int(limits[0])
      if len(limits) > 1:
        if limits[1].strip() == '':
          end = LAST_STEP
        else:
          end = int(limits[1])
      else:
        end = start
      if start < 1 or start > LAST_STEP or \
         end < 1 or end > LAST_STEP or \
         start > end:
        sys.stderr.write("ERROR! Wrong specification of steps. Use the format"+
                         " start-end like 2-3 or 3- for all the steps starting"+
                         " from 3.")
        sys.exit(1)
      firstStep = start
      lastStep  = end
  
  if options.cleanup and not options.force:
    choice = raw_input("Do you really want to delete the results? [y/N] ")
    if choice != 'y' and choice != 'Y':
      sys.exit(0)
    if options.steps != None:
      sys.stderr.write("WARNING! Since the '--cleanup' option has been given "+
                       "The '--steps' option will be ignored.")
      sys.exit(1)

  RUNNAME       = checkerCfg.getVariable('General', 'release')
  SMTPSERVER    = checkerCfg.getVariable('General', 'smtpserver')
  SMTPUSER      = checkerCfg.getVariable('General', 'smtpuser')
  SMTPPWD       = checkerCfg.getVariable('General', 'smtppwd')
  EMAIL         = checkerCfg.getVariable('General', 'email')
  USREMAIL      = ""
  SCRATCHDIR    = checkerCfg.getVariable('General', 'scratchdir')
  if checkerCfg.hasVariable('General','userlogfile'):
    USER_LOG_FILE = checkerCfg.getVariable('General', 'userlogfile')
  else:
    USER_LOG_FILE = ""
    

  readyFiledir = checkerCfg.getDirectory("ready")
  if not os.path.exists(readyFiledir):
    os.makedirs(readyFiledir)

  # If the scratch directory does not exist I need to create it
  scratchDir = checkerCfg.getDirectory("scratch")
  if not os.path.exists(scratchDir):
    os.makedirs(scratchDir)
  
  # If the logs directory does not exists I create it
  logsFiledir = checkerCfg.getDirectory( "logs" )
  if not os.path.exists(logsFiledir):
    os.makedirs(logsFiledir)
  
  # Opens the log file
  logFilepath = os.path.join(logsFiledir,checkerconfig.logFilename)
  logFile = open( logFilepath, 'a')
  
  logFile.write("\n"+("="*80)+"\n\n")
  
  log = logSystem(logFile)
  
  log.info( "New execution started. Executing steps from %d to %d" % (firstStep, lastStep) )
  
 
    
  # Then I execute each step individually
  for i in range(firstStep-1,lastStep):
    log.debug( ">>>>>>>> ENTERING STEP %s <<<<<<<<" % checkerconfig.steps[i])
    readyFilename = os.path.join(readyFiledir,checkerconfig.steps[i]+".ready")
    if os.path.exists(readyFilename):
      log.info( "Ready file for step %s does exist. Skipping this step..." % checkerconfig.steps[i] )
      continue
    # First I check that the results of the previous step are ready
    if i > 0:
      readyFilename = os.path.join(readyFiledir,checkerconfig.steps[i-1]+".ready")
      log.debug(readyFilename)
      if not os.path.exists(readyFilename):
        log.critical( "Ready file for step %s does not exist. Aborting..." % checkerconfig.steps[i-1] )
        if EMAIL != "":
          sendErrorMail(RUNNAME,logFilepath,EMAIL,i,SMTPSERVER,SMTPUSER,SMTPPWD)
          log.info( "An e-mail has been sent to %s" % EMAIL )
        sys.exit(1)
    # Then I execute the current step
    log.info( "----------------------------------------------------------------------------" )
    log.info( ">>>>>>>> EXECUTING STEP %s <<<<<<<<" % checkerconfig.steps[i] )
    runScript = os.path.join(sys.path[0],checkerconfig.steps[i],RUN_SCRIPT)
    #log.debug(os.getcwd())
    try:
      p = Popen( [runScript,configFilename], stdout=logFile, stderr=STDOUT )
      (pid,retcode) = os.waitpid(p.pid, 0)
      if retcode > 0:
        log.error( "Step %s produced some ERROR, please check (exit code %d)!" % (checkerconfig.steps[i],retcode) )
      elif retcode < 0:
        log.error( "Step %s was terminated by signal %d" % (checkerconfig.steps[i],-retcode))
    except OSError, e:
      log.critical( "Execution of step %s failed: %s" % (checkerconfig.steps[i],e) )
      sys.exit(1)
    log.info( ">>>>>>>> FINISHED STEP %s <<<<<<<<" % checkerconfig.steps[i])

  readyFilename = os.path.join(readyFiledir,checkerconfig.steps[lastStep-1]+".ready")
  if not os.path.exists(readyFilename):
    log.critical( "Ready file for step %s does not exist. Aborting..." % checkerconfig.steps[lastStep-1] )
    if EMAIL != "":
      sendErrorMail(RUNNAME,logFilepath,EMAIL,lastStep,SMTPSERVER,SMTPUSER,SMTPPWD)
      log.info( "An e-mail has been sent to %s" % EMAIL )
    sys.exit(1)

  log.info( "Execution of steps from %d to %d ended successfully." % (firstStep, lastStep) )

  if USER_LOG_FILE != "":
      logging.basicConfig( level=logging.DEBUG, \
                       format='%(asctime)s %(levelname)-8s %(message)s', \
                       datefmt='%Y-%m-%d %H:%M:%S', \
                       filename=USER_LOG_FILE, \
                       filemode='a')
      logging.info( checkerconfig.finishedString )
      logging.info( checkerconfig.validatedString )

  if EMAIL != "" or USREMAIL != "":
    resultsUrl = ""
    if USER_LOG_FILE != "":
      userLogFile = open( USER_LOG_FILE )
      for line in userLogFile:
        if line.find(checkerconfig.resultsUrlString) != -1:
          resultsUrl = line.rstrip("\n\r").split()[-1]
          break
      userLogFile.close()
   
    if EMAIL != "":
      sendSuccessMail("",resultsUrl,EMAIL,SMTPSERVER,SMTPUSER,SMTPPWD)
    if USREMAIL != "":
      sendSuccessMail("",resultsUrl,USREMAIL,SMTPSERVER,SMTPUSER,SMTPPWD)
    log.info( "An e-mail has been sent to %s" % EMAIL )

  log.detach()
  logFile.close()
  
# Main
main()
