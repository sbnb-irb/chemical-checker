#!/usr/bin/env python
#
# Creates the running script for a single job to be run on with the N1 Grid
# Engine System 6.0 from Sun using the qsub command. The sript accepts the
# command to be run as the last part of the command line and creates a running
# script to be passed to qsub. What remains to be done by the user is invoking
# qsub on the script with the command 'qsub job.sh'.


# Imports
from optparse import OptionParser
import stat
import sys
import os

# Constants
usageText = """%prog [options] command arg1 arg2 arg3 ...

Creates the running script for a single job to be run on with the N1 Grid
Engine System 6.0 from Sun using the qsub command. The sript accepts the
command to be run as the last part of the command line and creates a running
script to be passed to qsub. What remains to be done by the user is invoking
qsub on the script with the command

 > qsub job.sh"""

jobFilenamePrefix = "job-"
jobFilenameSuffix = ".sh"

defaultOptions = """\
#$ -S /bin/bash
#$ -r yes
"""

templateScript = """\
#!/bin/bash
#
# (c) 2008 Roberto Mosca
#

# Options for qsub
%(options)s
# End of qsub options

# Loads default environment configuration
if [[ -f $HOME/.bashrc ]]
then
  source $HOME/.bashrc
fi

# Runs the command
%(command)s
"""

# Functions
def main():
  parser = OptionParser(usage=usageText)
  parser.disable_interspersed_args()
  
  parser.set_defaults(cwd=True)

  parser.add_option("-o", "--stdout", dest="stdout",
                    help="redirect stdout to the given file FILE", metavar="FILE")
  parser.add_option("-e", "--stderr", dest="stderr",
                    help="redirect stderr to the given file FILE", metavar="FILE")
  parser.add_option("-N", "--name", dest="jobname",
                    help="give the job the name NAME", metavar="NAME")
  parser.add_option("-m", "--email", dest="email",
                    help="send an email at job completion", metavar="user@domain.org")
  parser.add_option("-j", "--join",
                    action="store_true", dest="join", default=True,
                    help="join stdout and stderr")
  parser.add_option("-x", "--exclude-check",
                    action="store_true", dest="exclude_check", default=False,
                    help="excludes checking for executable existence")
  parser.add_option("-H", "--homedir",
                    action="store_false", dest="cwd",
                    help="set current working directory to home dir")
  parser.add_option("-s", "--sync",
                    action="store_false", dest="sync",
                    help="set synchronization job")
                    
  (options, args) = parser.parse_args()
  
  if len(args) < 1:
    print "Error: No command entered, please enter a command or type '"+os.path.basename(sys.argv[0])+" -h' for help"
    sys.exit(1)
  if options.jobname == None:
    print "Error: No name given for the job, please specify a name with the '-N' option or type '"+os.path.basename(sys.argv[0])+" -h' for help"
    sys.exit(1)
  
  if not options.exclude_check:
    commandText = os.path.abspath( args[0] )
    if not os.access( commandText, os.X_OK ): #path.exists(commandText):
      print "Error: Is the file '"+args[0]+"' present and executable?"
      sys.exit(1)
  else:
    commandText = args[0]
    
  if len(args)>1:
    commandText += " "+str(" ").join( args[1:] )

  singleOptions = ["#$ -N "+options.jobname]
  if options.stdout != None:
    singleOptions.append( "#$ -o "+options.stdout )

  if options.stderr != None:
    singleOptions.append( "#$ -e "+options.stderr )
    
  if options.email != None:
    singleOptions.append( "#$ -M "+options.email )
    singleOptions.append( "#$ -m ea" )

  if options.join:
    singleOptions.append( "#$ -j yes" )
  
  if options.cwd:
    singleOptions.append( "#$ -cwd" )

  if options.sync:
    singleOptions.append( "#$ -sync y" )

  optionsText = defaultOptions+str("\n").join(singleOptions)
 
  jobFilename = jobFilenamePrefix+options.jobname+jobFilenameSuffix
  sys.stdout.write( "Writing file "+jobFilename+"..." )
  sys.stdout.flush()
  jobFile = open( jobFilename, "w" )
  jobFile.write( templateScript % {"options":optionsText, "command":commandText } )
  jobFile.close()
  sys.stdout.write( "done." )
  
  os.chmod( jobFilename, 0755 )
  
main()
