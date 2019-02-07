#!/usr/bin/env python
#
# Creates the running script for an array job to be run with the N1 Grid
# Engine System 6.0 from Sun using the qsub command. The sript accepts the
# command to be run as the last part of the command line and creates a running
# script to be passed to qsub. What remains to be done by the user is invoking
# qsub on the script with the command 'qsub job.sh'.


# Imports
from optparse import OptionParser
from math import log10
from math import floor
from math import ceil
import sys
import os

# Constants
usageText = """%prog [options] command arg1 arg2 arg3 ...

Creates the running script for a single job to be run with the N1 Grid
Engine System 6.0 from Sun using the qsub command. The sript accepts the
command to be run as the last part of the command line and creates a running
script to be passed to qsub. What remains to be done by the user is invoking
qsub on the script with the command

 > qsub job.sh"""

jobFilenamePrefix = "job-"
jobFilenameSuffix = ".sh"

taskDir            = "tasks"
taskFilenamePrefix = "task-"
taskFilenameSuffix = ".lst"

defaultOptions = """\
#$ -S /bin/bash
#$ -r yes
"""

templateScript = """\
#!/bin/bash
#
#

# Options for qsub
%(options)s
# End of qsub options

# Loads default environment configuration
if [[ -f $HOME/.bashrc ]]
then
  source $HOME/.bashrc
fi

# Current task id
taskId=`printf "%%0%(number_of_digits)dd" $SGE_TASK_ID`
taskFilename=%(task_dir)s/%(task_prefix)s.$taskId%(task_suffix)s

# Runs the command right
for i in `cat $taskFilename`
do
    %(command)s
done
"""

# Functions
def main():
  parser = OptionParser(usage=usageText)
  parser.disable_interspersed_args()
  
  parser.set_defaults(cwd=True)

  parser.add_option("-o", "--stdout", dest="stdout",
                    help="redirect stdout to the given file FILE", metavar="FILE")
  parser.add_option("-q", "--quiet", dest="quiet", default=False,
                    action="store_true", help="suppress program's output")
  parser.add_option("-e", "--stderr", dest="stderr",
                    help="redirect stderr to the given file FILE", metavar="FILE")
  parser.add_option("-l", "--joblist", dest="joblist",
                    help="loads the list of jobs from file FILE", metavar="FILE")
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
  parser.add_option("-t", "--tasks", dest="numtasks", type="int",
                    help="number of tasks")
  parser.add_option("-H", "--homedir",
                    action="store_false", dest="cwd",
                    help="set current working directory to home dir")
                    
  (options, args) = parser.parse_args()
  
  if len(args) < 1:
    print "Error: No command entered, please enter a command or type '"+os.path.basename(sys.argv[0])+" -h' for help"
    sys.exit(1)
  if options.jobname == None:
    print "Error: No name given for the job, please specify a name with the '-N' option or type '"+os.path.basename(sys.argv[0])+" -h' for help"
    sys.exit(1)
  if options.joblist == None:
    print "Error: No jobs list specified, please specify a file conaining the jobs list with the '-l' option or type '"+os.path.basename(sys.argv[0])+" -h' for help"
    sys.exit(1)
  if options.numtasks == None:
    print "Error: No number of tasks specified, please use the '-t' option or type '"+os.path.basename(sys.argv[0])+" -h' for help"
    sys.exit(1)
  else:
    numOfTasks = options.numtasks
  
  # First we need to check if the number of jobs is correct
  # Is the number of jobs in the jobs list greater than the number of
  # desired jobs?
  jobsListFile = open( options.joblist )
  jobs = []
  for line in jobsListFile:
    line = line.strip()
    if len(line) > 0:
      jobs.append(line)
  jobsListFile.close()
  numOfListedJobs = len(jobs)
  if numOfListedJobs < numOfTasks:
    if not options.quiet:
      print "Warning: The number of listed jobs is less than the number of tasks (%d < %d), resizing the number of tasks." % (numOfListedJobs,numOfTasks)
    numOfTasks = numOfListedJobs
  
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

  singleOptions.append( "#$ -t 1-"+str(numOfTasks) )
  optionsText = defaultOptions+str("\n").join(singleOptions)
 
  numOfDigits = int(floor(log10( numOfTasks )))+1
  
  # Creates a directory for every job and a script in every directory
  #for i in range( 1, numOfJobs ):
  #  os.mkdir( "job."+str().zfill(numOfZeros) )
  
  if not os.path.exists( taskDir ):
    os.makedirs( taskDir )
    
  # Creates files containing the single lists of jobs
  jobsPerBin = float(numOfListedJobs)/float(numOfTasks)
  #print jobsPerBin
  if not options.quiet:
    if jobsPerBin - float(int(jobsPerBin)) == 0.0:
      print "Creating "+str(numOfTasks)+" tasks of "+str(int(jobsPerBin))+" jobs each."
    else:  
      print "Creating "+str(numOfTasks)+" tasks of "+str(int(jobsPerBin))+" to "+str(int(jobsPerBin)+1)+" jobs each."
  for i in range( 0, numOfTasks ):
    taskFilename = taskDir+"/"+taskFilenamePrefix+options.jobname+"."+str(i+1).zfill(numOfDigits)+taskFilenameSuffix
    taskFile = open(taskFilename,"w")
    lowerLimit = int(float(i)*float(numOfListedJobs)/float(numOfTasks))
    #print lowerLimit
    upperLimit = min(int(float(i+1)*float(numOfListedJobs)/float(numOfTasks)),numOfListedJobs)
    #print upperLimit
    taskFile.write( str("\n").join(jobs[lowerLimit:upperLimit])+"\n" )
    taskFile.close()
  
  # Creates the final job.sh
  jobFilename = jobFilenamePrefix+options.jobname+jobFilenameSuffix
  if not options.quiet:
    sys.stdout.write( "Writing file "+jobFilename+"..." )
    sys.stdout.flush()
  jobFile = open( jobFilename, "w" )
  jobFile.write( templateScript % {"options":optionsText, "command":commandText,
                   "task_dir":taskDir,
                   "task_prefix":taskFilenamePrefix+options.jobname, 
                   "task_suffix":taskFilenameSuffix,
                   "number_of_digits":numOfDigits } )
  jobFile.close()
  if not options.quiet:
    sys.stdout.write( "done.\n" )

  os.chmod( jobFilename, 0755 )
  
main()
