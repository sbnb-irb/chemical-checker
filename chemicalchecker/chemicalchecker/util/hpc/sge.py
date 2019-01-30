"""SGE interace to send jobs to an HPC cluster

"""
import os
import paramiko
import datetime
import glob
import tarfile
import re
import math
import pickle
from itertools import islice
import uuid
import time

from chemicalchecker.util import logged

STARTED = "started"
DONE = "done"
READY = "ready"
ERROR = "error"


@logged
class sge():
    """Send tasks to an HPC cluster through SGE queueing system."""

    jobFilenamePrefix = "job-"
    jobFilenameSuffix = ".sh"

    jobStatusSuffix = ".status"

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

%(command)s

    """

    defaultOptions = """\
#$ -S /bin/bash
#$ -r yes
#$ -j yes
"""

    def __init__(self, config, dry_run=False):
        """Initialize the SGE object.

        """
        self.host = config.HPC.host
        self.queue = None
        self.statusFile = None
        self.status_id = None
        self.conn_params = {}
        if "queue" in config.HPC.asdict():
            self.queue = config.HPC.queue
        if "username" in config.HPC.asdict() and "password" in config.HPC.asdict():
            self.conn_params["username"] = config.HPC.username
            self.conn_params["password"] = config.HPC.password
        if not dry_run:
            try:
                ssh = paramiko.SSHClient()
                ssh.load_system_host_keys()
                ssh.connect(self.host, **self.conn_params)
            except paramiko.SSHException as sshException:
                self.__log.warning(
                    "Unable to establish SSH connection: %s" % sshException)
            finally:
                ssh.close()

    def _chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        if isinstance(l, list):
            for i in range(0, len(l), n):
                yield l[i:i + n]
        elif isinstance(l, dict):
            it = iter(l)
            for i in xrange(0, len(l)):
                yield {k: l[k] for k in islice(it, n)}
        else:
            raise Exception("Element datatype not supported: %s" % type(l))

    def submitMultiJob(self, command, **kwargs):
        """Submit a multi job/task.

         Args:
            command: The comand that will be executed in the cluster. It should contain a
                     <TASK_ID> string and a <FILE> string. These will be replaced by
                     the correponding task id and the pickle file with the elements that
                     the command will need.
            num_jobs:Number of jobs to run the command (default:1)
            cpu:Number of cores the job will use(default:1)
            wait:Wait for the job to finish (default:True)
            jobdir:Directotory where the job will run (default:'')
            job_name:Name of the job (default:10)
            elements:List of elements that will need to run on the command
            compress:Compress all generated files after job is done (default:True)
            check_error:Check for error message in output files(default:True)
            memory:Maximum memory the job can take in Gigabytes(default: 2)
            time: Maximum time(in minutes) the job can run on the cluster(default:infinite)

        """

        # get arguments or default values
        num_jobs = kwargs.get("num_jobs", 1)
        cpu = kwargs.get("cpu", 1)
        wait = kwargs.get("wait", True)
        self.jobdir = kwargs.get("jobdir", '')
        self.job_name = kwargs.get("job_name", 'hpc_cc_job')
        elements = kwargs.get("elements", [])
        compress_out = kwargs.get("compress", True)
        check_error = kwargs.get("check_error", True)
        memory = kwargs.get("memory", 2)
        maxtime = kwargs.get("time", None)

        submit_string = 'qsub -terse '

        if self.queue is not None:
            submit_string += " -q " + self.queue + " "

        if wait:
            submit_string += " -sync y "

        self.__log.debug("Job name is: " + self.job_name)

        if not os.path.exists(self.jobdir):
            os.makedirs(self.jobdir)

        jobParams = ["#$ -N " + self.job_name]
        jobParams.append("#$ -wd " + self.jobdir)

        if (len(elements) > 0 and num_jobs == 1) or (len(elements) == 0 and num_jobs > 1):
            raise("Number of specified jobs does not match to the number of elements")

        if num_jobs > 1:
            jobParams.append("#$ -t 1-" + str(num_jobs))
            tmpname = command.replace("<TASK_ID>", "$SGE_TASK_ID")
            command = tmpname

        if cpu > 1:
            jobParams.append("#$ -pe make " + str(cpu))

        if memory > 1:
            jobParams.append("#$ -l mem_free=" + str(memory) +
                             "G,h_vmem=" + str(memory + 0.5) + "G")

        if maxtime is not None:
            jobParams.append(
                "#$ -l h_rt=" + str(datetime.timedelta(minutes=maxtime)))

        if len(elements) > 0:
            chunk_size = int(
                math.ceil(len(elements) / float(num_jobs)))

            input_dict = dict()
            for cid, chunk in enumerate(self._chunks(elements, chunk_size), 1):
                input_dict[str(cid)] = chunk
            input_path = os.path.join(self.jobdir, str(uuid.uuid4()))
            with open(input_path, 'wb') as fh:
                pickle.dump(input_dict, fh)
            command = command.replace("<FILE>", input_path)

        # Creates the final job.sh
        paramsText = self.defaultOptions + str("\n").join(jobParams)
        jobFilename = os.path.join(
            self.jobdir, sge.jobFilenamePrefix + self.job_name + sge.jobFilenameSuffix)
        self.__log.info("Writing file " + jobFilename + "...")
        jobFile = open(jobFilename, "w")
        jobFile.write(sge.templateScript %
                      {"options": paramsText, "command": command})
        jobFile.close()

        os.chmod(jobFilename, 0o755)

        submit_string += jobFilename

        self.__log.debug("HPC submission: " + submit_string)

        time.sleep(15)

        try:
            ssh = paramiko.SSHClient()
            ssh.load_system_host_keys()
            ssh.connect(self.host, **self.conn_params)
            stdin, stdout, stderr = ssh.exec_command(
                submit_string, get_pty=True)

            job = stdout.readlines()

            if job[0].find(".") != -1:
                self.job_id = job[0][:job[0].find(".")]
            else:
                self.job_id = job[0]

            self.job_id = self.job_id.rstrip()
            self.__log.debug(self.job_id)
        except paramiko.SSHException as sshException:
            self.__log.warning(
                "Unable to establish SSH connection: %s" % sshException)
        finally:
            ssh.close()
            self.statusFile = os.path.join(
                self.jobdir, self.job_name + sge.jobStatusSuffix)
            with open(self.statusFile, "w") as f:
                f.write(STARTED)
            self.status_id = STARTED

        if wait:
            errors = None
            with open(self.statusFile, "w") as f:
                f.write(DONE)
            self.status_id = DONE

            if check_error:
                errors = self.check_errors()

            if compress_out and errors is None:
                self.compress()

            if errors is not None:
                return errors

        return self.job_id

    def check_errors(self):
        """Check for errors in the output logs of the jobs.

        If there are no errors and the status is "done", the status will change to "ready".

        Returns:
            errors(str): The lines in the output logs where the error is found.
                        The format of the errors is filename, line number and line text.
                        If there are no errors it returns None.

        """

        errors = ''
        self.__log.debug("Checking errors in job")

        for file_name in glob.glob(os.path.join(self.jobdir, self.job_name + '.o*')):
            with open(file_name) as f:
                num = 1
                for line in f:
                    if re.search(r'(?i)error', line):
                        errors += file_name + " " + str(num) + " " + line
                    if 'Traceback (most recent call last)' in line:
                        errors += file_name + " " + str(num) + " " + line

                    num += 1

        if len(errors) > 0:
            self.__log.debug("Found errors in job")
            if self.status_id == DONE:
                with open(self.statusFile, "w") as f:
                    f.write(ERROR)
                self.status_id = ERROR
            return errors
        else:
            if self.status_id == DONE:
                with open(self.statusFile, "w") as f:
                    f.write(READY)
                self.status_id = READY
            return None

    def compress(self):
        """Compress the output logs into a tar.gz file in the same job directory


        """

        self.__log.debug("Compressing output job files...")
        tar = tarfile.open(os.path.join(
            self.jobdir, self.job_name + ".tar.gz"), "w:gz")
        for file_name in glob.glob(os.path.join(self.jobdir, self.job_name + '.o*')):
            tar.add(file_name, os.path.basename(file_name))
        tar.close()
        for file_name in glob.glob(os.path.join(self.jobdir, self.job_name + '.o*')):
            os.remove(file_name)

    def status(self):
        """Gets the status of the job submission

           The status is None if there is no job submission.
           The status is also saved in a *.status file in the job directory.

        Returns:
            status(str): There are three possible status if there was a submission
                         "started": Job started but not finished
                         "done": Job finished
                         "ready": Job finished without errors
        """

        if self.statusFile is None:
            return None

        if self.status_id == STARTED:
            try:
                ssh = paramiko.SSHClient()
                ssh.load_system_host_keys()
                ssh.connect(self.host, **self.conn_params)
                stdin, stdout, stderr = ssh.exec_command(
                    'qstat -j ' + self.job_id)

                message = stdout.readlines()

                self.__log.debug(message)

                # if message[0].find("do not exist") != -1:
                if len(message) == 0:
                    self.status_id = DONE
                    with open(self.statusFile, "w") as f:
                        f.write(self.status_id)
            except paramiko.SSHException as sshException:
                self.__log.warning(
                    "Unable to establish SSH connection: %s" % sshException)
            finally:
                ssh.close()

        return self.status_id
