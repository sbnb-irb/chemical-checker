"""Submit jobs to an HPC cluster through SGE queueing system."""
import os
import re
import glob
import uuid
import time
import pickle
import tarfile
import datetime
import paramiko
import numpy as np
import math

from chemicalchecker.util import logged

STARTED = "started"
DONE = "done"
READY = "ready"
ERROR = "error"


@logged
class sge():
    """SGE job class."""

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

    def __init__(self, **kwargs):
        """Initialize a SGE instance."""
        self.host = kwargs.get("host", '')
        self.queue = kwargs.get("queue", None)
        self.username = kwargs.get("username", '')
        self.password = kwargs.get("password", '')
        self.error_finder = kwargs.get("error_finder", self.__find_error)
        dry_run = kwargs.get("dry_run", False)
        self.specificNode = kwargs.get("specificNode", None)
        if self.specificNode:
            defaultOptions = """\
#$ -S /bin/bash
#$ -r yes
#$ -j yes
#$ -q all.q@{}
""".format(self.specificNode)

        self.statusFile = None
        self.status_id = None
        self.conn_params = {}

        if self.username != '' and self.password != '':
            self.conn_params["username"] = self.username
            self.conn_params["password"] = self.password
        if not dry_run:
            try:
                ssh = paramiko.SSHClient()
                ssh.load_system_host_keys()
                ssh.connect(self.host, **self.conn_params)
            except paramiko.SSHException as sshException:
                raise Exception(
                    "Unable to establish SSH connection: %s" % sshException)
            finally:
                ssh.close()

    def _chunks(self, l, n):
        """Yields n successive chunks from l."""
        if isinstance(l, list) or isinstance(l, np.ndarray):
            # a list of 60 000 entries split in 6000 chunks
            for i in np.array_split(l, n):
                yield i   # yields one of the 6000 chunks of 10 elements
        elif isinstance(l, dict):
            keys = list(l.keys())
            keys.sort()
            for i in np.array_split(keys, n):
                yield {k: l[k] for k in i}

        # NS: to correct a bug on D1 sign0 calculation
        elif isinstance(l, type(dict().keys())):
            keys = list(l).sort()
            for i in np.array_split(keys, n):
                yield {k: l[k] for k in i}
        else:
            raise Exception("Element datatype not supported: %s" % type(l))

    def submitMultiJob(self, command, **kwargs):
        """Submit multiple job/task."""
        # get arguments or default values
        # ex: 1/10th of the size of the iterable to be processed
        num_jobs = int(kwargs.get("num_jobs", 1))
        cpu = kwargs.get("cpu", 1)
        wait = kwargs.get("wait", True)
        self.jobdir = kwargs.get("jobdir", '')
        self.job_name = kwargs.get("job_name", 'hpc_cc_job')
        elements = kwargs.get("elements", [])
        compress_out = kwargs.get("compress", True)
        check_error = kwargs.get("check_error", True)
        memory = kwargs.get("memory", 2)
        maxtime = kwargs.get("time", None)
        cpusafe = kwargs.get("cpusafe", True)
        membycore = int(kwargs.get("mem_by_core", 2))

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

        if (len(elements) == 0 and num_jobs > 1):
            raise Exception(
                "Number of specified jobs does not match"
                " to the number of elements")

        if num_jobs == 0:
            raise Exception("Number of specified jobs is zero")

        if num_jobs > 1 or command.find("<TASK_ID>") != -1:
            jobParams.append("#$ -t 1-" + str(num_jobs))
            tmpname = command.replace("<TASK_ID>", "$SGE_TASK_ID")
            command = tmpname

        mem_need = memory
        if memory > membycore:
            if cpu > 1:
                newcpu = max(int(math.ceil(memory / membycore)), cpu)
                memory = membycore
            else:
                newcpu = int(math.ceil(memory / membycore))
                memory = membycore

            if newcpu != cpu:
                self.__log.warning(
                    "The memory job requirements needs to "
                    "change the number of cores needed by the job. "
                    "(%d --> %d)" % (cpu, newcpu))
                cpu = newcpu

        if cpu > 1:
            jobParams.append("#$ -pe make " + str(cpu))

        jobParams.append("#$ -l mem_free=" + str(mem_need) +
                         "G,h_vmem=" + str(memory + 0.2) + "G")

        if maxtime is not None:
            jobParams.append(
                "#$ -l h_rt=" + str(datetime.timedelta(minutes=maxtime)))

        # NS, where elements turns into <FILE>
        if len(elements) > 0:
            self.__log.debug("Num elements submitted " + str(len(elements)))
            self.__log.debug("Num Job submitted " + str(num_jobs))

            input_dict = dict()

            # Yield num-jobs successive chunks of 10 elements from the input array of signatures.
            # If some jobs fail, recover their index from the pickle, '535' :
            # {'REP.A028_YAPC_24H:K01': {'file':
            # '/aloy/web_checker/package_cc/2020_01/full/D/D1/D1.001/sign0/raw/models/signatures/REP.A028_YAPC_24H:K01.h5'},.....}
            for cid, chunk in enumerate(self._chunks(elements, num_jobs), 1):
                input_dict[str(cid)] = chunk

            # i.e a random name input file:
            # d8e918f5-4817-4df5-9ab9-5efbf23f63c7
            input_path = os.path.join(self.jobdir, str(uuid.uuid4()))

            # Write the dictionary of 6000 chunks into it {'1': chunk1,
            # '2':chunk2 etc}
            with open(input_path, 'wb') as fh:
                pickle.dump(input_dict, fh, protocol=2)

            command = command.replace("<FILE>", input_path)

        if cpusafe:
            # set environment variable that limit common libraries cpu
            # ubscription for the command
            env_vars = [
                'OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS',
                'NUMEXPR_NUM_THREADS',
                'NUMEXPR_MAX_THREADS'
            ]
            command = ' '.join(["%s=%s" % (v, str(cpu))
                                for v in env_vars] + [command])

        # Creates the final job.sh
        paramsText = self.defaultOptions + str("\n").join(jobParams)
        jobFilename = os.path.join(
            self.jobdir, sge.jobFilenamePrefix + self.job_name +
            sge.jobFilenameSuffix)

        self.__log.info("Writing file " + jobFilename + "...")
        jobFile = open(jobFilename, "w")
        jobFile.write(sge.templateScript %
                      {"options": paramsText, "command": command})
        jobFile.close()

        os.chmod(jobFilename, 0o755)

        submit_string += jobFilename

        self.__log.debug("HPC submission: " + submit_string)

        time.sleep(2)

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
            self.__log.debug('Job id: %s' % self.job_id)
        except paramiko.SSHException as sshException:
            raise Exception(
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

    def __find_error(self, files):
        errors = ''
        for file_name in files:
            with open(file_name, 'r') as f:
                num = 1
                for line in f:
                    if re.search(r'(?i)error', line):
                        errors += file_name + " " + str(num) + " " + line
                    if 'Traceback (most recent call last)' in line:
                        errors += file_name + " " + str(num) + " " + line
                    num += 1
        return errors

    def check_errors(self):
        """Check for errors in the output logs of the jobs."""
        errors = ''
        self.__log.debug("Checking errors in job")
        files = []
        dir_regex = os.path.join(self.jobdir, self.job_name + '.o*')
        for file_name in glob.glob(dir_regex):
            files.append(file_name)

        errors = self.error_finder(files)

        if len(errors) > 0:
            self.__log.debug("Found errors in job (at directory {})".format(
                os.path.join(self.jobdir)))
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
        """Compress the output logs."""
        self.__log.debug("Compressing output job files...")
        tar = tarfile.open(os.path.join(
            self.jobdir, self.job_name + ".tar.gz"), "w:gz")
        dir_regex = os.path.join(self.jobdir, self.job_name + '.o*')
        for file_name in glob.glob(dir_regex):
            tar.add(file_name, os.path.basename(file_name))
        tar.close()
        for file_name in glob.glob(dir_regex):
            os.remove(file_name)

    def status(self):
        """Gets the status of the job submission."""
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
