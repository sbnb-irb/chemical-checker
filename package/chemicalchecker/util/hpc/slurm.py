"""Submit jobs to an HPC cluster through SLURM queueing system."""
import os
import re
import glob
import uuid
import time
import pickle
import tarfile
import paramiko
import numpy as np

from chemicalchecker.util import logged

STARTED = "started"
DONE = "done"
READY = "ready"
ERROR = "error"


@logged
class slurm():
    """SLURM job class."""

    jobFilenamePrefix = "job-"
    jobFilenameSuffix = ".sh"

    jobStatusSuffix = ".status"

    templateScript = """\
#!/bin/bash
#
#

#SBATCH -p sbnb_cpu_zen3,irb_cpu_iclk
#SBATCH --nodelist=irbccn[05-08],irbccn16
#SBATCH --nice=2147483645

#SBATCH --time=10-00:00:00
#SBATCH --qos=long

# Options for sbatch
%(options)s
# End of qsub options

# Loads default environment configuration
if [[ -f $HOME/.bashrc ]]
then
  source $HOME/.bashrc
fi

export SINGULARITY_BIND="/home/sbnb:/aloy/home,/data/sbnb/data:/aloy/data,/data/sbnb/scratch:/aloy/scratch,/data/sbnb/chemicalchecker:/aloy/web_checker,/data/sbnb/web_updates:/aloy/web_repository"

%(command)s

    """

    defaultOptions = """\
#$ -S /bin/bash
#$ -r yes
"""

    def __init__(self, **kwargs):
        """Initialize a SLURM instance."""
        self.host = kwargs.get("host", '')
        self.queue = kwargs.get("queue", None)
        self.username = kwargs.get("username", '')
        self.password = kwargs.get("password", '')
        self.error_finder = kwargs.get("error_finder", self.__find_error)
        dry_run = kwargs.get("dry_run", False)
        self.statusFile = None
        self.status_id = None
        self.conn_params = {}

        if self.username != '' and self.password != '':
            self.conn_params["username"] = self.username
            self.conn_params["password"] = self.password
        if not dry_run:
            try:
                ssh_config = paramiko.SSHConfig()
                user_config_file = os.path.expanduser("~/.ssh/config")
                if os.path.exists(user_config_file):
                    with open(user_config_file) as f:
                        ssh_config.parse(f)
                cfg = ssh_config.lookup(self.host)
                ssh = paramiko.SSHClient()
                ssh.load_system_host_keys()
                #ssh.connect(cfg['hostname'], username=cfg['user'], key_filename=cfg['identityfile'][0])
                ssh.connect(self.host, **self.conn_params)
            except paramiko.SSHException as sshException:
                self.__log.warning(
                    "Unable to establish SSH connection: %s" % sshException)
            finally:
                ssh.close()

    def _chunks(self, l, n):
        """Yields n successive chunks from l."""
        if isinstance(l, list) or isinstance(l, np.ndarray):
            # a list of 60 000 entries split in 6000 chunks
            ind = list( range( len(l) ) )
            for i in np.array_split(ind, n):
                tmp = []
                for idx in i:
                    tmp.append( l[idx] )
                yield tmp   # yields one of the 6000 chunks of 10 elements
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
        num_jobs = kwargs.get("num_jobs", 1)
        cpu = kwargs.get("cpu", 1)
        wait = kwargs.get("wait", True)
        self.jobdir = kwargs.get("jobdir", '')
        self.job_name = kwargs.get("job_name", 'hpc_cc_job')
        elements = kwargs.get("elements", [])
        compress_out = kwargs.get("compress", True)
        check_error = kwargs.get("check_error", True)
        #memory = kwargs.get("memory", 2)
        maxtime = kwargs.get("time", None)
        cpusafe = kwargs.get("cpusafe", True)
        membycore = int(kwargs.get("mem_by_core", None))
        custom_elements = kwargs.get("custom_chunks", [])
        
        submit_string = 'sbatch --parsable '
            
        if wait:
            submit_string += " --wait "

        self.__log.debug("Job name is: " + self.job_name)

        if not os.path.exists(self.jobdir):
            os.makedirs(self.jobdir)

        jobParams = ["#SBATCH -J " + self.job_name]
        jobParams.append("#SBATCH --chdir=" + self.jobdir)

        if( membycore != None ):
            jobParams.append( f"#SBATCH --mem-per-cpu={ membycore }G" )

        if ( len(custom_elements) == 0 and len(elements) == 0 and num_jobs > 1):
            raise Exception(
                "Number of specified jobs does not match to the number of elements")

        if num_jobs == 0:
            raise Exception("Number of specified jobs is zero")

        if num_jobs > 1 or command.find("<TASK_ID>") != -1:
            jobParams.append("#SBATCH --array=1-" + str(num_jobs))
            tmpname = command.replace("<TASK_ID>", "$SLURM_ARRAY_TASK_ID")
            command = tmpname

        if cpu > 1:
            jobParams.append("#SBATCH --nodes=1")
            jobParams.append("#SBATCH --ntasks=" + str(cpu))

        #if memory > 1:
        #    jobParams.append("#SBATCH --mem=" + str(memory) + "G")

        if maxtime is not None:
            jobParams.append(
                "#SBATCH --time=" + str(maxtime))

        input_dict = dict()
        if len(elements) > 0:
            self.__log.debug("Num elements submitted " + str(len(elements)))
            self.__log.debug("Num Job submitted " + str(num_jobs))

            for cid, chunk in enumerate(self._chunks(elements, num_jobs), 1):
                input_dict[str(cid)] = chunk
            
        if( len(custom_elements) > 0 ):
            input_dict = custom_elements
        
        if( len(elements) > 0 or len(custom_elements) > 0 ):    
            input_path = os.path.join(self.jobdir, str(uuid.uuid4()))
            with open(input_path, 'wb') as fh:
                pickle.dump(input_dict, fh)
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
        paramsText = str("\n").join(jobParams)
        jobFilename = os.path.join(
            self.jobdir, slurm.jobFilenamePrefix + self.job_name + slurm.jobFilenameSuffix)
        self.__log.info("Writing file " + jobFilename + "...")
        jobFile = open(jobFilename, "w")
        jobFile.write(slurm.templateScript %
                      {"options": paramsText, "command": command})
        jobFile.close()

        os.chmod(jobFilename, 0o755)

        submit_string += jobFilename

        self.__log.debug("HPC submission: " + submit_string)

        time.sleep(2)

        try:
            ssh_config = paramiko.SSHConfig()
            user_config_file = os.path.expanduser("~/.ssh/config")
            if os.path.exists(user_config_file):
                with open(user_config_file) as f:
                    ssh_config.parse(f)
            cfg = ssh_config.lookup(self.host)
            ssh = paramiko.SSHClient()
            ssh.load_system_host_keys()
            #ssh.connect(cfg['hostname'], username=cfg['user'], key_filename=cfg['identityfile'][0])
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
                self.jobdir, self.job_name + slurm.jobStatusSuffix)
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
        files = []
        for file_name in glob.glob(os.path.join(self.jobdir, 'slurm-*.out')):
            files.append(file_name)

        errors = self.error_finder(files)

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
        """Compress the output logs."""
        self.__log.debug("Compressing output job files...")
        tar = tarfile.open(os.path.join(
            self.jobdir, self.job_name + ".tar.gz"), "w:gz")
        for file_name in glob.glob(os.path.join(self.jobdir, 'slurm-*.out')):
            tar.add(file_name, os.path.basename(file_name))
        tar.close()
        for file_name in glob.glob(os.path.join(self.jobdir, 'slurm-*.out')):
            os.remove(file_name)

    def status(self):
        """Gets the status of the job submission."""
        if self.statusFile is None:
            return None

        if self.status_id == STARTED:
            try:
                ssh_config = paramiko.SSHConfig()
                user_config_file = os.path.expanduser("~/.ssh/config")
                if os.path.exists(user_config_file):
                    with open(user_config_file) as f:
                        ssh_config.parse(f)
                cfg = ssh_config.lookup(self.host)
                ssh = paramiko.SSHClient()
                ssh.load_system_host_keys()
                ssh.connect( self.host, **self.conn_params )
                stdin, stdout, stderr = ssh.exec_command(
                    'squeue --job ' + self.job_id)

                message = stdout.readlines()

                self.__log.debug(message)

                # if message[0].find("do not exist") != -1:
                if len(message) == 1:
                    self.status_id = DONE
                    with open(self.statusFile, "w") as f:
                        f.write(self.status_id)
            except paramiko.SSHException as sshException:
                self.__log.warning(
                    "Unable to establish SSH connection: %s" % sshException)
            finally:
                ssh.close()

        return self.status_id
