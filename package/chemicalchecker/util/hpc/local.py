"""Run job in local system."""
import os
import re
import glob
import uuid
import pickle
import tarfile
import subprocess
import numpy as np

from chemicalchecker.util import logged

STARTED = "started"
DONE = "done"
READY = "ready"
ERROR = "error"


@logged
class local():
    """Local job class."""

    jobStatusSuffix = ".status"

    def __init__(self, **kwargs):
        """Initialize the local class."""
        self.error_finder = kwargs.get("error_finder", self.__find_error)
        self.statusFile = None
        self.status_id = None

    def _chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        if isinstance(l, list) or isinstance(l, np.ndarray):
            ind = list( range( len(l) ) )
            for i in np.array_split(ind, n):
                tmp = []
                for idx in i:
                    tmp.append( l[idx] )
                yield tmp
        elif isinstance(l, dict):
            keys = list(l.keys())
            for i in np.array_split(keys, n):
                yield {k: l[k] for k in i}
        else:
            raise Exception("Element datatype not supported: %s" % type(l))

    def submitMultiJob(self, command, **kwargs):
        """Submit multiple job/task."""
        # get arguments or default values
        num_jobs = int(kwargs.get("num_jobs", 1))
        self.jobdir = kwargs.get("jobdir", '')
        self.job_name = kwargs.get("job_name", 'hpc_cc_job')
        elements = kwargs.get("elements", [])
        compress_out = kwargs.get("compress", True)
        check_error = kwargs.get("check_error", True)
        cpu = 2
        cpusafe = kwargs.get("cpusafe", True)

        # Remove the call to singularity since we are already in a singularity
        # image
        cmd_split = command.split()
        if 'singularity' in cmd_split:
            sidx = cmd_split.index('singularity')
            cmd_split_tmp = cmd_split[:sidx] + cmd_split[sidx + 3:]
            command = ' '.join(cmd_split_tmp)
            command = command.replace('SINGULARITYENV_', '')

        self.__log.debug("Job name is: " + self.job_name)

        if not os.path.exists(self.jobdir):
            os.makedirs(self.jobdir)

        if (len(elements) == 0 and num_jobs > 1):
            raise Exception(
                "Number of specified jobs does not match to the number of elements")

        if num_jobs == 0:
            raise Exception("Number of specified jobs is zero")

        if len(elements) > 0:
            self.__log.debug("Num elements submitted " + str(len(elements)))

            input_dict = dict()
            for cid, chunk in enumerate(self._chunks(elements, num_jobs), 1):
                input_dict[str(cid)] = chunk
            input_path = os.path.join(self.jobdir, str(uuid.uuid4()))
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
                'NUMEXPR_NUM_THREADS'
            ]
            command = ' '.join(["%s=%s" % (v, str(cpu))
                                for v in env_vars] + [command])

        # Creates the final job.sh
        self.__log.info("Writing script file for in dir " + self.jobdir)

        for i in range(num_jobs):
            cmd_run = command.replace("<TASK_ID>", str(i + 1))
            self.__log.info("Running job %d/%d", i + 1, num_jobs)
            self.__log.debug("CMD: %s", cmd_run)
            with open(os.path.join(self.jobdir, "log" + str(i + 1) + ".txt"), 'w') as f:
                subprocess.call([cmd_run], stdout=f, stderr=f, shell=True)

        self.statusFile = os.path.join(
            self.jobdir, self.job_name + self.jobStatusSuffix)
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

    def __find_error(self, files, print_logs):
        errors = ''
        for file_name in files:
            if print_logs:
                self.__log.debug('*' * 40)
                self.__log.debug('*   START log for: %s', file_name)
                self.__log.debug('*' * 40)
            with open(file_name, 'r') as f:
                num = 1
                for line in f:
                    print('*  ' + line.strip())
                    if re.search(r'(?i)error', line):
                        errors += file_name + " " + str(num) + " " + line
                    if 'Traceback (most recent call last)' in line:
                        errors += file_name + " " + str(num) + " " + line
                    num += 1
            if print_logs:
                self.__log.debug('*' * 40)
                self.__log.debug('*   END log for: %s', file_name)
                self.__log.debug('*' * 40)
        return errors

    def check_errors(self, print_logs=True):
        """Check for errors in the output logs of the jobs."""
        errors = ''
        self.__log.debug("Checking errors in job")
        files = []
        for file_name in glob.glob(os.path.join(self.jobdir, 'log*')):
            files.append(file_name)

        errors = self.error_finder(files, print_logs)

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
        for file_name in glob.glob(os.path.join(self.jobdir, 'log*')):
            tar.add(file_name, os.path.basename(file_name))
        tar.close()
        for file_name in glob.glob(os.path.join(self.jobdir, 'log*')):
            os.remove(file_name)

    def status(self):
        """Gets the status of the job submission."""
        if self.statusFile is None:
            return None
        return self.status_id
