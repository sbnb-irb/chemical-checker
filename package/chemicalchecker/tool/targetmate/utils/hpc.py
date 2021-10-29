import os

import tempfile
from chemicalchecker.util import Config
from chemicalchecker.util.hpc import HPC
from chemicalchecker.util import logged
import multiprocessing
import numpy as np
import time
import pickle

@logged
class HPCUtils:

    def __init__(self, **kwargs):
        self.job_paths = []

    @staticmethod
    def cpu_count():
        return multiprocessing.cpu_count()

    def waiter(self, jobs, secs=3):
        """Wait for jobs to finish"""
        if not jobs: return
        self.__log.info("Waiting for jobs to finish...")
        while np.any([job.status() != "done" for job in jobs]):
            for job in jobs:
                self.__log.debug(job.status())
            time.sleep(secs)
        self.__log.info("Jobs done.")

    def func_hpc(self, func_name, *args, **kwargs):
        """Execute the *any* method on the configured HPC.

        Args:
            args(tuple): the arguments for of the function method
            kwargs(dict): arguments for the HPC method.
        """
        # read config file
        cc_config = kwargs.get("cc_config", os.environ['CC_CONFIG'])
        cfg = Config(cc_config)
        # create job directory if not available
        job_base_path = kwargs.get("job_base_path", cfg.PATH.CC_TMP)
        tmp_dir = tempfile.mktemp(prefix='tmp_', dir=job_base_path)
        job_path = kwargs.get("job_path", tmp_dir)
        self.job_paths += [job_path]
        if not os.path.isdir(job_path):
            os.makedirs(job_path, exist_ok=True)
        # check cpus
        cpu = kwargs.get("cpu", 1)
        memory = kwargs.get("memory", int(cpu*2))
        # create script file
        script_lines = [
            "import os, sys",
            "import pickle",
            "with open(sys.argv[1], 'rb') as f:"
            "    obj, args = pickle.load(f)",
            "obj.%s(*args)" % func_name,
            "print('JOB DONE')"
        ]
        script_name = '%s_%s_hpc.py' % (self.__class__.__name__, func_name)
        script_path = os.path.join(job_path, script_name)
        with open(script_path, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # pickle self and fit args
        pickle_file = '%s_%s_hpc.pkl' % (self.__class__.__name__, func_name)
        pickle_path = os.path.join(job_path, pickle_file)
        with open(pickle_path, "wb") as f:
            try:
                pickle.dump((self, args), f)
            except:
                pickle.dump((self, args), f, protocol = 4) #Added by Paula: Protocol 4 allows pickling of larger data objects. Consider only using this protocol in case of large object (see difference in in file size) 11/12/20
        # hpc parameters
        params = kwargs
        params["num_jobs"] = 1
        params["jobdir"] = job_path
        params["job_name"] = script_name
        params["cpu"] = cpu
        params["memory"] = memory
        params["wait"] = False
        # job command
        singularity_image = cfg.PATH.SINGULARITY_IMAGE
        command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={}" +\
            " singularity exec --bind /aloy/web_checker/:/aloy/web_checker {} python {} {}"
        command = command.format(
            os.path.join(cfg.PATH.CC_REPO, 'package'), cc_config,
            singularity_image, script_name, pickle_file)
        # submit jobs

        cluster = HPC.from_config(cfg)
        cluster.submitMultiJob(command, **params)
        return cluster
