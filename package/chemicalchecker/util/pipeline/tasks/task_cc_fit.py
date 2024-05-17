"""CCFit task.

This task allow submitting HPC jobs which will call the `fit` method
for a specific signature type (e.g. 'sign1').
It allows submitting jobs for all spaces of the CC at once and passing
parameters specific for each of them.
We should avoid adding too much logic in here, and simply pass `args` and
`kwargs` to signatures. The specific signature classes should check that
the parameters are OK.
"""
import os
import shutil
import tempfile

from chemicalchecker.database import Dataset
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util.pipeline import BaseTask
from chemicalchecker.util import logged, HPC

VALID_TYPES = ['sign', 'neig', 'clus', 'proj']

FIT_SCRIPT = """
import sys
import os
import pickle
import logging

import chemicalchecker
from chemicalchecker import ChemicalChecker, Config

logging.log(logging.DEBUG, 'chemicalchecker: {{}}'.format(
    chemicalchecker.__path__))
logging.log(logging.DEBUG, 'CWD: {{}}'.format(os.getcwd()))
config = Config()
task_id = sys.argv[1]  # <TASK_ID>
filename = sys.argv[2]  # <FILE>
inputs = pickle.load(open(filename, 'rb'))  # load pickled data
sign_args = inputs[task_id][0][0]
sign_kwargs = inputs[task_id][0][1]
fit_args = inputs[task_id][0][2]
fit_kwargs = inputs[task_id][0][3]
cc = ChemicalChecker('{cc_root}')
sign = cc.get_signature(*sign_args, **sign_kwargs)
sign.fit(*fit_args, **fit_kwargs)
print('JOB DONE')
"""


@logged
class CCFit(BaseTask):

    def __init__(self, cc_root, cctype, molset, **params):
        """Initialize CC fit task.

        Args:
            cc_root (str): The CC root path (Required)
            cctype (str): The CC type where the fit is applied (Required)
            molset (str): The signature molset (e.g. `full` or `reference`)
                on which the `fit` method will be called.
            name (str): The name of the task (default:cctype)
            datasets (list): The list of dataset codes to apply the fit
                (Optional, by default 'essential' which includes all essential
                CC datasets)
            sign_args (dict): A dictionary where key is dataset code and
                value is a list with all dataset specific parameters for
                initializing the signature. (Optional)
            sign_kwargs (dict): A dictionary where key is dataset code and
                value is a dictionary with all dataset specific key-worded
                parameters for initializing the signature. (Optional)
            fit_args (dict): A dictionary where key is dataset code and
                value is a list with all dataset specific parameters for
                calling the signature `fit` method. (Optional)
            fit_kwargs (dict): A dictionary where key is dataset code and
                value is a dictionary with all dataset specific key-worded
                calling the signature `fit` method. (Optional)
            hpc_kwargs (dict): A dictionary where key is dataset code and
                value is a dictionary with key-worded parameters for the
                `HPC` module. (Optional)

            ref_datasets (list): List of reference datasets for fitting sign3.
                (specific for `sign3`)

        """
        if not any([cctype.startswith(t) for t in VALID_TYPES]):
            raise Exception("cctype '%s' is not recognized.")

        self.name = params.get('name', cctype)
        BaseTask.__init__(self, self.name)
        self.cctype = cctype
        self.cc_root = cc_root
        self.molset = molset
        self.datasets = params.get('datasets', 'essential')
        if self.datasets == 'essential':
            self.datasets = [ds.code for ds in Dataset.get(essential=True)]
        self.sign_args = params.get('sign_args', {})
        self.sign_kwargs = params.get('sign_kwargs', {})
        self.fit_args = params.get('fit_args', {})
        self.fit_kwargs = params.get('fit_kwargs', {})
        self.hpc_kwargs = params.get('hpc_kwargs', {})

        def_ref_datasets = [ds.code for ds in Dataset.get(exemplary=True)]
        self.ref_datasets = params.get('reference_datasets', def_ref_datasets)

    def run(self):
        """Run the task."""
        # exclude dataset that have been already fitted
        cc = ChemicalChecker(self.cc_root)
        dataset_codes = list()
        for ds in self.datasets:
            sign = cc.get_signature(self.cctype, self.molset, ds)
            if sign.is_fit():
                continue
            dataset_codes.append(ds)
        if len(dataset_codes) == 0:
            self.__log.warning('All dataset are already fitted.')
            self.mark_ready()
            return
        
        # Preparing dataset_params
        # for each dataset we want to define a set of signature parameters
        # (i.e. sign_pars, used when loading the signature) and a set of
        # parameters used when calling the 'fit' method (i.e. fit_pars)
        # FIXME can be further harmonized fixing individual signature classes
        dataset_params = list()
        for ds_code in dataset_codes:
            sign_args = list()
            fit_args = list()
            sign_args.extend(self.sign_args.get(ds_code, list()))
            fit_args.extend(self.fit_args.get(ds_code, list()))
            sign_kwargs = dict()
            fit_kwargs = dict()
            sign_kwargs.update(self.sign_kwargs.get(ds_code, dict()))
            fit_kwargs.update(self.fit_kwargs.get(ds_code, dict()))
            # we add arguments which are used by CCFit but are also needed
            # by a signature fit
            sign_args.insert(0, self.cctype)
            sign_args.insert(1, self.molset)
            sign_args.insert(2, ds_code)
            # prepare it as tuple that will be serialized
            dataset_params.append(
                (sign_args, sign_kwargs, fit_args, fit_kwargs))
            self.__log.info('%s sign_args: %s', ds_code, str(sign_args))
            self.__log.info('%s sign_kwargs: %s', ds_code, str(sign_kwargs))
            self.__log.info('%s fit_args: %s', ds_code, str(fit_args))
            self.__log.info('%s fit_kwargs: %s', ds_code, str(fit_kwargs))

        # Create script file that will launch signx fit for each dataset
        job_path = tempfile.mkdtemp(
            prefix='jobs_%s_' % self.cctype, dir=self.tmpdir)
        script_name = os.path.join(job_path, self.cctype + '_script.py')
        script_content = FIT_SCRIPT.format(cc_root=self.cc_root)
        with open(script_name, 'w') as fh:
            fh.write(script_content)

        # HPC job parameters
        params = {}
        params["num_jobs"] = len(dataset_codes)
        params["jobdir"] = job_path
        params["job_name"] = "CC_" + self.cctype.upper()
        params["elements"] = dataset_params
        params["wait"] = True
        params["memory"] = 80
        params.update(self.hpc_kwargs)

        # prepare job command and submit job
        cc_config_path = self.config.config_path
        cc_package = os.path.join(self.config.PATH.CC_REPO, 'package')
        singularity_image = self.config.PATH.SINGULARITY_IMAGE
        command = ("SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} "
                   "singularity exec {} python {} <TASK_ID> <FILE>").format(
            cc_package, cc_config_path, singularity_image, script_name)
        self.__log.debug('CMD CCFIT: %s', command)
        # submit jobs
        cluster = HPC.from_config(self.config)
        jobs = cluster.submitMultiJob(command, **params)
        self.__log.info("Job with jobid '%s' ended.", str(jobs))

        # Check if signatures are indeed fitted
        dataset_not_done = []
        for ds_code in dataset_codes:
            sign = cc.get_signature(
                self.cctype, self.molset, ds_code)
            if sign.is_fit():
                continue
            dataset_not_done.append(ds_code)
            self.__log.warning(
                self.cctype + " fit failed for dataset code: " + ds_code)

        if len(dataset_not_done) == 0:
            self.mark_ready()
            if not self.keep_jobs:
                self.__log.info("Deleting job path: %s", job_path)
                shutil.rmtree(job_path, ignore_errors=True)
        else:
            if not self.custom_ready():
                raise Exception("Not all dataset fits are done")

    def execute(self, context):
        """Same as run but for Airflow."""
        self.tmpdir = context['params']['tmpdir']
        self.run()
