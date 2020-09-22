"""CCFit task.

This task allow submitting HPC jobs which will 'fit' for a specific signature
type (e.g. 'sign1').
It allows submitting jobs for all spaces of the CC at once and passing
parameters specific for each of them.
We should avoid adding too much logic in here, and simply pass args and
kwargs to signatures. The specific signature classes should check that
the parameters are OK.
"""
import os
import h5py
import shutil
import tempfile
import numpy as np

from chemicalchecker.database import Dataset
from chemicalchecker.core.sign3 import sign3
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util.pipeline import BaseTask
from chemicalchecker.util import logged, Config, HPC


VALID_TYPES = ['sign', 'neig', 'clus', 'proj', 'diag']

DEPENDENCIES = {
    'sign0': ['sign0'],
    'sign1': ['sign0'],
    'sign2': ['sign1', 'neig1'],
    'sign3': ['sign2'],
    'neig1': ['sign1'],
    'neig2': ['sign2'],
    'neig3': ['sign3'],
    'clus1': ['sign1'],
    'clus2': ['sign2'],
    'clus3': ['sign3'],
    'proj1': ['sign1'],
    'proj2': ['sign2'],
    'proj3': ['sign3']
}

HPC_PARAMS = {
    'sign0': {'memory': 44, 'cpu': 22},
    'sign1': {'memory': 40, 'cpu': 10},
    'sign2': {'memory': 20, 'cpu': 16},
    'sign3': {'memory': 2,  'cpu': 32},
    'neig1': {'memory': 30, 'cpu': 15},
    'neig2': {'memory': 30, 'cpu': 15},
    'neig3': {'memory': 30, 'cpu': 15},
    'clus1': {'memory': 20, 'cpu': 10},
    'clus2': {'memory': 20, 'cpu': 10},
    'clus3': {'memory': 20, 'cpu': 10},
    'proj1': {'memory': 20, 'cpu': 10},
    'proj2': {'memory': 20, 'cpu': 10},
    'proj3': {'memory': 20, 'cpu': 10}
}

SIGN_PARAMS = {
    'sign2': {'adanet': {'cpu': 16}, 'node2vec': {'cpu': 4}},
    'sign3': {'cpu': 32},
    'neig1': {'cpu': 15},
    'neig2': {'cpu': 15},
    'neig3': {'cpu': 15},
    'clus1': {'cpu': 10},
    'clus2': {'cpu': 10},
    'clus3': {'cpu': 10},
    'proj1': {'cpu': 10},
    'proj2': {'cpu': 10},
    'proj3': {'cpu': 10}
}

MOLSET_FIT = {
    'sign0': 'full',
    'sign1': 'full',
    'sign2': 'reference',
    'sign3': 'full',
    'neig1': 'reference',
    'neig2': 'reference',
    'neig3': 'reference',
    'clus1': 'reference',
    'clus2': 'reference',
    'clus3': 'reference',
    'proj1': 'reference',
    'proj2': 'reference',
    'proj3': 'reference'
}

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

"""
CC_SCRIPT_FR = [
    'sign_new_ref = cc.get_signature("<CCTYPE>", "reference", ds, **sign_pars)',
    "sign_new_ref.fit(sign_ref, **fit_pars)",
    "sign_new_full = cc.get_signature('<CCTYPE>', 'full', ds, **sign_pars)",
    "sign_new_ref.predict(sign_full, destination=sign_new_full.data_path)",
    "sign_new_full.mark_ready()"
]

CC_SCRIPT_F = [
    'sign_new_full = cc.get_signature("<CCTYPE>", "full", ds, **sign_pars)',
    "sign_new_full.fit(sign_full)"
    "if sign_new_full.cctype.startswith('sign'):",
    '    diag=cc.diagnosis(sign_new_full)',
    '    diag.canvas()'
]

SIGN2_SCRIPT_FR = [
    'neig1_ref = cc.get_signature("neig1", "reference", ds)',
    'sign2_ref = cc.get_signature("sign2", "reference", ds, **sign_pars)',
    "sign2_ref.fit(sign_ref, neig1_ref, reuse=False)",
    "sign2_full = cc.get_signature('sign2', 'full', ds, **sign_pars)",
    "sign2_ref.predict(sign_full, destination=sign2_full.data_path)",
    "sign2_full.validate()",
    "sign2_full.mark_ready()"
]

SIGN2_SCRIPT_F = [
    'neig1_full = cc.get_signature("neig1", "full", ds)',
    'sign2_full = cc.get_signature("sign2", "full", ds, **sign_pars)',
    "sign2_full.fit(sign_full, neig1_full, reuse=False)"
]

SIGN3_SCRIPT_F = [
    "sign1_full = cc.get_signature('sign1', 'full', ds, **sign_pars)",
    "sign3_full = cc.get_signature('sign3', 'full', ds, **sign_pars)",
    "sign2_list = [cc.get_signature('sign2', 'full', ds) for ds in sign2_src_list]",
    "sign2_list.append(cc.get_signature('sign2', 'full', ds))",
    "sign3_full.fit(sign2_list, sign_full, sign1_full)",
    'diag=cc.diagnosis(sign3_full)',
    'diag.canvas()'
]

SIGN0_SCRIPT_FR = [
    "if len(fit_pars) == 0:",
    "    prepro_file = cc.preprocess(sign_full)",
    "    cc_old = ChemicalChecker(CC_OLD_PATH)",
    "    pars['data_file'] = prepro_file",
    "    pars['cc'] = cc_old",
    "sign_full.fit(**fit_pars)",
    'diag=cc.diagnosis(sign_full)',
    'diag.canvas()'
]

SIGN1_SCRIPT_FR = [
    'sign_new_full = cc.get_signature("sign1", "full", ds)',
    'sign_new_full.fit(sign_full, **fit_pars)'
    'diag=cc.diagnosis(sign_new_full)',
    'diag.canvas()'
]


SPECIFIC_SCRIPTS = {
    'sign1': (SIGN1_SCRIPT_FR, SIGN1_SCRIPT_FR),
    'sign2': (SIGN2_SCRIPT_FR, SIGN2_SCRIPT_F),
}


SIGN3_SCRIPT_FR = [
    "sign1_full = cc.get_signature('sign1', 'full', data, **sign_pars)",
    "sign3_full = cc.get_signature('sign3', 'full', data, **sign_pars)",
    "sign2_src_list = [%s]" % (str(self.ref_datasets)[1:-1]),
    "sign2_list = [cc.get_signature('sign2', 'full', ds) for ds in sign2_src_list]",
    "sign3_full.fit(sign2_list,sign_full,sign1_full,sign2_universe='%s', sign2_coverage='%s')" % (
        full_universe, full_coverage)
]
SPECIFIC_SCRIPTS['sign3'] = (
    SIGN3_SCRIPT_FR, ["sign2_src_list = [%s]" %
                      (str(self.ref_datasets)[1:-1])] + SIGN3_SCRIPT_F)

"""


@logged
class CCFit(BaseTask):

    def __init__(self, cc_root, cctype, **params):
        """Initialize CC fit task.

        Args:
            cc_root (str): The CC root path (Required)
            cctype (str): The CC type where the fit is applied (Required)
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
            sign = cc.get_signature(self.cctype, MOLSET_FIT[self.cctype], ds)
            if sign.is_fit():
                continue
            # special exception to avoid recomputing from the beginning
            # sign0 of D1.001, even if not completed
            if os.path.exists(sign.signature_path):
                if ds == 'D1.001' and self.cctype == 'sign0':
                    continue
            dataset_codes.append(ds)
        if len(dataset_codes) == 0:
            self.__log.warning('All dataset are already fitted.')
            return

        # sign3 specific pre-calculations
        # FIXME this should go to the main pipeline
        if self.cctype == 'sign3':
            # if target datasets are also in the in the reference list, then
            # we can precompute the shared universe and coverage matrices
            # otherwise the sign3 class is computing for each dataset
            if all(np.isin(self.datasets, self.ref_datasets)):
                full_universe = os.path.join(self.tmpdir, "universe_full")
                full_coverage = os.path.join(self.tmpdir, "coverage_full")
                sign2_list = [cc.get_signature('sign2', 'full', ds)
                              for ds in self.ref_datasets]
                sign3.save_sign2_universe(sign2_list, full_universe)
                sign3.save_sign2_coverage(sign2_list, full_coverage)
                # check for universe to be readable
                try:
                    with h5py.File(full_universe, 'r') as hf:
                        hf.keys()
                except Exception as e:
                    self.__log.error(e)
                    raise Exception("Universe full file is corrupted.")

        # Preparing dataset_params
        # for each dataset we want to define a set of signature parameters
        # (i.e. sign_pars, used when loading the signature) and a set of
        # parameters used when calling the 'fit' method (i.e. fit_pars)
        # FIXME this should be harmonized fixing individual signature classes
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
            sign_args.insert(1, MOLSET_FIT[self.cctype])
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
        params.update(self.hpc_kwargs)

        # prepare job command and submit job
        cc_config_path = os.environ['CC_CONFIG']
        cc_package = os.path.join(Config().PATH.CC_REPO, 'package')
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = ("SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} "
                   "singularity exec {} python {} <TASK_ID> <FILE>").format(
            cc_package, cc_config_path, singularity_image, script_name)
        self.__log.debug('CMD CCFIT: %s', command)
        # submit jobs
        cluster = HPC.from_config(Config())
        jobs = cluster.submitMultiJob(command, **params)
        self.__log.info("Job with jobid '%s' ended.", str(jobs))

        # Check if signatures are indeed fitted
        dataset_not_done = []
        for ds_code in dataset_codes:
            sign = cc.get_signature(
                self.cctype, MOLSET_FIT[self.cctype], ds_code)
            if sign.is_fit():
                continue
            dataset_not_done.append(ds_code)
            self.__log.warning(
                self.cctype + " fit failed for dataset code: " + ds_code)

        if len(dataset_not_done) == 0:
            self.mark_ready()
            self.__log.info("Deleting job path: %s", job_path)
            shutil.rmtree(job_path, ignore_errors=True)
        else:
            if not self.custom_ready():
                raise Exception("Not all dataset fits are done")

    def execute(self, context):
        """Same as run but for Airflow."""
        self.tmpdir = context['params']['tmpdir']
        self.run()
