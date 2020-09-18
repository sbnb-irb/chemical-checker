"""CCFit task.

This class allows to pipe different ``fit`` tasks in the Pipeline framework.
It tries to work for all possible CC elements but considering that signatures
are changing quite often it might need to be updated.
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

FIT_PARAMS = {

}

MOLSET_FIT = {
    'sign0': ['full'],
    'sign1': ['reference'],
    'sign2': ['reference'],
    'sign3': ['full'],
    'neig1': ['reference'],
    'neig2': ['reference'],
    'neig3': ['reference'],
    'clus1': ['reference'],
    'clus2': ['reference'],
    'clus3': ['reference'],
    'proj1': ['reference'],
    'proj2': ['reference'],
    'proj3': ['reference']
}

BASE_SCRIPT = [
    "import sys, os",
    "import pickle",
    "import logging",
    "import chemicalchecker",
    "from chemicalchecker import ChemicalChecker, Config",
    "logging.log(logging.DEBUG,'chemicalchecker: %s' % chemicalchecker.__path__)",
    "logging.log(logging.DEBUG,'CWD: %s' % os.getcwd())",
    "config = Config()",
    "task_id = sys.argv[1]",  # <TASK_ID>
    "filename = sys.argv[2]",  # <FILE>
    "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
    "ds = inputs[task_id][0][0]",
    "fit_pars = inputs[task_id][0][1]",
    "sign_pars = inputs[task_id][0][1]",
]

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
            full_reference (bool): The fit is for both full and reference
                (default:True)
            ds_data_params (dict): A dictionary where key is dataset code and
                value is  another dictionary with all specific parameters for
                that dataset. (Optional)
            general_data_params (dict): A dictionary with general parameters
                for all datasets (Optional)

            cc_old_path (str): The CC root path for a previous release of CC
                (specific for sign0)
            ref_datasets (list): List of reference datasets for
                (specific for sign3)

        """
        self.name = params.get('name', cctype)
        BaseTask.__init__(self, self.name)

        self.cctype = cctype
        self.cc_root = cc_root
        self.datasets = params.get('datasets', 'essential')
        if self.datasets == 'essential':
            self.datasets = [ds.code for ds in Dataset.get(essential=True)]
        self.ds_data_params = params.get('ds_params', None)
        self.general_data_params = params.get('general_params', None)

        def_ref_datasets = [ds.code for ds in Dataset.get(exemplary=True)]
        self.ref_datasets = params.get('reference_datasets', def_ref_datasets)
        self.cc_old_path = params.get('cc_old_path', None)

        if self.cctype == 'sign0':
            if self.ds_data_params is None and self.cc_old_path is None:
                raise Exception("CCFit for sign0 requires 'cc_old_path' "
                                "if no 'ds_data_params' is provided")

    def run(self):
        """Run the task."""
        # exclude dataset that have been already fitted
        cc = ChemicalChecker(self.cc_root)
        dataset_codes = list()
        for ds in self.datasets:
            sign = cc.get_signature(self.cctype, MOLSET_FIT[self.cctype], ds)
            if not sign.is_fit():
                # special exception to avoid recumputing D1 sign0
                # even if not completed
                if os.path.exists(sign.signature_path):
                    if ds == 'D1.001' and self.cctype == 'sign0':
                        continue
                dataset_codes.append(ds)
        if len(dataset_codes) == 0:
            self.l__log.warninig('All dataset are already fitted.')
            return

        # sign3 specific precalculations
        # FIXME this should go to the main pipeline
        if self.cctype == 'sign3':
            # if target datasets are also in the in the reference list, then
            # we can precumpute the shared universe and coverage matrices
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

        # Filling dataset_params
        # for each dataset we want to define a set of signature parameters
        # (i.e. sign_pars, used when loading the signature) and a set of
        # parameters used when callinf the 'fit' method (i.e. fit_pars)
        # this should be harmonized working on individual signature classes
        dataset_params = list()
        for ds_code in dataset_codes:
            sign_pars = {}
            fit_pars = {}
            # NS self.data_param is None is our case
            if isinstance(self.ds_data_params, Config):
                temp_dict = self.ds_data_params.asdict()
            else:
                temp_dict = self.ds_data_params  # NS i.e None

            if temp_dict is not None and ds_code in temp_dict.keys():
                if isinstance(temp_dict[ds_code], Config):
                    dict_params = temp_dict[ds_code].asdict()
                else:
                    dict_params = temp_dict[ds_code]

                if self.cctype in SPECIAL_PARAMS:
                    dict_params.update(SPECIAL_PARAMS[self.cctype])

                if self.general_data_params is not None:
                    dict_params.update(self.general_data_params)
                dataset_params.append((ds_code, dict_params))
            else:
                dataset_params.append((ds_code, None))  # i.e ('A1.001', None)

        self.__log.info('dataset_params:')
        for p in dataset_params:
            self.__log.info('\t%s' % str(p))
        # NS: checking dependencies depending on which sign we are calculating
        # NS, some CC_types have dependencies, eg sign2 need for ex sign1 and
        # neig1 to work
        for dependency in DEPENDENCIES[self.cctype]:
            for ds in dataset_codes:
                if dependency == self.cctype:  # NS no dependency
                    continue
                if self.full_reference:  # NS: True by default
                    branch = "reference"
                else:
                    branch = "full"

                # again, generate a sign or neig object to fulfill the
                # dependency before the required signature
                sign = cc.get_signature(dependency, branch, ds)

                if not sign.available():
                    raise Exception(
                        dependency + " CC type is not available and it is required for " + self.cctype)
                else:
                    print("INFO: Dependency {} is available for calculating {}".format(
                        dependency, self.cctype))

        if self.cc_old_path is not None:
            SIGN0_SCRIPT_FR.insert(0, "CC_OLD_PATH = '%s'" % self.cc_old_path)

        else:
            SIGN0_SCRIPT_FR.insert(0, "CC_OLD_PATH = None")

        SPECIFIC_SCRIPTS['sign0'] = (SIGN0_SCRIPT_FR, SIGN0_SCRIPT_FR)

        # Create script file that will launch signx fit for each dataset
        job_path = tempfile.mkdtemp(
            prefix='jobs_%s_' % self.cctype, dir=self.tmpdir)
        script_lines = BASE_SCRIPT
        script_lines += [
            "cc = ChemicalChecker('%s')" % self.cc_root,
            'sign_full = cc.get_signature("%s", "full", ds)' % DEPENDENCIES[
                self.cctype][0],
            'sign_ref = cc.get_signature("%s", "reference", ds)' % DEPENDENCIES[
                self.cctype][0]
        ]
        # append signature specific snippets
        if self.full_reference:  # True for sign0
            if self.cctype in SPECIFIC_SCRIPTS:
                # i.e SIGN0_SCRIPT_FR, i.e will call cc.preprocess
                script_lines += SPECIFIC_SCRIPTS[self.cctype][0]
            else:
                script_lines += [sub.replace('<CCTYPE>', self.cctype)
                                 for sub in CC_SCRIPT_FR]
        else:
            if self.cctype in SPECIFIC_SCRIPTS:
                # # same for sign0 i.e SIGN0_SCRIPT_FR, i.e will call cc.preprocess
                script_lines += SPECIFIC_SCRIPTS[self.cctype][1]
            else:
                # here the fit method is called
                script_lines += [sub.replace('<CCTYPE>', self.cctype)
                                 for sub in CC_SCRIPT_F]
        script_lines += ["print('JOB DONE')"]
        script_name = os.path.join(job_path, self.cctype + '_script.py')
        # write script to file
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')

        # HPC job parameters
        params = {}
        params["num_jobs"] = len(dataset_codes)
        params["jobdir"] = job_path
        params["job_name"] = "CC_" + self.cctype.upper()
        params["elements"] = dataset_params
        params["wait"] = True
        params.update(HPC_PARAMS[self.cctype])

        # prepare job command and submit job
        cc_config_path = os.environ['CC_CONFIG']
        cc_package = os.path.join(Config().PATH.CC_REPO, 'package')
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} "
        "singularity exec {} python {} <TASK_ID> <FILE>".format(
            cc_package, cc_config_path, singularity_image, script_name)
        # submit jobs
        cluster = HPC.from_config(Config())
        jobs = cluster.submitMultiJob(command, **params)
        self.__log.info("Job '%s' ended.", str(jobs))

        # Check if signatures are indeed fitted
        dataset_not_done = []
        for ds_code in dataset_codes:
            sign = cc.get_signature(self.cctype, "full", ds_code)
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
