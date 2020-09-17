"""CCFit task.

This class allows to pipe different ``fit`` tasks in the Pipeline framework.
It tries to work for all possible CC elements but considering that signatures
are changing quite often it might need to be updated.
"""
import os
import h5py
import shutil
import tempfile

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
    'proj3': ['sign3'],
    'diag0': ['sign0'],
    'diag1': ['sign1']
}

# NS: changed sign1 requirement from (20,10) to (40,10)
MEM_CPU = {
    'sign0': (44, 22),
    'sign1': (40, 10),
    'sign2': (20, 16),
    'sign3': (2, 32),
    'neig1': (30, 15),
    'neig2': (30, 15),
    'neig3': (30, 15),
    'clus1': (20, 10),
    'clus2': (20, 10),
    'clus3': (20, 10),
    'proj1': (20, 10),
    'proj2': (20, 10),
    'proj3': (20, 10),
    'diag0': (20, 10),
    'diag1': (20, 10)
}

SPECIAL_PARAMS = {
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
    'proj3': {'cpu': 10},
    'diag0': {'cpu': 10},
    'diag1': {'cpu': 10},
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
    'proj3': ['reference'],
    'diag1': ['reference']
}

CC_SCRIPT_FR = [
    'sign_new_ref = cc.get_signature("<CC_TYPE>", "reference", data,**pars)',
    "sign_new_ref.fit(sign_ref)",
    "sign_new_full = cc.get_signature('<CC_TYPE>', 'full', data,**pars)",
    "sign_new_ref.predict(sign_full, destination=sign_new_full.data_path)",
    "sign_new_full.mark_ready()"
]

CC_SCRIPT_F = [
    'sign_new_full = cc.get_signature("<CC_TYPE>", "full", data,**pars)',
    "sign_new_full.fit(sign_full)"
]

SIGN2_SCRIPT_FR = [
    'neig1_ref = cc.get_signature("neig1", "reference", data)',
    'sign2_ref = cc.get_signature("sign2", "reference", data,**pars)',
    "sign2_ref.fit(sign_ref, neig1_ref, reuse=False)",
    "sign2_full = cc.get_signature('sign2', 'full', data,**pars)",
    "sign2_ref.predict(sign_full, destination=sign2_full.data_path)",
    "sign2_full.validate()",
    "sign2_full.mark_ready()"
]

SIGN2_SCRIPT_F = [
    'neig1_full = cc.get_signature("neig1", "full", data)',
    'sign2_full = cc.get_signature("sign2", "full", data,**pars)',
    "sign2_full.fit(sign_full, neig1_full, reuse=False)"
]

SIGN3_SCRIPT_F = [
    "sign1_full = cc.get_signature('sign1', 'full', data,**pars)",
    "sign3_full = cc.get_signature('sign3', 'full', data,**pars)",
    "sign2_list = [cc.get_signature('sign2', 'full', ds) for ds in sign2_src_list]",
    "sign2_list.append(cc.get_signature('sign2', 'full', data))",
    "sign3_full.fit(sign2_list, sign_full, sign1_full)"
]

SIGN0_SCRIPT_FR = [
    "if len(pars) == 0:",
    "    prepro_file = cc.preprocess(sign_full)",
    "    cc_old = ChemicalChecker(CC_OLD_PATH)",
    "    pars['data_file'] = prepro_file",
    "    pars['cc'] = cc_old",
    "sign_full.fit(**pars)"
]

SIGN1_SCRIPT_FR = [
    'sign_new_full = cc.get_signature("sign1", "full", data)',
    'sign_new_full.fit(sign_full, **pars)'
]

DIAG0_SCRIPT_FR = [
    's0 = cc.get_signature("sign0", "full", data)',
    'diag=cc.diagnosis(s0)',
    'diag.canvas()'
]

DIAG1_SCRIPT_FR = [
    's1 = cc.get_signature("sign1", "full", data)',
    'diag=cc.diagnosis(s1)',
    'diag.canvas()'
]

SPECIFIC_SCRIPTS = {
    'sign1': (SIGN1_SCRIPT_FR, SIGN1_SCRIPT_FR),
    'sign2': (SIGN2_SCRIPT_FR, SIGN2_SCRIPT_F),
    'diag0': (DIAG0_SCRIPT_FR, DIAG0_SCRIPT_FR),
    'diag1': (DIAG1_SCRIPT_FR, DIAG1_SCRIPT_FR)
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
        cc = ChemicalChecker(self.cc_root)
        # exclude dataset that have been already fitted
        dataset_codes = list()
        for ds in self.datasets:
            sign = cc.get_signature(self.cctype, MOLSET_FIT[self.cctype], ds)
            if not sign.is_fit():
                dataset_codes.append(ds)

        # sign3 specific precalculations
        if self.cctype == 'sign3':

            # precompute universe and coverage (that are shared)
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
                raise Exception("Universe full file is corrupted")

            SIGN3_SCRIPT_FR = [
                "sign1_full = cc.get_signature('sign1', 'full', data,**pars)",
                "sign3_full = cc.get_signature('sign3', 'full', data,**pars)",
                "sign2_src_list = [%s]" % (str(self.ref_datasets)[1:-1]),
                "sign2_list = [cc.get_signature('sign2', 'full', ds) for ds in sign2_src_list]",
                "sign3_full.fit(sign2_list,sign_full,sign1_full,sign2_universe='%s', sign2_coverage='%s')" % (
                    full_universe, full_coverage)
            ]
            SPECIFIC_SCRIPTS['sign3'] = (SIGN3_SCRIPT_FR, ["sign2_src_list = [%s]" %
                                                           (str(self.ref_datasets)[1:-1])] + SIGN3_SCRIPT_F)

        else:  # NS if not sign3

            if self.datasets is None:  # NS: this is our case in the update pipeline
                all_datasets = Dataset.get()  # sqlaclchemy objects representing A1.001, B1.001 etc

                # NS: Selecting the dataset codes to process
                for ds in all_datasets:
                    if not ds.essential:
                        continue

                    # returns a signx object
                    sign = cc.get_signature(
                        self.cctype, "full", ds.dataset_code)

                    if sign.is_fit():
                        continue

                    # NS D1 preprocess is very long, do not delete it if it has
                    # crashed
                    if not (sign.dataset == 'D1.001' and sign.cctype == 'sign0') and os.path.exists(sign.signature_path):
                        #print("Attempting to delete ", sign.signature_path)
                        # shutil.rmtree(sign.signature_path,ignore_errors=True)
                        #print("DELETED: ", sign.signature_path)
                        pass

                    # NS The fit is for full & reference branches
                    # (default:True)
                    if self.full_reference:
                        # NS molset: reference
                        sign = cc.get_signature(
                            self.cctype, "reference", ds.dataset_code)

                        if not (sign.dataset == 'D1.001' and sign.cctype == 'sign0') and os.path.exists(sign.signature_path):
                            #print("Attempting to delete signature path: ", sign.signature_path)
                            #shutil.rmtree(sign.signature_path, ignore_errors=True)
                            #print("DELETED: ", sign.signature_path)
                            pass

                    dataset_codes.append(ds.dataset_code)

            else:  # NS with custom dataset
                for ds in self.datasets:
                    sign = cc.get_signature(self.cctype, "full", ds)
                    if sign.is_fit():
                        continue
                    dataset_codes.append(ds)  # NS i.e: 'A1.001', 'B1.001' etc

        dataset_params = list()

        # NS: now filling dataset_params
        for ds_code in dataset_codes:
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

        job_path = None

        if self.cc_old_path is not None:
            SIGN0_SCRIPT_FR.insert(0, "CC_OLD_PATH = '%s'" % self.cc_old_path)

        else:
            SIGN0_SCRIPT_FR.insert(0, "CC_OLD_PATH = None")

        SPECIFIC_SCRIPTS['sign0'] = (SIGN0_SCRIPT_FR, SIGN0_SCRIPT_FR)

        if len(dataset_codes) > 0:

            # NS: going through sorted dataset_codes ['A1.001', 'B1.001' etc]
            dataset_codes.sort()

            job_path = tempfile.mkdtemp(
                prefix='jobs_' + self.cctype + '_', dir=self.tmpdir)

            if not os.path.isdir(job_path):  # create directory for running the job
                os.mkdir(job_path)

            # create script file that will launch signx fit jobs for ALL the
            # select dataset
            script_lines = [
                "import sys, os",
                "import pickle",
                "import logging",
                "import chemicalchecker",
                "from chemicalchecker.util import Config",
                "from chemicalchecker.core import ChemicalChecker",
                "logging.log(logging.DEBUG,'chemicalchecker: %s' % chemicalchecker.__path__)",
                "logging.log(logging.DEBUG,'CWD: %s' % os.getcwd())",
                "config = Config()",
                "task_id = sys.argv[1]",  # <TASK_ID>
                "filename = sys.argv[2]",  # <FILE>
                # load pickled data
                "inputs = pickle.load(open(filename, 'rb'))",
                "data = inputs[task_id][0][0]",  # elements for current job
                "pars = inputs[task_id][0][1]",  # elements for current job
                # elements are indexes
                "cc = ChemicalChecker('%s')" % self.cc_root,
                'if pars is None: pars = {}',
                # start import
                'sign_full = cc.get_signature("%s","full",data)' % DEPENDENCIES[
                    self.cctype][0],
                # start import
                'sign_ref = cc.get_signature("%s","reference",data)' % DEPENDENCIES[
                    self.cctype][0]
            ]

            # Preprocessing scripts
            if self.full_reference:  # True for sign0
                if self.cctype in SPECIFIC_SCRIPTS:
                    # i.e SIGN0_SCRIPT_FR, i.e will call cc.preprocess
                    script_lines += SPECIFIC_SCRIPTS[self.cctype][0]
                else:
                    script_lines += [sub.replace('<CC_TYPE>', self.cctype)
                                     for sub in CC_SCRIPT_FR]
            else:
                if self.cctype in SPECIFIC_SCRIPTS:
                    # # same for sign0 i.e SIGN0_SCRIPT_FR, i.e will call cc.preprocess
                    script_lines += SPECIFIC_SCRIPTS[self.cctype][1]
                else:
                    # here the fit method is called
                    script_lines += [sub.replace('<CC_TYPE>', self.cctype)
                                     for sub in CC_SCRIPT_F]

            script_lines += ["print('JOB DONE')"]

            script_name = os.path.join(job_path, self.cctype + '_script.py')

            # Python script to launch jobs on the cluster
            with open(script_name, 'w') as fh:
                for line in script_lines:
                    fh.write(line + '\n')
            # hpc parameters

            params = {}
            params["num_jobs"] = len(dataset_codes)
            params["jobdir"] = job_path
            params["job_name"] = "CC_" + self.cctype.upper()
            params["elements"] = dataset_params
            params["wait"] = True
            params["memory"] = MEM_CPU[self.cctype][0]
            params["cpu"] = MEM_CPU[self.cctype][1]

            if self.cctype == 'sign1':
                # h_vmem parameter NS some sign (memory limit)
                params["mem_by_core"] = 20
                # NS tmp, remove afterwards!!
                params['specificNode'] = 'pac-one301'

            # job command
            cc_config_path = os.environ['CC_CONFIG']
            cc_package = os.path.join(Config().PATH.CC_REPO, 'package')
            singularity_image = Config().PATH.SINGULARITY_IMAGE
            command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} singularity exec {} python {} <TASK_ID> <FILE>".format(
                cc_package, cc_config_path, singularity_image, script_name)
            # submit jobs
            cluster = HPC.from_config(Config())
            jobs = cluster.submitMultiJob(command, **params)

        # Calculating the signature
        dataset_not_done = []

        for code in dataset_codes:

            sign = cc.get_signature(self.cctype, "full", code)
            if sign.is_fit():
                continue

            dataset_not_done.append(code)
            self.__log.warning(
                self.cctype + " fit failed for dataset code: " + code)

        if len(dataset_not_done) == 0:
            self.mark_ready()
            if job_path is not None:
                print("Attempting to delete job path ", job_path)
                shutil.rmtree(job_path, ignore_errors=True)
        else:
            if not self.custom_ready():
                raise Exception("Not all dataset fits are done")

    def execute(self, context):
        """Same as run but for Airflow."""
        self.tmpdir = context['params']['tmpdir']
        self.run()
