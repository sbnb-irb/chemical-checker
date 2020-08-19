"""CCSmileConverter task.

This tasks is mainly thought to be used on signature3 to create a model to
predict signatures3 from smiles.
Maybe, it could be generalized to other signatures.
"""
import tempfile
import os
import shutil
import h5py
import numpy as np
from airflow.models import BaseOperator
from airflow import AirflowException
from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.util import Config
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util.pipeline import BaseTask
from chemicalchecker.util import HPC


@logged
class CCSmileConverter(BaseTask, BaseOperator):

    def __init__(self, name=None, cc_type='sign3', **params):
        """Initialize CC SmileConverter task.

        Args:
            name (str): The name of the task (default:None)
            cc_type (str): The CC type where the fit is applied (Required,
                default: sign3)
            CC_ROOT (str): The CC root path (Required)
            datasets (list): The list of dataset codes to create the smile
                converter model(Optional, all datasets taken by default)
        """
        if cc_type is None:
            raise Exception("CCSmileConverter requires a cc_type")

        if name is None:
            name = "smile_to _" + cc_type
        args = []
        task_id = params.get('task_id', None)
        if task_id is None:
            params['task_id'] = name

        BaseTask.__init__(self, name, **params)
        BaseOperator.__init__(self, *args, **params)

        self.cc_type = cc_type

        self.datasets = params.get('datasets', None)

        self.CC_ROOT = params.get('CC_ROOT', None)
        if self.CC_ROOT is None:
            raise Exception('CC_ROOT parameter is not set')

    def run(self):
        """Run the task."""
        config_cc = Config()
        dataset_codes = list()
        cc = ChemicalChecker(self.CC_ROOT)

        if self.datasets is None:
            all_datasets = Dataset.get()

            for ds in all_datasets:
                if not ds.exemplary:
                    continue
                sign = cc.get_signature(self.cc_type, "full", ds.dataset_code)
                if os.path.exists(
                    os.path.join(
                        sign.model_path,
                        "adanet_sign0_A1.001_final/savedmodel/saved_model.pb")):
                    continue

                dataset_codes.append(ds.dataset_code)

            sign = cc.get_signature(self.cc_type, "full", "ZZ.000")
            if sign.available() and not os.path.exists(
                os.path.join(
                    sign.model_path,
                    "adanet_sign0_A1.001_final/savedmodel/saved_model.pb")):
                dataset_codes.append("ZZ.000")

        else:
            for ds in self.datasets:
                sign = cc.get_signature(self.cc_type, "full", ds)
                if os.path.exists(
                    os.path.join(
                        sign.model_path,
                        "adanet_sign0_A1.001_final/savedmodel/saved_model.pb")):
                    continue
                dataset_codes.append(ds)

        dataset_codes.sort()

        dataset_params = {"cpu": 32}

        job_path = None
        if len(dataset_codes) > 0:

            job_path = tempfile.mkdtemp(
                prefix='jobs_smiles_' + self.cc_type + '_', dir=self.tmpdir)

            if not os.path.isdir(job_path):
                os.mkdir(job_path)

            # create script file
            cc_config_path = os.environ['CC_CONFIG']
            cc_package = os.path.join(config_cc.PATH.CC_REPO, 'package')
            script_lines = [
                "import sys, os",
                "import pickle",
                "from chemicalchecker.util import Config",
                "from chemicalchecker.core import ChemicalChecker",
                "config = Config()",
                "task_id = sys.argv[1]",  # <TASK_ID>
                "filename = sys.argv[2]",  # <FILE>
                # load pickled data
                "inputs = pickle.load(open(filename, 'rb'))",
                "data = str(inputs[task_id][0])",  # elements for current job
                # elements are indexes
                "cc = ChemicalChecker('%s' )" % self.CC_ROOT,
                'pars = %s' % dataset_params,
                "sign_full = cc.get_signature('%s', 'full', data,**pars)" % self.cc_type,
                "sign0_full = cc.get_signature('sign0', 'full', 'A1.001')",
                "sign_full.fit_sign0(sign0_full, include_confidence=False)",
                "print('JOB DONE')"

            ]

            script_name = os.path.join(
                job_path, self.cc_type + '_short_script.py')
            with open(script_name, 'w') as fh:
                for line in script_lines:
                    fh.write(line + '\n')
            # hpc parameters

            params = {}
            params["num_jobs"] = len(dataset_codes)
            params["jobdir"] = job_path
            params["job_name"] = "CC_SML_" + self.cc_type.upper()
            params["elements"] = dataset_codes
            params["wait"] = True
            params["cpu"] = 32
            # job command
            singularity_image = Config().PATH.SINGULARITY_IMAGE
            command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} singularity exec {} python {} <TASK_ID> <FILE>".format(
                cc_package, cc_config_path, singularity_image, script_name)
            # submit jobs
            cluster = HPC.from_config(config_cc)
            jobs = cluster.submitMultiJob(command, **params)

        dataset_not_done = []

        for code in dataset_codes:

            sign = cc.get_signature(self.cc_type, "full", code)
            if os.path.exists(
                os.path.join(
                    sign.model_path,
                    "adanet_sign0_A1.001_final/savedmodel/saved_model.pb")):
                continue

            dataset_not_done.append(code)
            self.__log.warning(
                "Smiles " + self.cc_type +
                " fit failed for dataset code: " + code)

        if len(dataset_not_done) == 0:
            self.mark_ready()
            if os.path.isdir(job_path):
                shutil.rmtree(job_path)
        else:
            if not self.custom_ready():
                raise AirflowException("Some predictions failed")

    def execute(self, context):
        """Same as run but for Airflow."""
        self.tmpdir = context['params']['tmpdir']
        self.run()
