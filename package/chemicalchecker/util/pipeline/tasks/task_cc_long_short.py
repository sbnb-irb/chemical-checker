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
from chemicalchecker.core.signature_data import DataSignature
from chemicalchecker.util.pipeline import BaseTask
from chemicalchecker.util import HPC


@logged
class CCLongShort(BaseTask, BaseOperator):

    def __init__(self, config=None, name=None, cc_type=None, **params):

        if cc_type is None:
            raise Exception("CCLongShort requires a cc_type")

        if name is None:
            name = cc_type + "_long_short"
        args = []
        task_id = params.get('task_id', None)
        if task_id is None:
            params['task_id'] = name

        BaseTask.__init__(self, config, name, **params)
        BaseOperator.__init__(self, *args, **params)

        self.cc_type = cc_type

        self.datasets = params.get('datasets', None)
        self.dataset_sign_long = params.get('dataset_sign_long', 'ZZ.001')
        self.dataset_sign_short = params.get('dataset_sign_short', 'ZZ.000')
        self.epochs = params.get('epochs', 800)
        self.encoding_dim = params.get('encoding_dim', 512)
        self.CC_ROOT = params.get('CC_ROOT', None)
        if self.CC_ROOT is None:
            raise Exception('CC_ROOT parameter is not set')

    def run(self):
        """Run the CCLongShort task."""

        config_cc = Config()
        dataset_codes = list()
        cc = ChemicalChecker(self.CC_ROOT)

        if self.datasets is None:
            all_datasets = Dataset.get()

            for ds in all_datasets:
                if not ds.exemplary:
                    continue
                sign = cc.get_signature(
                    self.cc_type, "full", ds.dataset_code)
                if not sign.available():
                    raise Exception(
                        self.cc_type + " not available for dataset " + ds.dataset_code)

                dataset_codes.append(ds.dataset_code)

        else:
            for ds in self.datasets:
                sign = cc.get_signature(self.cc_type, "full", ds)
                if not sign.available():
                    raise Exception(
                        self.cc_type + " not available for dataset " + ds)
                dataset_codes.append(ds)

        dataset_codes.sort()

        dataset_params = {"cpu": 32, "epochs": self.epochs,
                          "encoding_dim": self.encoding_dim,
                          "input_dataset": "V"}

        s = cc.get_signature(self.cc_type, 'full', dataset_codes[0])

        rows = s.shape[0]

        input_dim = s.shape[1]

        cols = input_dim * len(dataset_codes)

        s_long = cc.get_signature(self.cc_type, 'full', self.dataset_sign_long)

        if not s_long.is_fit():

            with h5py.File(s_long.data_path, "w") as hf:
                hf.create_dataset("keys", data=np.array(
                    [str(k) for k in s.keys], DataSignature.string_dtype()))
                hf.create_dataset("V", (rows, cols))

            for i, code in enumerate(dataset_codes):
                sing = cc.get_signature(self.cc_type, 'full', dataset_codes[i])
                with h5py.File(s_long.data_path, "r+") as hf, h5py.File(sing.data_path, 'r') as dh5in:
                    dataset = dh5in["V"][:]
                    hf["V"][:, i * input_dim:(i + 1) * input_dim] = dataset

            s_long.validate()
            s_long.mark_ready()

        job_path = None
        if len(dataset_codes) > 0:

            job_path = tempfile.mkdtemp(
                prefix='jobs_long_short_' + self.cc_type + '_', dir=self.tmpdir)

            if not os.path.isdir(job_path):
                os.mkdir(job_path)

            # create script file
            cc_config_path = os.environ['CC_CONFIG']
            cc_package = os.path.join(config_cc.PATH.CC_REPO, 'package')
            script_lines = [
                "import sys, os",
                "import pickle",
                "import h5py",
                "import numpy as np",
                "from chemicalchecker.util import Config",
                "from chemicalchecker.core import ChemicalChecker",
                "from chemicalchecker.tool.autoencoder import AutoEncoderSiamese",
                "config = Config()",
                # elements are indexes
                "cc = ChemicalChecker('%s')" % self.CC_ROOT,
                # start import
                'sign_full = cc.get_signature("%s", "full", "%s")' % (
                    self.cc_type, self.dataset_sign_long),
                'sign_short = cc.get_signature("%s", "full", "%s")' % (
                    self.cc_type, self.dataset_sign_short),
                # start import
                'params = %s' % dataset_params,
                'ae = AutoEncoderSiamese(sign_short.model_path, **params)',
                'ae.fit(sign_full.data_path)',
                'ae.encode(sign_full.data_path,sign_short.data_path, input_dataset="V")',
                "sign_short.validate()",
                "sign_short.mark_ready()",
                "print('JOB DONE')"
            ]

            script_name = os.path.join(
                job_path, self.cc_type + '_short_script.py')
            with open(script_name, 'w') as fh:
                for line in script_lines:
                    fh.write(line + '\n')
            # hpc parameters

            params = {}
            params["jobdir"] = job_path
            params["job_name"] = "CC_SH_" + self.cc_type.upper()
            params["wait"] = True
            params["memory"] = 2
            params["cpu"] = 32
            # job command
            singularity_image = Config().PATH.SINGULARITY_IMAGE
            command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} singularity exec {} python {} ".format(
                cc_package, cc_config_path, singularity_image, script_name)
            # submit jobs
            cluster = HPC.from_config(config_cc)
            jobs = cluster.submitMultiJob(command, **params)

        sign_short = cc.get_signature(
            self.cc_type, "full", self.dataset_sign_short)

        if sign_short.is_fit():
            self.mark_ready()
            if os.path.isdir(job_path):
                shutil.rmtree(job_path)
        else:
            self.__log.warning(
                self.dataset_sign_short + " Long to Short failed please check")
            if not self.custom_ready():
                raise AirflowException("Long to Short failed")

    def execute(self, context):
        """Run the molprops step."""

        self.tmpdir = context['params']['tmpdir']

        self.run()
