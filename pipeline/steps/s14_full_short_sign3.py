import tempfile
import os
import shutil
import h5py
import numpy as np
from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.util import Config
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util import BaseStep
from chemicalchecker.util import HPC


@logged
class Sign3FullShort(BaseStep):

    def __init__(self, config, name, **params):

        BaseStep.__init__(self, config, name, **params)

    def run(self):
        """Run the molprops step."""

        all_datasets = Dataset.get()
        config_cc = Config()

        dataset_sign3_full = "ZZ.001"

        dataset_sign3_short = "ZZ.000"

        cc = ChemicalChecker(config_cc.PATH.CC_ROOT)
        dataset_codes = list()
        for ds in all_datasets:
            if not ds.exemplary:
                continue

            dataset_codes.append(ds)

        s3 = cc.get_signature('sign3', 'full', dataset_codes[0])

        rows = s3.shape[0]

        cols = s3.shape[1] * len(dataset_codes)

        dataset_codes.sort()

        s3_full = cc.get_signature('sign3', 'full', dataset_sign3_full)

        if not s3_full.is_fit():

            with h5py.File(s3_full.data_path, "w") as hf:
                hf.create_dataset("keys", data=np.array(
                    [str(k) for k in s3.keys]))
                hf.create_dataset("V", (rows, cols))

            for i, code in enumerate(dataset_codes):
                s3 = cc.get_signature('sign3', 'full', dataset_codes[i])
                with h5py.File(s3_full.data_path, "r+") as hf, h5py.File(s3.data_path, 'r') as dh5in:
                    dataset = dh5in["V"][:]
                    hf["V"][:, i * s3.shape[1]:(i + 1) * s3.shape[1]] = dataset

            s3_full.validate()
            s3_full.mark_ready()

        job_path = tempfile.mkdtemp(
            prefix='jobs_sign3short_', dir=self.tmpdir)

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
            "from numpy import linalg as LA",
            # cc_config location
            "os.environ['CC_CONFIG'] = '%s'" % cc_config_path,
            "sys.path.append('%s')" % cc_package,  # allow package import
            "from chemicalchecker.util import Config",
            "from chemicalchecker.core import ChemicalChecker",
            "from chemicalchecker.tool.autoencoder import AutoEncoder",
            "config = Config()",
            # elements are indexes
            'cc = ChemicalChecker(config.PATH.CC_ROOT )',
            # start import
            'sign3_full = cc.get_signature("sign3", "full", "%s")' % dataset_sign3_full,
            'sign3_short = cc.get_signature("sign3", "full", "%s")' % dataset_sign3_short,
            # start import
            'output_file = os.path.join(sign3_full.signature_path, "sign3_norm.h5")',
            'with h5py.File(output_file, "w") as hf:',
            '    hf.create_dataset("keys", data=np.array([str(k) for k in sign3_full.keys]))',
            '    hf.create_dataset("x", sign3_full.shape, dtype=np.float32)',
            'num_datasets = %d' % len(dataset_codes),
            'for i in range(0, num_datasets):',
            "    with h5py.File(output_file, 'r+') as hf, h5py.File(sign3_full.data_path, 'r') as dh5in:",
            '        dataset = dh5in["V"][:, i * 128:(i+1)*128]',
            '        norms = LA.norm(dataset, axis=1)',
            '        hf["x"][:, i * 128:(i+1)*128] = dataset / norms[:, None]',
            'params = {"cpu": 32, "epochs": 800, "encoding_dim": 512}',
            'ae = AutoEncoder(sign3_short.model_path, **params)',
            'ae.fit(output_file)',
            'ae.encode(output_file,sign3_short.data_path, input_dataset="x")',
            "sign3_short.validate()",
            "sign3_short.mark_ready()",
            "print('JOB DONE')"
        ]

        script_name = os.path.join(job_path, 'sign3_short.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters

        params = {}
        params["jobdir"] = job_path
        params["job_name"] = "CC_SHORT3"
        params["wait"] = True
        params["memory"] = 30
        params["cpu"] = 32
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} ".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC.from_config(config_cc)
        jobs = cluster.submitMultiJob(command, **params)

        sign3_short = cc.get_signature("sign3", "full", dataset_sign3_short)

        if sign3_short.is_fit():
            self.mark_ready()
            shutil.rmtree(job_path)
        else:
            self.__log.warning(
                "Sign3 Full to Short failed please check")
