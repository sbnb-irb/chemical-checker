import tempfile
import os
import shutil
import h5py
from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.util import Config
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util import BaseStep
from chemicalchecker.util import HPC


@logged
class SmilesSignature3(BaseStep):

    def __init__(self, config, name, **params):

        BaseStep.__init__(self, config, name, **params)

    def run(self):
        """Run the molprops step."""

        all_datasets = Dataset.get()
        config_cc = Config()

        cc = ChemicalChecker(config_cc.PATH.CC_ROOT)

        dataset_codes = list()
        for ds in all_datasets:
            if not ds.exemplary:
                continue
            sign3 = cc.get_signature("sign3", "full", ds.dataset_code)
            if os.path.exists(os.path.join(sign3.model_path, "adanet_sign0_A1.001_final/savedmodel/saved_model.pb")):
                continue

            # if os.path.exists(sign3.signature_path):
            #    shutil.rmtree(sign3.signature_path)

            # dataset_codes.append(ds.dataset_code)

        dataset_codes.append("ZZ.000")
        job_path = tempfile.mkdtemp(
            prefix='jobs_smiles_sign3_', dir=self.tmpdir)

        if not os.path.isdir(job_path):
            os.mkdir(job_path)

        dataset_codes.sort()

        if len(dataset_codes) > 0:

            # create script file
            cc_config_path = os.environ['CC_CONFIG']
            cc_package = os.path.join(config_cc.PATH.CC_REPO, 'package')
            script_lines = [
                "import sys, os",
                "import pickle",
                # cc_config location
                "os.environ['CC_CONFIG'] = '%s'" % cc_config_path,
                "sys.path.append('%s')" % cc_package,  # allow package import
                "from chemicalchecker.util import Config",
                "from chemicalchecker.core import ChemicalChecker",
                "config = Config()",
                "task_id = sys.argv[1]",  # <TASK_ID>
                "filename = sys.argv[2]",  # <FILE>
                # load pickled data
                "inputs = pickle.load(open(filename, 'rb'))",
                "data = str(inputs[task_id][0])",  # elements for current job
                # elements are indexes
                'cc = ChemicalChecker(config.PATH.CC_ROOT )',
                # start import
                "pars = {'cpu': 32}",
                # start import
                "sign3_full = cc.get_signature('sign3', 'full', data,**pars)",
                "sign0_full = cc.get_signature('sign0', 'full', 'A1.001')",
                "sign3_full.fit_sign0(sign0_full, include_confidence=False)",
                "print('JOB DONE')"
            ]

            script_name = os.path.join(job_path, 'smiles_sign3_script.py')
            with open(script_name, 'w') as fh:
                for line in script_lines:
                    fh.write(line + '\n')
            # hpc parameters

            dataset_codes.sort()

            params = {}
            params["num_jobs"] = len(dataset_codes)
            params["jobdir"] = job_path
            params["job_name"] = "CC_SMLS3"
            params["elements"] = dataset_codes
            params["wait"] = True
            params["memory"] = 50
            params["cpu"] = 32
            # job command
            singularity_image = Config().PATH.SINGULARITY_IMAGE
            command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
                singularity_image, script_name)
            # submit jobs
            cluster = HPC.from_config(config_cc)
            jobs = cluster.submitMultiJob(command, **params)

        dataset_not_done = []

        for code in dataset_codes:

            sign3 = cc.get_signature("sign3", "full", code)
            if os.path.exists(os.path.join(sign3.model_path, "adanet_sign0_A1.001_final/savedmodel/saved_model.pb")):
                continue

            dataset_not_done.append(code)
            self.__log.warning(
                "Smiles Signature3 fit failed for dataset code: " + code)

        if len(dataset_not_done) == 0:
            self.mark_ready()
            # shutil.rmtree(job_path)
