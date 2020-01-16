import tempfile
import os
import shutil

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.util import Config
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util import BaseStep
from chemicalchecker.util import HPC


@logged
class Signature2(BaseStep):

    def __init__(self, config=None, name='signature2', **params):

        BaseStep.__init__(self, config, name, **params)

        self.datasets = params.get('datasets', None)
        self.full_reference = params.get('full_reference', True)

    def run(self):
        """Run the molprops step."""

        config_cc = Config()
        dataset_codes = list()
        cc = ChemicalChecker(config_cc.PATH.CC_ROOT)

        if self.datasets is None:
            all_datasets = Dataset.get()

            for ds in all_datasets:
                if not ds.essential:
                    continue
                sign2 = cc.get_signature("sign2", "full", ds.dataset_code)
                if sign2.is_fit():
                    continue

                if os.path.exists(sign2.signature_path):
                    shutil.rmtree(sign2.signature_path)

                if self.full_reference:
                    sign2 = cc.get_signature(
                        "sign2", "reference", ds.dataset_code)
                    if os.path.exists(sign2.signature_path):
                        shutil.rmtree(sign2.signature_path)

                dataset_codes.append(ds.dataset_code)

        else:
            for ds in self.datasets:
                sign1 = cc.get_signature("sign2", "full", ds)
                if sign1.is_fit():
                    continue
                dataset_codes.append(ds)

        job_path = None
        if len(dataset_codes) > 0:

            job_path = tempfile.mkdtemp(
                prefix='jobs_sign2_', dir=self.tmpdir)

            if not os.path.isdir(job_path):
                os.mkdir(job_path)
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
                "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
                "data = str(inputs[task_id][0])",  # elements for current job
                # elements are indexes
                'cc = ChemicalChecker(config.PATH.CC_ROOT )',
                # start import
                'sign1_full = cc.get_signature("sign1","full",data)',
                # start import
                'sign1_ref = cc.get_signature("sign1","reference",data)',
                "pars = {'adanet':{'cpu': 16},'node2vec': {'cpu': 4}}"
            ]

            if self.full_reference:
                script_lines += [
                    'neig1_ref = cc.get_signature("neig1", "reference", data)',
                    'sign2_ref = cc.get_signature("sign2", "reference", data,**pars)',
                    "sign2_ref.fit(sign1_ref, neig1_ref, reuse=False)",
                    "sign2_full = cc.get_signature('sign2', 'full', data,**pars)",
                    "sign2_ref.predict(sign1_full, destination=sign2_full.data_path)",
                    "sign2_full.validate()",
                    "sign2_full.mark_ready()",
                    "print('JOB DONE')"
                ]
            else:
                script_lines += [
                    'neig1_full = cc.get_signature("neig1", "full", data)',
                    'sign2_full = cc.get_signature("sign2", "full", data,**pars)',
                    "sign2_full.fit(sign1_full, neig1_full, reuse=False)",
                    "print('JOB DONE')"
                ]

            script_name = os.path.join(job_path, 'sign2_script.py')
            with open(script_name, 'w') as fh:
                for line in script_lines:
                    fh.write(line + '\n')
            # hpc parameters

            params = {}
            params["num_jobs"] = len(dataset_codes)
            params["jobdir"] = job_path
            params["job_name"] = "CC_SIGN2"
            params["elements"] = dataset_codes
            params["wait"] = True
            params["memory"] = 20
            params["cpu"] = 16
            # job command
            singularity_image = Config().PATH.SINGULARITY_IMAGE
            command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
                singularity_image, script_name)
            # submit jobs
            cluster = HPC.from_config(config_cc)
            jobs = cluster.submitMultiJob(command, **params)

        dataset_not_done = []

        for code in dataset_codes:

            sign2 = cc.get_signature("sign2", "full", code)
            if sign2.is_fit():
                continue

            dataset_not_done.append(code)
            self.__log.warning(
                "Signature2 fit failed for dataset code: " + code)

        if len(dataset_not_done) == 0:
            self.mark_ready()
            shutil.rmtree(job_path)
