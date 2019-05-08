import tempfile
import os

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.util import Config
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util import BaseStep
from chemicalchecker.util import HPC


@logged
class Signature1(BaseStep):

    def __init__(self, config, name, **params):

        BaseStep.__init__(self, config, name, **params)

    def run(self):
        """Run the molprops step."""

        all_datasets = Dataset.get()
        config = Config()

        cc = ChemicalChecker(config.PATH.CC_ROOT)
        dataset_codes = list()
        for ds in all_datasets:
            sign1 = cc.get_signature("sign1", "full", ds.dataset_code)
            if sign1.is_fit():
                continue

            dataset_codes.append(ds.dataset_code)

        job_path = tempfile.mkdtemp(
            prefix='jobs_sign1_', dir=self.tmpdir)

        dataset_params = list()

        ds_data_params = self.config.STEPS[self.name]

        for ds_code in dataset_codes:
            if ds_code in ds_data_params:
                dataset_params.append((ds_code, ds_data_params[ds_code]))
            else:
                dataset_params.append((ds_code, None))

        if not os.path.isdir(job_path):
            os.mkdir(job_path)
        # create script file
        cc_config = os.environ['CC_CONFIG']
        cc_package = os.path.join(config.PATH.CC_REPO, 'package')
        script_lines = [
            "import sys, os",
            "import pickle",
            "os.environ['CC_CONFIG'] = '%s'" % cc_config,  # cc_config location
            "sys.path.append('%s')" % cc_package,  # allow package import
            "from chemicalchecker.util import Config",
            "from chemicalchecker.core import ChemicalChecker",
            "config = Config()",
            "task_id = sys.argv[1]",  # <TASK_ID>
            "filename = sys.argv[2]",  # <FILE>
            "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
            "data = inputs[task_id][0][0]",  # elements for current job
            "topics = inputs[task_id][0][1]",  # elements for current job
            # elements are indexes
            'cc = ChemicalChecker(config.PATH.CC_ROOT )',
            # start import
            'sign0_full = cc.get_signature("sign0","full",data)',
            # start import
            'sign0_ref = cc.get_signature("sign0","reference",data)',
            'pars = {"num_topics": topics, "max_freq": 0.9}',
            # start import
            'sign1_ref = cc.get_signature("sign1", "reference", data,**pars)',
            "sign1_ref.fit(sign0_ref)",
            "sign1_full = cc.get_signature('sign1', 'full', data,**pars)",
            "sign1_ref.predict(sign0_full, destination=sign1_full.data_path)",
            "sign1_full.mark_ready()",
            "print('JOB DONE')"
        ]

        script_name = os.path.join(job_path, 'sign0_script.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters

        params = {}
        params["num_jobs"] = len(dataset_codes)
        params["jobdir"] = job_path
        params["job_name"] = "CC_SIGN1"
        params["elements"] = dataset_params
        params["wait"] = True
        params["memory"] = 20
        params["cpu"] = 10
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC(config)
        jobs = cluster.submitMultiJob(command, **params)

        dataset_not_done = []

        for code in dataset_codes:

            sign0 = cc.get_signature("sign1", "full", code)
            if sign0.is_fit():
                continue

            dataset_not_done.append(code)
            self.__log.warning(
                "Signature0 fit failed for dataset code: " + code)

        if len(dataset_not_done) == 0:
            self.mark_ready()
