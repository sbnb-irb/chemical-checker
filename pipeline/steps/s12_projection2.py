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
class Projection2(BaseStep):

    def __init__(self, config, name, **params):

        BaseStep.__init__(self, config, name, **params)

    def run(self):
        """Run the molprops step."""

        all_datasets = Dataset.get()
        config_cc = Config()

        cc = ChemicalChecker(config_cc.PATH.CC_ROOT)
        dataset_codes = list()
        for ds in all_datasets:
            if not ds.essential:
                continue
            sign2 = cc.get_signature("proj2", "full", ds.dataset_code)
            if sign2.is_fit():
                continue

            dataset_codes.append(ds.dataset_code)

        job_path = tempfile.mkdtemp(
            prefix='jobs_proj2_', dir=self.tmpdir)

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
            'sign2_full = cc.get_signature("sign2","full",data)',
            # start import
            'sign2_ref = cc.get_signature("sign2","reference",data)',
            "pars = {'cpu': 10}",
            # start import
            'proj2_ref = cc.get_signature("proj2", "reference", data,**pars)',
            "proj2_ref.fit(sign2_ref)",
            "proj2_full = cc.get_signature('proj2', 'full', data,**pars)",
            "proj2_ref.predict(sign2_full, destination=proj2_full.data_path)",
            "proj2_full.mark_ready()",
            "print('JOB DONE')"
        ]

        script_name = os.path.join(job_path, 'proj2_script.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters

        params = {}
        params["num_jobs"] = len(dataset_codes)
        params["jobdir"] = job_path
        params["job_name"] = "CC_PROJ2"
        params["elements"] = dataset_codes
        params["wait"] = True
        params["memory"] = 20
        params["cpu"] = 10
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC.from_config(config_cc)
        jobs = cluster.submitMultiJob(command, **params)

        dataset_not_done = []

        for code in dataset_codes:

            sign0 = cc.get_signature("proj2", "full", code)
            if sign0.is_fit():
                continue

            dataset_not_done.append(code)
            self.__log.warning(
                "Projection2 fit failed for dataset code: " + code)

        if len(dataset_not_done) == 0:
            self.mark_ready()
            shutil.rmtree(job_path)
