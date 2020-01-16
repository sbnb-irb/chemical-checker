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
class Neigh1(BaseStep):

    def __init__(self, config=None, name='neigh1', **params):

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
                neig1 = cc.get_signature("neig1", "full", ds.dataset_code)
                if neig1.is_fit():
                    continue
                dataset_codes.append(ds.dataset_code)
        else:
            for ds in self.datasets:
                neig1 = cc.get_signature("neig1", "full", ds)
                if neig1.is_fit():
                    continue
                dataset_codes.append(ds)

        job_path = None
        if len(dataset_codes) > 0:
            job_path = tempfile.mkdtemp(
                prefix='jobs_neig1_', dir=self.tmpdir)

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
                "pars = {'cpu': 15}"
            ]

            if self.full_reference:
                script_lines += ['neig1_ref = cc.get_signature("neig1", "reference", data,**pars)',
                                 "neig1_ref.fit(sign1_ref)",
                                 "neig1_full = cc.get_signature('neig1', 'full', data,**pars)",
                                 "neig1_ref.predict(sign1_full, destination=neig1_full.data_path)",
                                 "neig1_full.mark_ready()",
                                 "print('JOB DONE')"
                                 ]
            else:
                script_lines += ['neig1_full = cc.get_signature("neig1", "full", data,**pars)',
                                 "neig1_full.fit(sign1_full)",
                                 "print('JOB DONE')"
                                 ]

            script_name = os.path.join(job_path, 'neig1_script.py')
            with open(script_name, 'w') as fh:
                for line in script_lines:
                    fh.write(line + '\n')
            # hpc parameters

            params = {}
            params["num_jobs"] = len(dataset_codes)
            params["jobdir"] = job_path
            params["job_name"] = "CC_NEIG1"
            params["elements"] = dataset_codes
            params["wait"] = True
            params["memory"] = 30
            params["cpu"] = 15
            # job command
            singularity_image = Config().PATH.SINGULARITY_IMAGE
            command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
                singularity_image, script_name)
            # submit jobs
            cluster = HPC.from_config(config_cc)
            jobs = cluster.submitMultiJob(command, **params)

        dataset_not_done = []

        for code in dataset_codes:

            neig1 = cc.get_signature("neig1", "full", code)
            if neig1.is_fit():
                continue

            dataset_not_done.append(code)
            self.__log.warning(
                "Neigh1 fit failed for dataset code: " + code)

        if len(dataset_not_done) == 0:
            self.mark_ready()
            if job_path is not None:
                shutil.rmtree(job_path)
