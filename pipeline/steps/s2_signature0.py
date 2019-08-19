import tempfile
import os

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.util import Config
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util import BaseStep
from chemicalchecker.util import HPC


@logged
class Signature0(BaseStep):

    def __init__(self, config, name, **params):

        BaseStep.__init__(self, config, name, **params)

    def run(self):
        """Run the molprops step."""

        all_datasets = Dataset.get()
        config = Config()

        cc = ChemicalChecker(config.PATH.CC_ROOT)
        dataset_codes = list()
        for ds in all_datasets:
            if not ds.essential:
                continue
            sign0 = cc.get_signature("sign0", "reference", ds.dataset_code)
            if sign0.is_fit():
                continue

            dataset_codes.append(ds.dataset_code)

        job_path = tempfile.mkdtemp(
            prefix='jobs_sign0_', dir=self.tmpdir)

        if not os.path.isdir(job_path):
            os.mkdir(job_path)
        # create script file
        cc_config = os.environ['CC_CONFIG']
        cc_package = os.path.join(config.PATH.CC_REPO, 'package')
        script_lines = [
            "import sys, os",
            "import pickle, h5py",
            "os.environ['CC_CONFIG'] = '%s'" % cc_config,  # cc_config location
            "sys.path.append('%s')" % cc_package,  # allow package import
            "from chemicalchecker.util.remove_near_duplicates import RNDuplicates",
            "from chemicalchecker.util import Config",
            "from chemicalchecker.core import ChemicalChecker",
            "task_id = sys.argv[1]",  # <TASK_ID>
            "filename = sys.argv[2]",  # <FILE>
            "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
            "dataset = str(inputs[task_id][0])",  # elements for current job
            "config = Config()",
            'cc = ChemicalChecker(config.PATH.CC_ROOT )',
            # start import
            'sign0_full = cc.get_signature("sign0","full",dataset)',
            "sign0_full.fit()",
            "sign0_ref = cc.get_signature('sign0', 'reference', dataset)",
            "rnd = RNDuplicates(cpu=10)",
            "rnd.remove(sign0_full.data_path, save_dest=sign0_ref.data_path)",
            "f5 = h5py.File(sign0_full.data_path)",
            "features = f5['features'][:]",
            "f5.close()",
            "with h5py.File(sign0_ref.data_path, 'a') as hf:",
            "    hf.create_dataset('features', data=features)",
            "sign0_ref.mark_ready()",
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
        params["job_name"] = "CC_SIGN0"
        params["elements"] = dataset_codes
        params["wait"] = True
        # config.HPC.queue = "all.q@pac-one301"
        params["memory"] = 40
        params["cpu"] = 10
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC.from_config(config)
        jobs = cluster.submitMultiJob(command, **params)

        dataset_not_done = []

        for code in dataset_codes:

            sign0 = cc.get_signature("sign0", "reference", code)
            if sign0.is_fit():
                continue

            dataset_not_done.append(code)
            self.__log.warning(
                "Signature0 fit failed for dataset code: " + code)

        if len(dataset_not_done) == 0:
            self.mark_ready()
