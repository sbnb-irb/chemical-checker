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
class Signature3(BaseStep):

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
            sign3 = cc.get_signature("sign3", "full", ds)
            if sign3.is_fit():
                continue

            if os.path.exists(sign3.signature_path):
                shutil.rmtree(sign3.signature_path)

            dataset_codes.append(ds)

        job_path = tempfile.mkdtemp(
            prefix='jobs_sign3_', dir=self.tmpdir)

        if not os.path.isdir(job_path):
            os.mkdir(job_path)

        dataset_codes.sort()

        if len(dataset_codes) > 0:

            sign2_list = [cc.get_signature('sign2', 'full', ds)
                          for ds in dataset_codes]
            full_universe = os.path.join(self.tmpdir, "universe_full")
            sign3_full = cc.get_signature('sign3', 'full', "A1.001")
            sign3_full.save_sign2_universe(sign2_list, full_universe)

            try:
                with h5py.File(full_universe, 'r') as hf:
                    keys = hf.keys()
            except Exception, e:

                self.__log.error(e)
                raise Exception("Universe full file is corrupted")

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
                "sign2_full = cc.get_signature('sign2', 'full', data)",
                "sign2_list = [cc.get_signature('sign2', 'full', ds) for ds in cc.datasets_exemplary()]",
                "sign3_full.fit(sign2_list,sign2_full,sign2_universe='%s')" % full_universe,
                "print('JOB DONE')"
            ]

            script_name = os.path.join(job_path, 'sign3_script.py')
            with open(script_name, 'w') as fh:
                for line in script_lines:
                    fh.write(line + '\n')
            # hpc parameters

            dataset_codes.sort()

            params = {}
            params["num_jobs"] = len(dataset_codes)
            params["jobdir"] = job_path
            params["job_name"] = "CC_SIGN3"
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
            if sign3.is_fit():
                continue

            dataset_not_done.append(code)
            self.__log.warning(
                "Signature3 fit failed for dataset code: " + code)

        if len(dataset_not_done) == 0:
            self.mark_ready()
            # shutil.rmtree(job_path)
