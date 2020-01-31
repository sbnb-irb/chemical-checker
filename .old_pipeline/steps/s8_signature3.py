import tempfile
import os
import shutil
import h5py
from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.util import Config
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util import BaseTask
from chemicalchecker.util import HPC


@logged
class Signature3(BaseTask):

    def __init__(self, config=None, name='signature3', **params):

        BaseTask.__init__(self, config, name, **params)

        self.target_datasets = params.get('target_datasets', None)
        self.ref_datasets = params.get('reference_datasets', None)

    def run(self):
        """Run the molprops step."""

        config_cc = Config()
        dataset_codes = list()
        cc = ChemicalChecker(config_cc.PATH.CC_ROOT)

        if self.ref_datasets is None:

            self.ref_datasets = []
            all_datasets = Dataset.get()
            for ds in all_datasets:
                if not ds.exemplary:
                    continue

                self.ref_datasets.append(ds.dataset_code)

        if self.target_datasets is None:

            for ds in self.ref_datasets:
                sign3 = cc.get_signature("sign3", "full", ds)
                if sign3.is_fit():
                    continue
                dataset_codes.append(ds)
        else:
            for ds in self.target_datasets:
                sign3 = cc.get_signature("sign3", "full", ds)
                if sign3.is_fit():
                    continue
                dataset_codes.append(ds)

        job_path = None

        if len(dataset_codes) > 0:

            job_path = tempfile.mkdtemp(
                prefix='jobs_sign3_', dir=self.tmpdir)

            if not os.path.isdir(job_path):
                os.mkdir(job_path)

            dataset_codes.sort()

            sign2_list = [cc.get_signature('sign2', 'full', ds)
                          for ds in self.ref_datasets]

            if self.target_datasets is None:
                full_universe = os.path.join(self.tmpdir, "universe_full")
                full_coverage = os.path.join(self.tmpdir, "coverage_full")
                sign3_full = cc.get_signature('sign3', 'full', "A1.001")
                sign3_full.save_sign2_universe(sign2_list, full_universe)
                sign3_full.save_sign2_coverage(sign2_list, full_coverage)
                try:
                    with h5py.File(full_universe, 'r') as hf:
                        keys = hf.keys()
                except Exception as e:

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
                "sign2_src_list = [%s]" % (str(self.ref_datasets)[1:-1]),
                "sign2_list = [cc.get_signature('sign2', 'full', ds) for ds in sign2_src_list]"
            ]

            if self.target_datasets is None:

                script_lines += [
                    "sign3_full.fit(sign2_list,sign2_full,sign2_universe='%s', sign2_coverage='%s')" % (
                        full_universe, full_coverage),
                    "print('JOB DONE')"
                ]

            else:

                script_lines += [
                    "sign2_list.append(cc.get_signature('sign2', 'full', data))",
                    "sign3_full.fit(sign2_list,sign2_full)",
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
            if job_path is not None:
                shutil.rmtree(job_path)
