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

    def __init__(self, config=None, name='signature1', **params):

        BaseStep.__init__(self, config, name, **params)

        self.datasets = params.get('datasets', None)
        self.full_reference = params.get('full_reference', True)
        self.ds_data_params = params.get('ds_params', None)

    def run(self):
        """Run the molprops step."""

        config_cc = Config()
        dataset_codes = list()
        cc = ChemicalChecker(config_cc.PATH.CC_ROOT)

        if self.ds_data_params is None and self.config is not None:
            self.ds_data_params = self.config.STEPS[self.name]
        if self.ds_data_params is None:
            raise Exception("No parameters set for datasets")

        if self.datasets is None:
            all_datasets = Dataset.get()

            for ds in all_datasets:
                if not ds.essential:
                    continue
                self.ds_data_params[ds.dataset_code]['discrete'] = ds.discrete
                sign1 = cc.get_signature("sign1", "full", ds.dataset_code)
                if sign1.is_fit():
                    continue
                dataset_codes.append(ds.dataset_code)
        else:
            for ds in self.datasets:
                sign1 = cc.get_signature("sign1", "full", ds)
                if sign1.is_fit():
                    continue
                dataset_codes.append(ds)

        job_path = tempfile.mkdtemp(
            prefix='jobs_sign1_', dir=self.tmpdir)

        dataset_params = list()

        for ds_code in dataset_codes:
            if isinstance(self.ds_data_params, Config):
                temp_dict = self.ds_data_params.asdict()
            else:
                temp_dict = self.ds_data_params
            if ds_code in temp_dict.keys():
                if isinstance(temp_dict[ds_code], Config):
                    dict_params = temp_dict[ds_code].asdict()
                else:
                    dict_params = temp_dict[ds_code]
                dataset_params.append(
                    (ds_code, dict_params))
            else:
                dataset_params.append((ds_code, None))

        job_path = None
        if len(dataset_codes) > 0:

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
                "from chemicalchecker.database import Dataset",
                "config = Config()",
                "task_id = sys.argv[1]",  # <TASK_ID>
                "filename = sys.argv[2]",  # <FILE>
                "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
                "data = inputs[task_id][0][0]",  # elements for current job
                "pars = inputs[task_id][0][1]",  # elements for current job
                # elements are indexes
                'cc = ChemicalChecker(config.PATH.CC_ROOT )',
                # start import
                'sign0_full = cc.get_signature("sign0","full",data)',
                # start import
                'sign0_ref = cc.get_signature("sign0","reference",data)',
                'if pars is None: pars = {}']

            if self.full_reference:
                script_lines += ['sign1_ref = cc.get_signature("sign1", "reference", data,**pars)',
                                 "sign1_ref.fit(sign0_ref)",
                                 "sign1_full = cc.get_signature('sign1', 'full', data,**pars)",
                                 "sign1_ref.predict(sign0_full, destination=sign1_full.data_path)",
                                 "sign1_full.validate()",
                                 "sign1_full.mark_ready()",
                                 "print('JOB DONE')"
                                 ]
            else:
                script_lines += ['sign1_full = cc.get_signature("sign1", "full", data,**pars)',
                                 "sign1_full.fit(sign0_full)",
                                 "print('JOB DONE')"
                                 ]

            script_name = os.path.join(job_path, 'sign1_script.py')
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
            cluster = HPC.from_config(config_cc)
            jobs = cluster.submitMultiJob(command, **params)

        dataset_not_done = []

        for code in dataset_codes:

            sign0 = cc.get_signature("sign1", "full", code)
            if sign0.is_fit():
                continue

            dataset_not_done.append(code)
            self.__log.warning(
                "Signature1 fit failed for dataset code: " + code)

        if len(dataset_not_done) == 0:
            self.mark_ready()
