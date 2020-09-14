"""CCPredict task.

This class allows to pipe different ``predict`` tasks in the Pipeline
framework.
It tries to work for all possible CC elements but considering that signatures
are changing quite often it might need to be updated.
"""
import os
import time
import shutil
import tempfile

from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util.pipeline import BaseTask
from chemicalchecker.util import logged, Config, HPC


VALID_TYPES = ['sign', 'neig', 'clus', 'proj']

CC_TYPES_DEPENDENCIES = {'sign0': ['sign0'], 'sign1': ['sign0'],
                         'sign2': ['sign1'], 'sign3': ['sign2'],
                         'neig1': ['sign1'], 'neig2': ['sign2'], 'neig3': ['sign3'],
                         'clus1': ['sign1'], 'clus2': ['sign2'], 'clus3': ['sign3'],
                         'proj1': ['sign1'], 'proj2': ['sign2'], 'proj3': ['sign3']}

CC_TYPES_MEM_CPU = {'sign0': (44, 22), 'sign1': (20, 10), 'sign2': (20, 16), 'sign3': (2, 32),
                    'neig1': (30, 15), 'neig2': (30, 15), 'neig3': (30, 15),
                    'clus1': (20, 10), 'clus2': (20, 10), 'clus3': (20, 10),
                    'proj1': (20, 10), 'proj2': (20, 10), 'proj3': (20, 10)}

SPECIAL_PARAMS = {'sign2': {'adanet': {'cpu': 16}, 'node2vec': {'cpu': 4}},
                  'neig1': {'cpu': 15},
                  'neig2': {'cpu': 15},
                  'neig3': {'cpu': 15},
                  'sign3': {'cpu': 32},
                  'clus1': {'cpu': 10},
                  'clus2': {'cpu': 10},
                  'clus3': {'cpu': 10},
                  'proj1': {'cpu': 10},
                  'proj2': {'cpu': 10},
                  'proj3': {'cpu': 10}}


@logged
class CCPredict(BaseTask):

    def __init__(self, name=None, cc_type=None, **params):
        """Initialize CC predict task.

        Args:
            name (str): The name of the task (default:None)
            cc_type (str): The CC type where the fit is applied (Required)
            CC_ROOT (str): The CC root path (Required)
            datasets (list): The list of dataset codes to apply the fit
                (Optional, all datasets taken by default)
            ds_data_params (dict): A dictionary with key is dataset code and
                value is another dictionary with all specific parameters for
                that dataset. (Optional)
            general_data_params (dict): A dictionary with general parameters
                for all datasets (Optional)
            datasets_input_files (dict): A dictionary with is a dataset code
                and value the input file for the predict.
            output_path (str): The path where to save the output files from
                the predict.
            output_file (str): Name of the output file for the prediction.
        """
        if cc_type is None:
            raise Exception("CCPredict requires a cc_type")

        if name is None:
            name = cc_type
        args = []
        task_id = params.get('task_id', None)
        if task_id is None:
            params['task_id'] = name
        BaseTask.__init__(self, name, **params)

        if cc_type not in CC_TYPES_DEPENDENCIES.keys():
            raise Exception('CC Type ' + cc_type + ' not supported')

        self.cc_type = cc_type

        self.CC_ROOT = params.get('CC_ROOT', None)
        if self.CC_ROOT is None:
            raise Exception('CC_ROOT parameter is not set')

        self.ds_data_params = params.get('ds_params', None)
        self.general_data_params = params.get('general_params', None)

        self.output_path = params.get("output_path", None)
        self.output_file = params.get("output_file", None)
        self.datasets_input_files = params.get("datasets_input_files", None)
        self.datasets = params.get("datasets", None)

        if self.output_file is None:
            self.output_file = self.cc_type + ".h5"

        if self.output_path is None:
            raise Exception("No output_path defined")

        if self.datasets is None:
            raise Exception("There is no datasets to predict " + self.cc_type)

    def run(self):
        """Run the task."""

        config_cc = Config()
        cc = ChemicalChecker(self.CC_ROOT)

        branch = 'reference'

        if self.cc_type == 'sign0' or self.cc_type == 'sign3':
            branch = 'full'

        # If not in sign0 then we need to get input data files
        if self.cc_type != 'sign0':
            if self.datasets_input_files is None:

                dataset_codes_files = {}
                for code in self.datasets:
                    dataset_codes_files[code] = os.path.join(
                        self.output_path, code,
                        CC_TYPES_DEPENDENCIES[self.cc_type][0] + ".h5")
                self.datasets_input_files = dataset_codes_files

            else:
                for ds, filename in self.datasets_input_files.items():
                    if not os.path.exists(self.datasets_input_files[ds]):
                        raise Exception(
                            "Expected input file %s not present" %
                            self.datasets_input_files[code])

        for ds in self.datasets:
            sign = cc.get_signature(self.cc_type, branch, ds)
            if not sign.is_fit():
                raise Exception("Dataset %s is not trained yet" % ds)

        dataset_params = list()

        for ds_code in self.datasets:

            input_data_file = None
            if self.datasets_input_files is not None:
                input_data_file = self.datasets_input_files[ds_code]

            if isinstance(self.ds_data_params, Config):
                temp_dict = self.ds_data_params.asdict()
            else:
                temp_dict = self.ds_data_params
            if temp_dict is not None and ds_code in temp_dict.keys():
                if isinstance(temp_dict[ds_code], Config):
                    dict_params = temp_dict[ds_code].asdict()
                else:
                    dict_params = temp_dict[ds_code]

                if self.cc_type in SPECIAL_PARAMS:
                    dict_params.update(SPECIAL_PARAMS[self.cc_type])
                if self.general_data_params is not None:
                    dict_params.update(self.general_data_params)

                dataset_params.append(
                    (ds_code, input_data_file, dict_params))
            else:
                dataset_params.append(
                    (ds_code, input_data_file, None))

        job_path = None
        if len(self.datasets) > 0:

            self.datasets.sort()

            job_path = tempfile.mkdtemp(
                prefix='jobs_' + self.cc_type + '_pred_', dir=self.tmpdir)

            if not os.path.isdir(job_path):
                os.mkdir(job_path)

            # create script file
            cc_config_path = os.environ['CC_CONFIG']
            cc_package = os.path.join(config_cc.PATH.CC_REPO, 'package')
            script_lines = [
                "import sys, os",
                "import pickle",
                "from chemicalchecker.core import ChemicalChecker",
                "from chemicalchecker.core import DataSignature",
                "task_id = sys.argv[1]",  # <TASK_ID>
                "filename = sys.argv[2]",  # <FILE>
                # load pickled data
                "inputs = pickle.load(open(filename, 'rb'))",
                # elements for current job
                "dataset = inputs[task_id][0][0]",
                "dataset_file = inputs[task_id][0][1]",
                "pars = inputs[task_id][0][2]",  # elements for current job
                # elements for current job
                "input_file = dataset_file",
                "cc = ChemicalChecker('%s' )" % self.CC_ROOT,
                'if pars is None: pars = {}',
                "output_file=os.path.join('%s', dataset, '%s')" % (
                    self.output_path, self.output_file),
                "if not os.path.exists(os.path.dirname(output_file)):",
                "    os.makedirs(os.path.dirname(output_file))"

            ]

            if self.cc_type == 'sign0':
                script_lines += [
                    'sign_full = cc.get_signature("%s","%s",dataset)' % (
                        self.cc_type, branch),
                    "pars['destination'] = output_file",
                    "sign_full.predict(**pars)"]
            else:
                script_lines += [
                    'sign_full = cc.get_signature("%s","%s",dataset, **pars)' % (
                        self.cc_type, branch),
                    "sign_full.predict(DataSignature(input_file),output_file)"]

            script_lines += ["print('JOB DONE')"]

            script_name = os.path.join(
                job_path, self.cc_type + '_pred_script.py')
            with open(script_name, 'w') as fh:
                for line in script_lines:
                    fh.write(line + '\n')
            # hpc parameters

            params = {}
            params["num_jobs"] = len(self.datasets)
            params["jobdir"] = job_path
            params["job_name"] = "CC_PRD_" + self.cc_type.upper()
            params["elements"] = dataset_params
            params["wait"] = True
            params["memory"] = CC_TYPES_MEM_CPU[self.cc_type][0]
            params["cpu"] = CC_TYPES_MEM_CPU[self.cc_type][1]
            # job command
            singularity_image = Config().PATH.SINGULARITY_IMAGE
            command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} singularity exec {} python {} <TASK_ID> <FILE>".format(
                cc_package, cc_config_path, singularity_image, script_name)
            # submit jobs
            cluster = HPC.from_config(config_cc)
            jobs = cluster.submitMultiJob(command, **params)

        dataset_not_done = []
        time.sleep(5)

        for code in self.datasets:

            check_file = os.path.join(self.output_path, code, self.output_file)

            if not os.path.exists(check_file):
                dataset_not_done.append(code)
                self.__log.error(
                    self.cc_type + " predict failed for dataset code: " + code)

        if len(dataset_not_done) > 0:
            if not self.custom_ready():
                raise Exception("Some predictions failed")
        else:
            self.mark_ready()
            if job_path is not None:
                shutil.rmtree(job_path)

    def execute(self, context):
        """Same as run but for Airflow."""
        self.tmpdir = context['params']['tmpdir']
        self.run()
