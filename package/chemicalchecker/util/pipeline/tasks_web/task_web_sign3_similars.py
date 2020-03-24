import tempfile
import os
import shutil

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.util import Config
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util.pipeline import BaseTask
from chemicalchecker.util import HPC
from airflow.models import BaseOperator
from airflow import AirflowException


@logged
class SimilarsSign3(BaseTask, BaseOperator):

    def __init__(self, config=None, name=None, **params):

        args = []

        task_id = params.get('task_id', None)

        if task_id is None:
            params['task_id'] = name

        BaseTask.__init__(self, config, name, **params)
        BaseOperator.__init__(self, *args, **params)

        self.CC_ROOT = params.get('CC_ROOT', None)
        if self.CC_ROOT is None:
            raise Exception('CC_ROOT parameter is not set')

    def run(self):
        """Run the molprops step."""

        all_datasets = Dataset.get()
        config_cc = Config()

        filename = "similars.h5"

        cc = ChemicalChecker(self.CC_ROOT)
        dataset_codes = list()
        for ds in all_datasets:
            if not ds.exemplary:
                continue
            sign3 = cc.get_signature("sign3", "full", ds.dataset_code)

            similars_file = os.path.join(sign3.signature_path, filename)

            if os.path.exists(similars_file):
                continue

            dataset_codes.append(ds.dataset_code)

        job_path = None

        if len(dataset_codes) > 0:

            job_path = tempfile.mkdtemp(
                prefix='jobs_simsign3_', dir=self.tmpdir)

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
                "from chemicalchecker.core import ChemicalChecker",
                "task_id = sys.argv[1]",  # <TASK_ID>
                "filename = sys.argv[2]",  # <FILE>
                "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
                "data = str(inputs[task_id][0])",  # elements for current job
                # elements are indexes
                "cc = ChemicalChecker( '%s')" % self.CC_ROOT,
                # start import
                'sign3 = cc.get_signature("sign3", "full", data)',
                # start import
                "pars = {'cpu': 10}",
                'neig2_ref = cc.get_signature("neig2","reference",data, **pars)',
                'similars_file = os.path.join(sign3.signature_path, "%s")' % filename,
                # start import
                "neig2_ref.predict(sign3,destination=similars_file)",
                "print('JOB DONE')"
            ]

            script_name = os.path.join(job_path, 'sim3_script.py')
            with open(script_name, 'w') as fh:
                for line in script_lines:
                    fh.write(line + '\n')
            # hpc parameters

            params = {}
            params["num_jobs"] = len(dataset_codes)
            params["jobdir"] = job_path
            params["job_name"] = "CC_SIM3"
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

            sign3 = cc.get_signature("sign3", "full", code)

            similars_file = os.path.join(sign3.signature_path, filename)

            if os.path.exists(similars_file):
                continue

            dataset_not_done.append(code)
            self.__log.warning(
                "SimilarsSign3 fit failed for dataset code: " + code)

        if len(dataset_not_done) == 0:
            self.mark_ready()
            if job_path is not None:
                shutil.rmtree(job_path)
        else:
            if not self.custom_ready():
                raise AirflowException("Not all similars were calculated correctly")

    def execute(self, context):
        """Run the molprops step."""
        self.tmpdir = context['params']['tmpdir']

        self.run()
