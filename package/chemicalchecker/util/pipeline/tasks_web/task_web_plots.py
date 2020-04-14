import os
import h5py
import tempfile
from shutil import copyfile
import shutil
from chemicalchecker.util import logged
from chemicalchecker.util.pipeline import BaseTask
from chemicalchecker.util import Config
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util import HPC
from chemicalchecker.database import Dataset
from airflow.models import BaseOperator
from airflow import AirflowException

@logged
class Plots(BaseTask, BaseOperator):

    def __init__(self, name=None, **params):

        args = []

        task_id = params.get('task_id', None)

        if task_id is None:
            params['task_id'] = name

        BaseTask.__init__(self, name, **params)
        BaseOperator.__init__(self, *args, **params)

        self.DB = params.get('DB', None)
        if self.DB is None:
            raise Exception('DB parameter is not set')
        self.CC_ROOT = params.get('CC_ROOT', None)
        if self.CC_ROOT is None:
            raise Exception('CC_ROOT parameter is not set')
        self.MOLECULES_PATH = params.get('MOLECULES_PATH', None)
        if self.MOLECULES_PATH is None:
            raise Exception('MOLECULES_PATH parameter is not set')


    def run(self):
        """Run the coordinates step."""

        all_datasets = Dataset.get()
        config_cc = Config()

        cc_config_path = os.environ['CC_CONFIG']
        cc_package = os.path.join(config_cc.PATH.CC_REPO, 'package')
        script_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "scripts/make_plots.py")

        cc = ChemicalChecker(self.CC_ROOT)

        self.__log.info("Copying projections plots")

        plots_dir = os.path.join(self.CC_ROOT, "plots_web")

        if not os.path.exists(plots_dir):
            os.mkdir(plots_dir)

        for ds in all_datasets:
            if not ds.exemplary:
                continue

            proj2 = cc.get_signature('proj2', 'reference', ds.dataset_code)
            src_plot_file = os.path.join(proj2.stats_path, "largevis.png")
            dest_plot_file = os.path.join(
                plots_dir, ds.coordinate + "_largevis.png")

            if not os.path.exists(src_plot_file):
                raise Exception("Projection plot for dataset " +
                                ds.dataset_code + " is not available.")

            copyfile(src_plot_file, dest_plot_file)

        self.__log.info("Finding missing molecule plots")

        universe_file = os.path.join(self.tmpdir, "universe.h5")

        with h5py.File(universe_file, 'r') as h5:
            keys = h5["keys"][:]

        datasize = keys.shape[0]

        keys.sort()

        job_path = tempfile.mkdtemp(
            prefix='jobs_molplot_', dir=self.tmpdir)

        params = {}
        params["num_jobs"] = datasize / 1000
        params["jobdir"] = job_path
        params["job_name"] = "CC_MOLPLOT"
        params["elements"] = keys
        params["wait"] = True
        # job command
        singularity_image = config_cc.PATH.SINGULARITY_IMAGE
        command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} singularity exec {} python {} <TASK_ID> <FILE> {}"
        command = command.format(
            cc_package, cc_config_path, singularity_image, script_path, self.MOLECULES_PATH)
        # submit jobs
        cluster = HPC.from_config(config_cc)
        cluster.submitMultiJob(command, **params)

        if cluster.status() == HPC.READY:
            self.mark_ready()
            shutil.rmtree(job_path)
        else:
            if not self.custom_ready():
                        raise AirflowException("Some molecules did not get the plots right.")
            else:
                self.__log.error("Some molecules did not get the plots right.")

    def execute(self, context):
        """Run the molprops step."""
        self.tmpdir = context['params']['tmpdir']

        self.run()
