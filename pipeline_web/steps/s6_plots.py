import os
import h5py
import tempfile
from shutil import copyfile
import shutil
from chemicalchecker.util import logged
from chemicalchecker.util import BaseStep
from chemicalchecker.util import Config
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util import HPC
from chemicalchecker.database import Dataset


@logged
class Plots(BaseStep):

    def __init__(self, config, name, **params):

        BaseStep.__init__(self, config, name, **params)

    def run(self):
        """Run the coordinates step."""

        all_datasets = Dataset.get()
        config_cc = Config()

        cc_config_path = os.environ['CC_CONFIG']
        cc_package = os.path.join(config_cc.PATH.CC_REPO, 'package')
        script_path = os.path.join(
            config_cc.PATH.CC_REPO, "pipeline_web", "steps", "scripts", "make_plots.py")

        cc = ChemicalChecker(config_cc.PATH.CC_ROOT)

        self.__log.info("Copying projections plots")

        plots_dir = os.path.join(config_cc.PATH.CC_ROOT, "plots_web")

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

        with h5py.File(universe_file) as h5:
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
            cc_package, cc_config_path, singularity_image, script_path, self.config.MOLECULES_PATH)
        # submit jobs
        cluster = HPC(config_cc)
        cluster.submitMultiJob(command, **params)

        if cluster.status() == HPC.READY:
            self.mark_ready()
            shutil.rmtree(job_path)
        else:
            raise Exception("Some molecules did not get the plots right.")
