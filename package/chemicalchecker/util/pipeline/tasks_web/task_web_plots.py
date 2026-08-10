import os
from shutil import copyfile

from chemicalchecker.database import Dataset
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util.pipeline import BaseTask
from chemicalchecker.util import logged


@logged
class Plots(BaseTask):

    def __init__(self, name=None, **params):
        task_id = params.get('task_id', None)
        if task_id is None:
            params['task_id'] = name
        BaseTask.__init__(self, name, **params)

        self.DB = params.get('DB', None)
        if self.DB is None:
            raise Exception('DB parameter is not set')
        self.CC_ROOT = params.get('CC_ROOT', None)
        if self.CC_ROOT is None:
            raise Exception('CC_ROOT parameter is not set')

    def run(self):
        """Copy the per-space projection plots into `plots_web`.

        NB: this task used to also render a 2D SVG per molecule into a
        `MOLECULES_PATH` tree. That was superseded by the `molinfo` task,
        which stores the SVG in the `molecular_info.molsvg` column, so only
        the projection plots are handled here.
        """
        all_datasets = Dataset.get()
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

        self.mark_ready()

    def execute(self, context):
        """Run the molprops step."""
        self.tmpdir = context['params']['tmpdir']
        self.run()
