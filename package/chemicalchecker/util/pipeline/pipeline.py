"""Utility to run pipelines."""
import os
import logging

from chemicalchecker.util import logged


@logged
class Pipeline():
    """Pipeline class.

    Run pipelines according to config files.
    """

    def __init__(self, pipeline_path=None, **kwargs):
        """Initialize a Pipeline instance.

        pipeline_path (str): Path where the pipeline will set its structure.
        """
        self.tasks = []

        self.pipeline_path = pipeline_path

        if self.pipeline_path is None:
            raise Exception("pipeline_path parameter not set")
        else:
            Pipeline.__log.info("PIPELINE PATH: {}".format(self.pipeline_path) )

        self.readydir = os.path.join(self.pipeline_path, "ready")
        self.logdir = os.path.join(self.pipeline_path, "log")
        self.logfile = os.path.join(
            self.pipeline_path, "log", "pipeline.log")
        self.tmpdir = os.path.join(self.pipeline_path, "tmp")

        # check and make needed directories
        if not os.path.exists(self.pipeline_path):
            os.makedirs(self.pipeline_path)
        if not os.path.exists(self.readydir):
            os.makedirs(self.readydir)
        if not os.path.exists(self.tmpdir):
            os.makedirs(self.tmpdir)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        # log to file
        logger = logging.getLogger()
        fh = logging.FileHandler(self.logfile)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s [%(levelname)-8s] %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def add_task(self, task):
        """Add tasks to the pipeline."""
        task.set_dirs(self.readydir, self.tmpdir)
        self.tasks.append(task)

    def run(self):
        """Run the pipeline."""
        for task in self.tasks:

            # check if already done
            if task.is_ready():
                Pipeline.__log.info(
                    "Task: '%s' already done. Skipping...", task.name)
                continue
            # run it
            Pipeline.__log.info(
                "Starting task: '%s' ...", task.name)
            task.run()
            # after runnin we expect the step to be in ready state
            if not task.is_ready():
                # if not we report the error
                Pipeline.__log.error(
                    "Pipeline failed in task %s. Please, check errors.",
                    task.name)
                break
            else:
                Pipeline.__log.info(
                    "Done task: '%s' ...", task.name)

    def clean(self, step=None):
        """Clean all or some of the pipeline steps."""
        if step is None:

            for current_step in self.tasks:

                # clean it
                current_step.clean()

        else:

            for task in self.tasks:

                if task.name == step:
                    task.clean()
