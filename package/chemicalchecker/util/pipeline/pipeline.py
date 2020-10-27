"""Utility to run pipelines."""
import os
import logging

from chemicalchecker.util import logged


@logged
class Pipeline():
    """Pipeline class.

    Run pipelines according to config files.
    """

    def __init__(self, pipeline_path, config, keep_jobs=False):
        """Initialize a Pipeline instance.

        pipeline_path (str): Path where the pipeline will set its structure.
        keep_jobs (bool): If True temporary job directories will not be
            deleted.
        """
        self.tasks = []
        self.pipeline_path = pipeline_path
        Pipeline.__log.info("PIPELINE PATH: {}".format(self.pipeline_path))
        self.config = config

        self.readydir = os.path.join(self.pipeline_path, "ready")
        self.logdir = os.path.join(self.pipeline_path, "log")
        self.tmpdir = os.path.join(self.pipeline_path, "tmp")
        self.cachedir = os.path.join(self.pipeline_path, "cache")
        self.keep_jobs = keep_jobs

        # check and make needed directories
        if not os.path.exists(self.pipeline_path):
            os.makedirs(self.pipeline_path)
        if not os.path.exists(self.readydir):
            os.makedirs(self.readydir)
        if not os.path.exists(self.tmpdir):
            os.makedirs(self.tmpdir)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        if not os.path.exists(self.cachedir):
            os.makedirs(self.cachedir)

        # log to file
        self.logfile = os.path.join(pipeline_path, "log", "pipeline.log")
        logger = logging.getLogger()
        fh = logging.FileHandler(self.logfile)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s [%(levelname)-8s] %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def add_task(self, task):
        """Add tasks to the pipeline."""
        task.set_dirs(self.readydir, self.tmpdir, self.cachedir)
        task.keep_jobs = self.keep_jobs
        task.config = self.config
        self.tasks.append(task)

    def insert_task(self, position, task):
        """Add tasks to the pipeline."""
        task.set_dirs(self.readydir, self.tmpdir, self.cachedir)
        task.keep_jobs = self.keep_jobs
        task.config = self.config
        self.tasks.insert(position, task)

    def run(self, include_tasks=None):
        """Run the pipeline."""
        if include_tasks is None:
            include_tasks = [t.name for t in self.tasks]
        for task in self.tasks:
            # check if among those we want to tun
            if task.name not in include_tasks:
                Pipeline.__log.info(
                    "Task: '%s'. Skipping...", task.name)
                continue
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
