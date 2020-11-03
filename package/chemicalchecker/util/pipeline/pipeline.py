"""Utility to run pipelines."""
import os
import logging

from chemicalchecker.util import logged


@logged
class Pipeline():
    """Pipeline class.

    Run pipelines according to config files.
    """

    def __init__(self, pipeline_path, config, keep_jobs=False,
                 only_tasks=[], exclude_tasks=[]):
        """Initialize a Pipeline instance.

        pipeline_path (str): Path where the pipeline will set its structure.
        keep_jobs (bool): If True temporary job directories will not be
            deleted.
        """
        self.tasks = []
        self.pipeline_path = pipeline_path
        self.__log.info("PIPELINE PATH: {}".format(self.pipeline_path))
        self.config = config
        self.only_tasks = only_tasks
        self.exclude_tasks = exclude_tasks

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

    def _check_task(self, task):
        """Decide if task can be included."""
        if any([task.name.startswith(st) for st in self.exclude_tasks]):
            self.__log.info("Task EXCLUDED: '%s'", task.name)
            return False
        if len(self.only_tasks) > 0:
            if any([task.name.startswith(st) for st in self.only_tasks]):
                self.__log.info("Task SELECTED: '%s'", task.name)
                return True
            else:
                self.__log.info("Task EXCLUDED: '%s'", task.name)
                return False
        self.__log.info("Task INCLUDED: '%s'", task.name)
        return True

    def add_task(self, task):
        """Add tasks to the pipeline."""
        task.set_dirs(self.readydir, self.tmpdir, self.cachedir)
        task.keep_jobs = self.keep_jobs
        task.config = self.config
        if self._check_task(task):
            self.tasks.append(task)

    def insert_task(self, position, task):
        """Add tasks to the pipeline."""
        task.set_dirs(self.readydir, self.tmpdir, self.cachedir)
        task.keep_jobs = self.keep_jobs
        task.config = self.config
        if self._check_task(task):
            self.tasks.insert(position, task)

    def run(self):
        """Run the pipeline."""
        for task in self.tasks:
            # check if already done
            if task.is_ready():
                self.__log.info(
                    "Task: '%s' already done. Skipping...", task.name)
                continue
            # run it
            self.__log.info(
                "Starting task: '%s' ...", task.name)
            task.run()
            # after runnin we expect the step to be in ready state
            if not task.is_ready():
                # if not we report the error
                self.__log.error(
                    "Pipeline failed in task %s. Please, check errors.",
                    task.name)
                break
            else:
                self.__log.info(
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
