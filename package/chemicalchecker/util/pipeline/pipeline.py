"""Utility to run pipelines


"""
import os
import sys
import logging
from importlib import import_module

from chemicalchecker.util import logged


@logged
class Pipeline():
    """Run pipelines according to config files."""

    def __init__(self, config, steps_path):
        """Initialize the Pipeline object.

        config(Config): a `Config` object.
        steps_path(str): the path containing the step directory.
        """

        # read the config
        self.config = config
        self.steps_path = steps_path
        self.readydir = os.path.join(self.config.PATH, "ready")
        self.logdir = os.path.join(self.config.PATH, "log")
        self.logfile = os.path.join(self.config.PATH, "log", "pipeline.log")
        self.tmpdir = os.path.join(self.config.PATH, "tmp")
        # check steps directory structure
        step_init_file = os.path.join(steps_path, '__init__.py')
        if not os.path.exists(step_init_file):
            raise Exception("The directory %s " % steps_path +
                            "should contain a __init__.py " +
                            "file in order to run the pipeline.")
        # check and make needed directories
        if not os.path.exists(self.config.PATH):
            os.makedirs(self.config.PATH)
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

    def run(self):
        """Run the pipeline."""
        # read general parameters
        params = {}
        params["readydir"] = self.readydir
        params["tmpdir"] = self.tmpdir
        # iterate on the steps
        for step in self.config.RUN:
            # initialize the step
            try:
                Pipeline.__log.debug("initializing object '%s'", step)
                cwd = os.getcwd()
                os.chdir(self.steps_path)
                module_object = import_module('__init__')
                step_class = getattr(module_object, step)
                current_step = step_class(self.config, step, **params)
                os.chdir(cwd)
            except Exception as ex:
                Pipeline.__log.debug(ex)
                raise Exception("Step '%s' not available" % step)
            # check if already done
            if current_step.is_ready():
                Pipeline.__log.info(
                    "Step: '%s' already done. Skipping...", step)
                continue
            # run it
            current_step.run()
            # after runnin we expect the step to be in ready state
            if not current_step.is_ready():
                # if not we report the error
                Pipeline.__log.error(
                    "Pipeline failed in step %s. Please, check errors.",
                    current_step.name)
                break

    def clean(self, step=None, substep=None):
        """Clean all or some of the pipeline steps"""

        params = {}
        params["readydir"] = self.readydir
        params["tmpdir"] = self.tmpdir

        if step is None:

            for step in self.config.RUN:
                # initialize the step
                try:
                    Pipeline.__log.debug("initializing object '%s'", step)
                    cwd = os.getcwd()
                    os.chdir(self.steps_path)
                    module_object = import_module('__init__')
                    step_class = getattr(module_object, step)
                    current_step = step_class(self.config, step, **params)
                    os.chdir(cwd)
                except Exception as ex:
                    Pipeline.__log.debug(ex)

                # clean it
                current_step.clean()

        else:

            # initialize the step
            try:
                Pipeline.__log.debug("initializing object '%s'", step)
                cwd = os.getcwd()
                os.chdir(self.steps_path)
                module_object = import_module('__init__')
                step_class = getattr(module_object, step)
                current_step = step_class(self.config, step, **params)
                os.chdir(cwd)
            except Exception as ex:
                Pipeline.__log.debug(ex)

            # clean it
            current_step.clean(substep)
