"""Utility to run pipelines


"""
import os
import sys
from chemicalchecker.util import logged
import logging


@logged
class Pipeline():
    """Run pipelines according to config files."""

    def __init__(self, config, steps_path):
        """Initialize the Pipeline object.

        """

        self.steps_path = steps_path

        self.config = config

        self.readydir = os.path.join(self.config.PATH, "ready")

        self.logdir = os.path.join(self.config.PATH, "log")

        self.logfile = os.path.join(self.config.PATH, "log", "pipeline.log")

        self.tmpdir = os.path.join(self.config.PATH, "tmp")

        if os.path.exists(steps_path + "/steps") is False:
            raise Exception(
                "There is no directory steps in " + steps_path + " path")

        if os.path.exists(steps_path + "/steps/__init__.py") is False:
            raise Exception("The directory " + steps_path + "/steps should contain a __init__.py" +
                            " file in order to run the pipeline.")

        if os.path.exists(self.config.PATH) is False:
            os.makedirs(self.config.PATH)

        if os.path.exists(self.readydir) is False:
            os.makedirs(self.readydir)

        if os.path.exists(self.tmpdir) is False:
            os.makedirs(self.tmpdir)

        if os.path.exists(self.logdir) is False:
            os.makedirs(self.logdir)

        # get chemical checker logger
        logger = logging.getLogger()
        # we use a FileHandler to save to file
        fh = logging.FileHandler(self.logfile)
        # set logging level here
        fh.setLevel(logging.DEBUG)
        # define the formatter and set it
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s [%(levelname)-8s] %(message)s')
        fh.setFormatter(formatter)
        # add handler to logger
        logger.addHandler(fh)

    def run(self):
        """Run the pipeline."""

        sys.path.append(self.steps_path)

        from steps import *

        params = {}
        params["readydir"] = self.readydir
        params["tmpdir"] = self.tmpdir

        for step in self.config.RUN:

            try:
                Pipeline.__log.debug("initializing object %s", step)
                current_step = eval(step)(self.config, step, **params)
            except Exception as ex:
                Pipeline.__log.debug(ex)
                raise Exception("Step %s not available" % step)

            if current_step.is_ready():
                Pipeline.__log.info(
                    "Step: " + step + " already done. Skipping...")
                continue

            current_step.run()

            if not current_step.is_ready():
                Pipeline.__log.error(
                    "Pipeline failed in step " + current_step.name + ". Please, check errors.")
                break
