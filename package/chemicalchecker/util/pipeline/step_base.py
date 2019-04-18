"""Implementation of the abstract step class.

Each step class derived from this base class will have to implement several
methods. 
"""
import os
import six
import sys

from abc import ABCMeta, abstractmethod

from chemicalchecker.util import logged


@logged
@six.add_metaclass(ABCMeta)
class BaseStep(object):
    """A Step base class.

    Implements methods and checks common to all steps.
    """

    @abstractmethod
    def __init__(self, config, name, **params):
        """Initialize the Step with the config at the given path."""
        self.name = name
        self.readyfile = name + ".ready"
        self.readydir = params["readydir"]
        self.tmpdir = params["tmpdir"]
        self.config = config

    @abstractmethod
    def run(self):
        """Run the step."""
        BaseStep.__log.info("Running step " + self.name)

    def is_ready(self, substep=None):
        """Check if the step is already done."""
        BaseStep.__log.debug('is_ready')
        if substep is None:
            filename = self.readyfile
        else:
            filename = self.name + "_" + substep + ".ready"
        return os.path.exists(os.path.join(self.readydir, filename))

    def mark_ready(self, substep=None):
        """Mark the step as done."""
        if substep is None:
            filename = os.path.join(self.readydir, self.readyfile)
        else:
            filename = os.path.join(self.readydir, self.name + "_" + substep + ".ready")
        with open(filename, 'w') as fh:
            pass
