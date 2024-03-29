"""Abstract task class.

Each task class derived from this base class will have to implement several
methods.
"""
import os
import six
import glob

from abc import ABCMeta, abstractmethod

from chemicalchecker.util import logged


@logged
@six.add_metaclass(ABCMeta)
class BaseTask(object):
    """BaseTask class.

    A Step base class. Implements methods and checks common to all steps.
    """

    @abstractmethod
    def __init__(self, name, **params):
        """Initialize the Step."""
        self.name = name
        self.readyfile = name + ".ready"
        self.readydir = params.get("readydir", '')
        self.tmpdir = params.get("tmpdir", '')

    @abstractmethod
    def run(self):
        """Run the step."""
        BaseTask.__log.info("Running step " + self.name)

    def set_dirs(self, readydir, tmpdir, cachedir):
        self.tmpdir = tmpdir
        self.readydir = readydir
        self.cachedir = cachedir

    def custom_ready(self):
        return self.readydir != ''

    def is_ready(self, substep=None):
        """Check if the step is already done."""
        if substep is None:
            filename = self.readyfile
        else:
            filename = self.name + "_" + substep + ".ready"
        return os.path.exists(os.path.join(self.readydir, filename))

    def mark_ready(self, substep=None):
        """Mark the step as done."""
        if not self.custom_ready():
            BaseTask.__log.debug('Not ready dir so skip mark_ready')
        else:
            BaseTask.__log.debug('mark_ready')
            if substep is None:
                filename = os.path.join(self.readydir, self.readyfile)
            else:
                filename = os.path.join(
                    self.readydir, self.name + "_" + substep + ".ready")
            with open(filename, 'w') as fh:
                pass

    def clean(self, substep=None):
        """Clean the step."""
        BaseTask.__log.debug('clean')
        if substep is None:
            dir_regex = os.path.join(self.readydir, self.name + "_*.ready")
            for filename in glob.glob(dir_regex):
                os.remove(filename)
        else:
            filename = os.path.join(
                self.readydir, self.name + "_" + substep + ".ready")
            if os.path.exists(filename):
                os.remove(filename)

        filename = os.path.join(self.readydir, self.readyfile)
        if os.path.exists(filename):
            os.remove(filename)
