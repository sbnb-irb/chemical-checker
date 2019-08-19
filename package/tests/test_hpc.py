import os
import pytest
import unittest
import functools

from chemicalchecker.util import HPC
from chemicalchecker.util import Config


def skip_if_import_exception(function):
    """Assist in skipping tests failing because of missing dependencies."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ImportError as err:
            pytest.skip(str(err))
    return wrapper


class TestHPC(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        os.environ["CC_CONFIG"] = os.path.join(
            self.data_dir, 'config.json')

    @skip_if_import_exception
    def test_hpc(self):
        config = Config()
        conf_dict = config.HPC.asdict()
        conf_dict["dry_run"] = True
        cluster = HPC(**conf_dict)
        self.assertTrue(cluster.status() is None)
