import os
import unittest

from chemicalchecker.util.hpc import HPC
from chemicalchecker.util import Config


class TestHPC(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        os.environ["CC_CONFIG"] = os.path.join(
            self.data_dir, 'config.json')

    def test_hpc(self):
        config = Config()
        cluster = HPC(config, True)
        self.assertTrue(cluster.status() is None)
