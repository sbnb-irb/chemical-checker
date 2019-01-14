import os
import unittest
import pytest


from chemicalchecker.util import RNDuplicates


class TestChemicalChecker(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')

    #@pytest.mark.skip(reason="Faiss is not available on test enviroment")
    def test_remove_near(self):
        file = os.path.join(self.data_dir, 'test_remove.h5')
        rnd = RNDuplicates()
        keys, data, maps = rnd.remove(file)
        self.assertTrue(len(keys) == 1440)

    #@pytest.mark.skip(reason="Faiss is not available on test enviroment")
    def test_remove_only_duplicates(self):
        file = os.path.join(self.data_dir, 'test_remove.h5')
        rnd = RNDuplicates(only_duplicates=True)
        keys, data, maps = rnd.remove(file)
        self.assertTrue(len(keys) == 3648)
