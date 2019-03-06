import os
import pytest
import unittest
import functools

from chemicalchecker.util.remove_near_duplicates import RNDuplicates


def skip_if_import_exception(function):
    """Assist in skipping tests failing because of missing dependencies."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ImportError as err:
            pytest.skip(str(err))
    return wrapper


class TestChemicalChecker(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        os.environ["CC_CONFIG"] = os.path.join(
            self.data_dir, 'config.json')

    @skip_if_import_exception
    def test_remove_near(self):
        file = os.path.join(self.data_dir, 'test_remove.h5')
        rnd = RNDuplicates()
        keys, data, maps = rnd.remove(file)
        self.assertTrue(len(keys) == 1440)

    @skip_if_import_exception
    def test_remove_only_duplicates(self):
        file = os.path.join(self.data_dir, 'test_remove.h5')
        rnd = RNDuplicates(only_duplicates=True)
        keys, data, maps = rnd.remove(file)
        self.assertTrue(len(keys) == 3648)
