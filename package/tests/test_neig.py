import os
import mock
import h5py
import shutil
import pytest
import functools
import unittest
import numpy as np
from scipy.spatial.distance import cosine

from chemicalchecker.core import ChemicalChecker


def skip_if_import_exception(function):
    """Assist in skipping tests failing because of missing dependencies."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ImportError as err:
            pytest.skip(str(err))
    return wrapper


class TestNeigh(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        os.environ["CC_CONFIG"] = os.path.join(
            self.data_dir, 'config.json')

    def tearDown(self):
        if os.path.exists(self.cc_root):
            shutil.rmtree(self.cc_root)

    @skip_if_import_exception
    def test_neig1(self):
        cc_root = os.path.join(self.data_dir, 'alpha')
        self.cc_root = cc_root
        self.assertFalse(os.path.isdir(cc_root))

        sign1 = mock.Mock()
        sign1.data_path = os.path.join(self.data_dir, "mock_sign1.h5")
        sign1.molset = 'reference'
        self.data_path = sign1.data_path
        with h5py.File(sign1.data_path) as hf:
            self.shape = hf["V"].shape
        sign1.chunker = self.chunker
        cc = ChemicalChecker(cc_root)
        self.assertTrue(os.path.isdir(cc_root))
        coords = "B1.001"
        neig1_ref = cc.get_signature("neig1", "reference", coords)
        path_test1 = os.path.join(cc_root, 'reference', coords[
            :1], coords[:2], coords, "neig1")
        self.assertTrue(os.path.isdir(path_test1))
        path_test2 = os.path.join(cc_root, 'reference', coords[:1], coords[
            :2], coords, "neig1", "models")
        self.assertTrue(os.path.isdir(path_test2))
        neig1_ref.fit(sign1)

        with h5py.File(sign1.data_path) as hf:
            ini_data = hf["V"][:]

        self.assertTrue(os.path.isfile(os.path.join(path_test1, "neig.h5")))

        with h5py.File(os.path.join(path_test1, "neig.h5")) as hf:
            indices = hf["indices"][:]
            distances = hf["distances"][:]

            x = ini_data[2]
            y = ini_data[indices[2, 10]]

            val = 1.0 - (np.dot(x, y) / (np.sqrt(np.dot(x, x))
                                         * np.sqrt(np.dot(y, y))))

            val = cosine(x, y)

            a = "%.5f" % distances[2, 10]
            b = "%.5f" % val

            self.assertAlmostEqual(distances[2, 10], val, places=5)
            self.assertTrue(a == b)

    def chunker(self, size=2000):
        """Iterate on signatures."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        for i in range(0, self.shape[0], size):
            yield slice(i, i + size)
