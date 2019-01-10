import os
import unittest
import shutil
import mock
import h5py
import numpy as np
import pytest

from chemicalchecker import ChemicalChecker


class TestNeigh1(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')

    def tearDown(self):
        if os.path.exists(self.cc_root):
            shutil.rmtree(self.cc_root)

    @pytest.mark.skip(reason="Faiss is not available on test enviroment")
    def test_neig1(self):
        cc_root = os.path.join(self.data_dir, 'alpha')
        self.cc_root = cc_root
        self.assertFalse(os.path.isdir(cc_root))

        sign1 = mock.Mock()
        sign1.data_path = os.path.join(self.data_dir, "mock_sign1.h5")
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
        self.assertTrue(os.path.isfile(os.path.join(path_test2, "norms.h5")))

        with h5py.File(sign1.data_path) as hf:
            ini_data = hf["V"][:]

        self.assertTrue(os.path.isfile(os.path.join(path_test1, "neig1.h5")))

        with h5py.File(os.path.join(path_test1, "neig1.h5")) as hf:
            indices = hf["indices"][:]
            distances = hf["distances"][:]

            x = ini_data[2]
            y = ini_data[indices[2, 10]]

            val = 1.0 - (np.dot(x, y) / (np.sqrt(np.dot(x, x))
                                         * np.sqrt(np.dot(y, y))))

            a = "%.5f" % distances[2, 10]
            b = "%.5f" % val

            self.assertTrue(a == b)
