import os
import unittest
import shutil
import mock
import h5py
import numpy as np
import pytest

from chemicalchecker import ChemicalChecker


class TestProj1(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        os.environ["CC_CONFIG"] = os.path.join(
            self.data_dir, 'config.json')

    def tearDown(self):
        if os.path.exists(self.cc_root):
            shutil.rmtree(self.cc_root)
        if os.path.isfile(os.path.join(self.data_dir, "test_proj1.h5")):
            os.remove(os.path.join(self.data_dir, "test_proj1.h5"))

    #@pytest.mark.skip(reason="Faiss is not available on test enviroment")
    def test_proj1(self):
        cc_root = os.path.join(self.data_dir, 'alpha')
        self.cc_root = cc_root
        self.assertFalse(os.path.isdir(cc_root))

        sign1 = mock.Mock()
        sign1.data_path = os.path.join(self.data_dir, "mock_sign1.h5")
        sign1.background_distances = self.background_distances
        cc = ChemicalChecker(cc_root)
        self.assertTrue(os.path.isdir(cc_root))
        coords = "B1.001"
        proj1_ref = cc.get_signature("proj1", "reference", coords)
        proj1_ref.validation_path = os.path.join(self.data_dir, "validation_sets")
        path_test1 = os.path.join(cc_root, 'reference', coords[
            :1], coords[:2], coords, "proj1")
        self.assertTrue(os.path.isdir(path_test1))
        path_test2 = os.path.join(cc_root, 'reference', coords[:1], coords[
            :2], coords, "proj1", "models")
        self.assertTrue(os.path.isdir(path_test2))
        proj1_ref.fit(sign1)

        proj1_ref.predict(sign1, os.path.join(self.data_dir, "test_proj1.h5"))

        self.assertTrue(os.path.isfile(
            os.path.join(self.data_dir, "test_proj1.h5")))

        with h5py.File(os.path.join(self.data_dir, "test_proj1.h5")) as hf:
            proj_pred = hf["V"][:]

        self.assertTrue(os.path.isfile(os.path.join(path_test1, "proj1.h5")))

        with h5py.File(os.path.join(path_test1, "proj1.h5")) as hf:
            proj = hf["V"][:]

        np.testing.assert_array_almost_equal(proj, proj_pred)

    def background_distances(self, metric=None):

        bg_distances = {}
        if metric == "cosine":
            bg_file = os.path.join(self.model_path, "bg_cosine_distances.h5")
            if not os.path.isfile(bg_file):
                raise Exception(
                    "The background distances for metric " + metric + " are not available.")
            f5 = h5py.File(bg_file)
            bg_distances["distance"] = f5["distance"][:]
            bg_distances["pvalue"] = f5["pvalue"][:]

        if metric == "euclidean":
            bg_file = os.path.join(
                self.data_dir, "bg_euclidean_distances.h5")
            if not os.path.isfile(bg_file):
                raise Exception(
                    "The background distances for metric " + metric + " are not available.")
            f5 = h5py.File(bg_file)
            bg_distances["distance"] = f5["distance"][:]
            bg_distances["pvalue"] = f5["pvalue"][:]

        if len(bg_distances) == 0:
            raise Exception(
                "The background distances for metric " + metric + " are not available.")
        else:
            return bg_distances
