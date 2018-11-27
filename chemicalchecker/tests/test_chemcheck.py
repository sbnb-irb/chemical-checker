import os
import unittest
import shutil
import itertools

from chemicalchecker import ChemicalChecker


def coordinates():
    """Iterator on Chemical Checker coordinates."""
    for name, code in itertools.product("ABCDE", "12345"):
        yield name + code


class TestConfig(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')

    def test_init(self):
        cc_root = os.path.join(self.data_dir, 'full')
        self.assertFalse(os.path.isdir(cc_root))
        cc_full = ChemicalChecker(cc_root)
        self.assertTrue(os.path.isdir(cc_root))
        for coords in coordinates():
            path1 = os.path.join(cc_root, coords[:1])
            self.assertTrue(os.path.isdir(path1))
            path2 = os.path.join(cc_root, coords[:1], coords[:2])
            self.assertTrue(os.path.isdir(path2))

        sign_file = os.path.join(cc_root, 'A', 'A1', 'A1.001', 'sign1.h5')
        self.assertEqual(cc_full.get_data_path('sign1', 'A1.001'), sign_file)
        shutil.rmtree(cc_root)
