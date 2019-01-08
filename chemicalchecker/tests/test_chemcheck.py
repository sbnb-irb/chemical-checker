import os
import unittest
import shutil
import itertools

from chemicalchecker import ChemicalChecker


def coordinates():
    """Iterator on Chemical Checker coordinates."""
    for name, code in itertools.product("ABCDE", "12345"):
        yield name + code


class TestChemicalChecker(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')

    def tearDown(self):
        if os.path.exists(self.cc_root):
            shutil.rmtree(self.cc_root)

    def test_init(self):
        cc_root = os.path.join(self.data_dir, 'alpha')
        self.cc_root = cc_root
        self.assertFalse(os.path.isdir(cc_root))
        cc = ChemicalChecker(cc_root)
        self.assertTrue(os.path.isdir(cc_root))
        for coords in coordinates():
            path1 = os.path.join(cc_root, 'reference', coords[:1])
            self.assertTrue(os.path.isdir(path1))
            path2 = os.path.join(cc_root, 'reference', coords[:1], coords[:2])
            self.assertTrue(os.path.isdir(path2))
            path1 = os.path.join(cc_root, 'full', coords[:1])
            self.assertTrue(os.path.isdir(path1))
            path2 = os.path.join(cc_root, 'full', coords[:1], coords[:2])
            self.assertTrue(os.path.isdir(path2))

        sign_path = os.path.join(cc_root, 'reference',
                                 'A', 'A1', 'A1.001', 'sign1')
        self.assertEqual(cc.get_data_path(
            'sign1', 'reference', 'A1.001'), sign_path)