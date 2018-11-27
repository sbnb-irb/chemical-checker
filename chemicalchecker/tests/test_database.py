import os
import unittest

from chemicalchecker.util import Config


class TestDatabase(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        conffile = os.path.join(self.data_dir, 'database_config.json')
        Config(conffile)
        from chemicalchecker.database import GeneralProp
        self.GeneralProp = GeneralProp
        GeneralProp.create_table()

    def tearDown(self):
        test_dir = os.path.dirname(os.path.realpath(__file__))
        testfile = os.path.join(test_dir, 'test.sqlite')
        if os.path.exists(testfile):
            os.remove(testfile)

        if os.path.exists(testfile):
            print testfile

    def test_add_bulk(self):

        test_dir = os.path.dirname(os.path.realpath(__file__))
        testfile = os.path.join(test_dir, 'test.sqlite')
        if os.path.exists(testfile):
            print testfile

        res = self.GeneralProp.get('test2')
        self.assertIsNone(res)

        self.GeneralProp.add_bulk([["test1", 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171],
                                   ["test2", 1, 22, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]])

        res = self.GeneralProp.get('test1')
        self.assertTrue(hasattr(res, 'mw'))
        self.assertTrue(res.heavy == 21)

        res = self.GeneralProp.get('test2')
        self.assertTrue(hasattr(res, 'mw'))
        self.assertTrue(res.heavy == 22)
