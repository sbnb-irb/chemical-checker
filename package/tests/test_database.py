import os
import unittest
from chemicalchecker.util import Config
from chemicalchecker.database import set_db_config
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molecule


class TestDatabase(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        # tune the config for sqlite testing
        self.cfg = Config()
        self.cfg.DB.clear()
        self.cfg.DB.file = os.path.join(self.data_dir, 'test.sqlite')
        self.cfg.DB.dialect = 'sqlite'
        set_db_config(self.cfg)
        Dataset._create_table()
        Molecule._create_table()

    def tearDown(self):
        if os.path.exists(self.cfg.DB.file):
            os.remove(self.cfg.DB.file)
        set_db_config(None)

    def test_add_bulk(self):

        res = Molecule.get('test2')
        self.assertIsNone(res)

        Molecule.add_bulk([["test1", "ttt"],
                           ["test2", "zzz"]],
                          on_conflict_do_nothing=False)

        res = Molecule.get('test1')
        self.assertTrue(hasattr(res, 'inchi'))
        self.assertTrue(res.inchi == "ttt")

        res = Molecule.get('test2')
        self.assertTrue(hasattr(res, 'inchi'))
        self.assertTrue(res.inchi == "zzz")

    def test_add(self):

        res = Dataset.get('test2')
        self.assertIsNone(res)

        Dataset.add({"dataset_code": "A1.001",
                     "level": "A", "unknowns": True})

        res = Dataset.get('A1.001')
        self.assertTrue(hasattr(res, 'level'))
        self.assertTrue(res.level == "A")

        res = Dataset.get('A1.001')
        self.assertTrue(hasattr(res, 'unknowns'))
        self.assertTrue(res.unknowns)
