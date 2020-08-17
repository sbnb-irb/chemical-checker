import os
import unittest
from chemicalchecker.util import Config
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molecule


class TestDatabase(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        # set the enviroment path for the config to be found
        os.environ["CC_CONFIG"] = os.path.join(
            self.data_dir, 'database_config.json')
        self.cfg = Config()
        self.Dataset = Dataset
        Dataset._create_table()
        self.Molecule = Molecule
        Molecule._create_table()

    def tearDown(self):
        if os.path.exists(self.cfg.DB.file):
            os.remove(self.cfg.DB.file)

    def test_add_bulk(self):

        res = self.Molecule.get('test2')
        self.assertIsNone(res)

        self.Molecule.add_bulk([["test1", "ttt"],
                                ["test2", "zzz"]],
                               on_conflict_do_nothing=False)

        res = self.Molecule.get('test1')
        self.assertTrue(hasattr(res, 'inchi'))
        self.assertTrue(res.inchi == "ttt")

        res = self.Molecule.get('test2')
        self.assertTrue(hasattr(res, 'inchi'))
        self.assertTrue(res.inchi == "zzz")

    def test_add(self):

        res = self.Dataset.get('test2')
        self.assertIsNone(res)

        self.Dataset.add({"dataset_code": "A1.001",
                          "level": "A", "unknowns": True})

        res = self.Dataset.get('A1.001')
        self.assertTrue(hasattr(res, 'level'))
        self.assertTrue(res.level == "A")

        res = self.Dataset.get('A1.001')
        self.assertTrue(hasattr(res, 'unknowns'))
        self.assertTrue(res.unknowns)
