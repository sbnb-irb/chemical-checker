import os
import unittest
from chemicalchecker.util import Config
from chemicalchecker.database import GeneralProp
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
        self.GeneralProp = GeneralProp
        GeneralProp._create_table()
        self.Dataset = Dataset
        Dataset._create_table()
        self.Molecule = Molecule
        Molecule._create_table()

    def tearDown(self):
        if os.path.exists(self.cfg.DB.file):
            os.remove(self.cfg.DB.file)

    def test_add_bulk(self):

        res = self.GeneralProp.get('test2')
        self.assertIsNone(res)

        self.GeneralProp.add_bulk([
            ["test1", 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131,
             141, 151, 161, 171, "wqwwq"],
            ["test2", 1, 22, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
             17, "wqqw"]], on_conflict_do_nothing=False)

        res = self.GeneralProp.get('test1')
        self.assertTrue(hasattr(res, 'mw'))
        self.assertTrue(res.heavy == 21)

        res = self.GeneralProp.get('test2')
        self.assertTrue(hasattr(res, 'mw'))
        self.assertTrue(res.heavy == 22)

        res = self.Molecule.get('test2')
        self.assertIsNone(res)

        self.Molecule.add_bulk([["test1", "ttt"],
                                ["test2", "zzz"]], on_conflict_do_nothing=False)

        res = self.Molecule.get('test1')
        self.assertTrue(hasattr(res, 'inchi'))
        self.assertTrue(res.inchi == "ttt")

        res = self.Molecule.get('test2')
        self.assertTrue(hasattr(res, 'inchi'))
        self.assertTrue(res.inchi == "zzz")

    def test_add(self):

        res = self.Dataset.get('test2')
        self.assertIsNone(res)

        self.Dataset.add({"dataset_code": "A1.001", "level": "A", "unknowns": True})

        res = self.Dataset.get('A1.001')
        self.assertTrue(hasattr(res, 'level'))
        self.assertTrue(res.level == "A")

        res = self.Dataset.get('A1.001')
        self.assertTrue(hasattr(res, 'unknowns'))
        self.assertTrue(res.unknowns)
