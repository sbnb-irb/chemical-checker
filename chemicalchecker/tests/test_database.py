import os
import unittest
from chemicalchecker.util import Config
from chemicalchecker.database import GeneralProp
from chemicalchecker.database import Dataset
from chemicalchecker.database import Pubchem
from chemicalchecker.database import Libraries
from chemicalchecker.database import Structure


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
        self.Libraries = Libraries
        Libraries._create_table()
        self.Pubchem = Pubchem
        Pubchem._create_table()
        self.Structure = Structure
        Structure._create_table()

    def tearDown(self):
        if os.path.exists(self.cfg.DB.file):
            os.remove(self.cfg.DB.file)

    def test_add_bulk(self):

        res = self.GeneralProp.get('test2')
        self.assertIsNone(res)

        self.GeneralProp.add_bulk([
            ["test1", 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131,
             141, 151, 161, 171],
            ["test2", 1, 22, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
             17]])

        res = self.GeneralProp.get('test1')
        self.assertTrue(hasattr(res, 'mw'))
        self.assertTrue(res.heavy == 21)

        res = self.GeneralProp.get('test2')
        self.assertTrue(hasattr(res, 'mw'))
        self.assertTrue(res.heavy == 22)

        res = self.Pubchem.get()
        self.assertIsNone(res)

        self.Pubchem.add_bulk([[1, "t1", "t2", "t3", "t4"],
                               [2, "z1", "z2", "z3", "z4"]])

        res = self.Pubchem.get(cid=1)
        self.assertTrue(hasattr(res[0], 'name'))
        self.assertTrue(res[0].name == "t3")

        res = self.Pubchem.get(cid=2)
        self.assertTrue(hasattr(res[0], 'name'))
        self.assertTrue(res[0].name == "z3")

        res = self.Structure.get('test2')
        self.assertIsNone(res)

        self.Structure.add_bulk([["test1", "ttt"],
                                 ["test2", "zzz"]])

        res = self.Structure.get('test1')
        self.assertTrue(hasattr(res, 'inchi'))
        self.assertTrue(res.inchi == "ttt")

        res = self.Structure.get('test2')
        self.assertTrue(hasattr(res, 'inchi'))
        self.assertTrue(res.inchi == "zzz")

    def test_add(self):

        res = self.Dataset.get('test2')
        self.assertEqual([], res)

        self.Dataset.add({"code": "A1.001", "level": "A", "unknowns": True})

        res = self.Dataset.get('A1.001')
        res = res[0]
        self.assertTrue(hasattr(res, 'level'))
        self.assertTrue(res.level == "A")

        res = self.Dataset.get('A1.001')
        res = res[0]
        self.assertTrue(hasattr(res, 'unknowns'))
        self.assertTrue(res.unknowns)

        res = self.Libraries.get('test1')
        self.assertIsNone(res)

        self.Libraries.add({"lib": "test1", "files": "A", "name": "test",
                            "description": "A", "urls": "True", "rank": 6})

        res = self.Libraries.get('test1')
        self.assertTrue(hasattr(res, 'files'))
        self.assertTrue(res.files == "A")

        res = self.Libraries.get('test1')
        self.assertTrue(hasattr(res, 'rank'))
        self.assertTrue(res.rank == 6)
