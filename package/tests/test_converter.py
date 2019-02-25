import os
import unittest
import pytest

from chemicalchecker.util import Converter


class TestConverter(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        os.environ["CC_CONFIG"] = os.path.join(
            self.data_dir, 'config.json')

    #@pytest.mark.skip(reason="RDKit is not available on test enviroment")
    def test_smiles_to_inchi(self):
        smile = 'COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC4CCN(C)CC4'
        inchikey, inchi = Converter.smiles_to_inchi(smile)
        self.assertEqual(inchikey, 'UHTHHESEBZOYNR-UHFFFAOYSA-N')
        self.assertEqual(
            inchi,
            'InChI=1S/C22H24BrFN4O2/c1-28-7-5-14(6-8-28)12-30-21-11-19-16' +
            '(10-20(21)29-2)22(26-13-25-19)27-18-4-3-15(23)9-17(18)24/h3-4' +
            ',9-11,13-14H,5-8,12H2,1-2H3,(H,25,26,27)')

    def test_ctd_to_smiles(self):
        ctdid = 'C112297'
        smiles = Converter.ctd_to_smiles(ctdid)
        self.assertEqual(
            smiles, 'C1=CC=C2C(=C1)C(=O)C3=CC=CC=C3C2(CC4=CC=NC=C4)CC5=CC=NC=C5')
        with self.assertRaises(Exception):
            ctdid = 'C046983'
            smiles = Converter.ctd_to_smiles(ctdid)

    def test_chemical_name_to_smiles(self):
        ctdid = 'oxygen'
        smiles = Converter.chemical_name_to_smiles(ctdid)
        self.assertEqual(
            smiles, 'O')
        with self.assertRaises(Exception):
            ctdid = 'qwerqwerqwerqer'
            smiles = Converter.ctd_to_smiles(ctdid)
