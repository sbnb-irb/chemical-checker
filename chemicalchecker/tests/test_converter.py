import os
import unittest
import pytest

from chemicalchecker.util import Converter


class TestConverter(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')

    @pytest.mark.skip(reason="Skip because rdkit is not available")
    def test_smile_to_inchi(self):
        smile = 'COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC4CCN(C)CC4'
        inchikey, inchi = Converter.smile_to_inchi(smile)
        self.assertEqual(inchikey, 'UHTHHESEBZOYNR-UHFFFAOYSA-N')
        self.assertEqual(
            inchi,
            'InChI=1S/C22H24BrFN4O2/c1-28-7-5-14(6-8-28)12-30-21-11-19-16' +
            '(10-20(21)29-2)22(26-13-25-19)27-18-4-3-15(23)9-17(18)24/h3-4' +
            ',9-11,13-14H,5-8,12H2,1-2H3,(H,25,26,27)')
