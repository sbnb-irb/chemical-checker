import os
import unittest
import pytest

from chemicalchecker.util import Parser


class TestConfig(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')

    @pytest.mark.skip(reason="Skip because rdkit is not available")
    def test_bindingdb(self):
        file_path = os.path.join(self.data_dir, 'BindingDB_All.tsv')
        self.assertTrue(os.path.isfile(file_path))
        chunks = list(Parser.bindingdb(file_path, 'bindingdb'))
        self.assertEqual(len(chunks), 1)
        results = chunks[0]
        self.assertEqual(len(results), 99)

        expected = {'inchi': 'InChI=1S/C22H24BrFN4O2/c1-28-7-5-14(6-8-28)12-30-21-11-19-16(10-20(21)29-2)22(26-13-25-19)27-18-4-3-15(23)9-17(18)24/h3-4,9-11,13-14H,5-8,12H2,1-2H3,(H,25,26,27)',
                    'inchikey': 'UHTHHESEBZOYNR-UHFFFAOYSA-N',
                    'smile': 'COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC1CCN(C)CC1',
                    'src_id': 'UHTHHESEBZOYNR-UHFFFAOYSA-N',
                    'src_name': 'bindingdb'}
        self.assertDictEqual(expected, results[0])

    @pytest.mark.skip(reason="Skip because rdkit is not available")
    def test_chebi(self):
        file_path = os.path.join(self.data_dir, 'ChEBI_lite_3star.sdf')
        self.assertTrue(os.path.isfile(file_path))
        chunks = list(Parser.chebi(file_path, 'chebi'))
        self.assertEqual(len(chunks), 1)
        results = chunks[0]
        self.assertEqual(len(results), 2)

        expected = {'inchi': 'InChI=1S/C15H14O6/c16-8-4-11(18)9-6-13(20)15(21-14(9)5-8)7-1-2-10(17)12(19)3-7/h1-5,13,15-20H,6H2/t13-,15-/m1/s1',
                    'inchikey': 'PFTAWBLQPZVEMU-UKRRQHHQSA-N',
                    'smile': 'Oc1cc(O)c2c(c1)O[C@H](c1ccc(O)c(O)c1)[C@H](O)C2',
                    'src_id': 'CHEBI:90',
                    'src_name': 'chebi'}
        self.assertDictEqual(expected, results[0])
