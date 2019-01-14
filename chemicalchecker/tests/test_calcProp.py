import os
import unittest
import pytest

from chemicalchecker.util import Parser, PropCalculator


class TestPropCalculator(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        file_path = os.path.join(self.data_dir, 'BindingDB_All.tsv')
        chunks = list(Parser.bindingdb([file_path], 'bindingdb'))
        results = chunks[0]
        self.inchikey_inchi = {}
        for ele in results:
            self.inchikey_inchi[str(ele["inchikey"])] = str(ele["inchi"])

    #@pytest.mark.skip(reason="RDKit is not available on test enviroment")
    def test_fp2d(self):

        chunks = list(PropCalculator.fp2d(self.inchikey_inchi))
        self.assertEqual(len(chunks), 1)
        results = chunks[0]
        self.assertEqual(len(results), 99)

        expected = {'raw': '80,92,264,314,352,389,458,588,624,642,650,694,731,804,807,879,935,940,984,996,997,1019,1066,1088,1145,1199,1257,1328,1366,1380,1404,1456,1470,1487,1488,1503,1645,1750,1754,1775,1800,1820,1847,1873,1911,1920',
                    'inchikey': 'YXKFPFQIDHAWAU-XAZDILKDSA-N'}
        self.assertDictEqual(expected, results[0])

    @pytest.mark.skip(reason="RDKit is not available on test enviroment")
    def test_scaffolds(self):

        chunks = list(PropCalculator.scaffolds(self.inchikey_inchi))
        self.assertEqual(len(chunks), 1)
        results = chunks[0]
        self.assertEqual(len(results), 99)

        expected = {'raw': 'c42,c64,c80,c90,c121,c137,c170,c175,c254,c301,c314,c352,c356,c389,c432,c446,c458,c524,c528,c588,c624,c642,c650,c707,c709,c726,c730,c731,c751,c804,c849,c879,c887,c896,c900,c901,c926,c935,c938,c956,c984,c1019,f2,f4,f29,f33,f45,f80,f105,f133,f135,f152,f200,f226,f268,f285,f301,f327,f366,f400,f402,f427,f484,f622,f626,f647,f740,f832,f835,f842,f861,f887,f890,f926,f937,f1015,f1019',
                    'inchikey': 'YXKFPFQIDHAWAU-XAZDILKDSA-N'}
        self.assertDictEqual(expected, results[0])

    @pytest.mark.skip(reason="RDKit is not available on test enviroment")
    def test_subskeys(self):

        chunks = list(PropCalculator.subskeys(self.inchikey_inchi))
        self.assertEqual(len(chunks), 1)
        results = chunks[0]
        self.assertEqual(len(results), 99)

        expected = {'raw': '19,34,37,54,72,75,77,80,83,85,89,90,91,92,95,96,97,99,100,101,105,106,110,111,117,118,120,121,122,125,127,128,129,131,136,137,138,139,140,142,143,144,145,146,147,148,150,152,153,154,155,156,157,158,159,161,162,163,164,165',
                    'inchikey': 'YXKFPFQIDHAWAU-XAZDILKDSA-N'}
        self.assertDictEqual(expected, results[0])
