import os
import unittest
import pytest

from chemicalchecker.util.parser import Parser, PropCalculator


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
        os.environ["CC_CONFIG"] = os.path.join(
            self.data_dir, 'config.json')

    #@pytest.mark.skip(reason="RDKit is not available on test enviroment")
    def test_fp2d(self):

        chunks = list(PropCalculator.fp2d(self.inchikey_inchi))
        self.assertEqual(len(chunks), 1)
        results = chunks[0]
        self.assertEqual(len(results), 99)

        expected = {'raw': '80,92,264,314,352,389,458,588,624,642,650,694,731,804,807,879,935,940,984,996,997,1019,1066,1088,1145,1199,1257,1328,1366,1380,1404,1456,1470,1487,1488,1503,1645,1750,1754,1775,1800,1820,1847,1873,1911,1920',
                    'inchikey': 'YXKFPFQIDHAWAU-XAZDILKDSA-N'}
        output = next(item for item in results if item[
                      "inchikey"] == "YXKFPFQIDHAWAU-XAZDILKDSA-N")
        self.assertDictEqual(expected, output)

    @pytest.mark.skip(reason="It is too slow")
    def test_fp3d(self):

        chunks = list(PropCalculator.fp3d(
            {k: self.inchikey_inchi[k] for k in list(self.inchikey_inchi)[:2]}))
        self.assertEqual(len(chunks), 1)
        results = chunks[0]
        self.assertEqual(len(results), 2)

        expected = {'raw': '34,48,51,52,81,102,124,131,144,148,155,186,201,216,220,227,251,264,298,299,305,312,315,317,322,332,335,342,356,372,388,401,405,413,418,438,484,494,502,509,528,531,560,562,565,572,574,577,579,591,604,617,631,637,643,644,655,657,665,688,692,704,708,714,723,730,737,763,765,766,778,797,851,877,878,907,912,929,941,942,943,944,955,977,18,48,50,52,79,105,124,130,131,142,144,148,184,186,201,220,232,251,265,275,297,298,308,312,315,322,332,337,372,374,381,388,412,413,424,458,461,484,494,509,550,560,562,565,569,579,590,591,594,617,619,629,641,644,649,655,665,688,692,694,704,717,730,757,763,765,768,773,792,793,797,805,863,877,882,884,897,903,907,912,942,959,972,989,996,4,21,43,48,52,72,86,111,124,126,131,144,148,165,186,201,220,237,242,246,251,255,266,302,311,312,315,322,332,372,388,413,436,465,484,494,526,540,545,560,562,565,579,591,617,630,633,637,644,655,665,673,677,688,692,702,704,730,744,745,757,758,765,766,768,777,792,797,803,843,877,907,912,929,931,942,976,983,1017',
                    'inchikey': 'YXKFPFQIDHAWAU-XAZDILKDSA-N'}
        output = next(item for item in results if item[
                      "inchikey"] == "YXKFPFQIDHAWAU-XAZDILKDSA-N")
        self.assertDictEqual(expected, output)

    #@pytest.mark.skip(reason="RDKit is not available on test enviroment")
    def test_scaffolds(self):

        chunks = list(PropCalculator.scaffolds(self.inchikey_inchi))
        self.assertEqual(len(chunks), 1)
        results = chunks[0]
        self.assertEqual(len(results), 99)

        expected = {'raw': 'c42,c64,c80,c90,c121,c137,c170,c175,c254,c301,c314,c352,c356,c389,c432,c446,c458,c524,c528,c588,c624,c642,c650,c707,c709,c726,c730,c731,c751,c804,c849,c879,c887,c896,c900,c901,c926,c935,c938,c956,c984,c1019,f2,f4,f29,f33,f45,f80,f105,f133,f135,f152,f200,f226,f268,f285,f301,f327,f366,f400,f402,f427,f484,f622,f626,f647,f740,f832,f835,f842,f861,f887,f890,f926,f937,f1015,f1019',
                    'inchikey': 'YXKFPFQIDHAWAU-XAZDILKDSA-N'}
        output = next(item for item in results if item[
                      "inchikey"] == "YXKFPFQIDHAWAU-XAZDILKDSA-N")
        self.assertDictEqual(expected, output)

    #@pytest.mark.skip(reason="RDKit is not available on test enviroment")
    def test_subskeys(self):

        chunks = list(PropCalculator.subskeys(self.inchikey_inchi))
        self.assertEqual(len(chunks), 1)
        results = chunks[0]
        self.assertEqual(len(results), 99)

        expected = {'raw': '19,34,37,54,72,75,77,80,83,85,89,90,91,92,95,96,97,99,100,101,105,106,110,111,117,118,120,121,122,125,127,128,129,131,136,137,138,139,140,142,143,144,145,146,147,148,150,152,153,154,155,156,157,158,159,161,162,163,164,165',
                    'inchikey': 'YXKFPFQIDHAWAU-XAZDILKDSA-N'}
        output = next(item for item in results if item[
                      "inchikey"] == "YXKFPFQIDHAWAU-XAZDILKDSA-N")
        self.assertDictEqual(expected, output)

    @pytest.mark.skip(reason="Qed package not available in gitlab")
    def test_physchem(self):

        chunks = list(PropCalculator.physchem(self.inchikey_inchi))
        self.assertEqual(len(chunks), 1)
        results = chunks[0]
        self.assertEqual(len(results), 99)

        expected = {'heavy': 41, 'inchikey': 'YXKFPFQIDHAWAU-XAZDILKDSA-N', 'mr': 155.31259999999978, 'alerts_qed': 2, 'qed': 0.28857643000451316, 'hba': 5, 'hetero': 8, 'rings': 5, 'psa': 101.38999999999999,
                    'raw': '553.66,41,8,5,2,3,3.541,155.313,5,2,101.390,10,2,4,1,4,0.289', 'mw': 553.6590000000003, 'rotb': 10, 'ro3': 4, 'alogp': 3.5407000000000037, 'ringarom': 3, 'alerts_chembl': 4, 'hbd': 2, 'ro5': 1, 'ringaliph': 2}
        output = next(item for item in results if item[
                      "inchikey"] == "YXKFPFQIDHAWAU-XAZDILKDSA-N")
        self.assertDictEqual(expected, output)
