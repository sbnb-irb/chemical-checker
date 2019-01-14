import os
import unittest
import pytest
import shutil

from chemicalchecker.util import Parser


class TestParser(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')

    #@pytest.mark.skip(reason="RDKit is not available on test enviroment")
    def test_bindingdb(self):
        file_path = os.path.join(self.data_dir, 'BindingDB_All.tsv')
        self.assertTrue(os.path.isfile(file_path))
        chunks = list(Parser.bindingdb([file_path], 'bindingdb'))
        self.assertEqual(len(chunks), 1)
        results = chunks[0]
        self.assertEqual(len(results), 99)

        expected = {'inchi': 'InChI=1S/C22H24BrFN4O2/c1-28-7-5-14(6-8-28)12-30-21-11-19-16(10-20(21)29-2)22(26-13-25-19)27-18-4-3-15(23)9-17(18)24/h3-4,9-11,13-14H,5-8,12H2,1-2H3,(H,25,26,27)',
                    'inchikey': 'UHTHHESEBZOYNR-UHFFFAOYSA-N',
                    'smiles': 'COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC1CCN(C)CC1',
                    'src_id': 'UHTHHESEBZOYNR-UHFFFAOYSA-N',
                    'molrepo_name': 'bindingdb'}
        self.assertDictEqual(expected, results[0])

    #@pytest.mark.skip(reason="RDKit is not available on test enviroment")
    def test_chebi(self):
        file_path = os.path.join(self.data_dir, 'ChEBI_lite_3star.sdf')
        self.assertTrue(os.path.isfile(file_path))
        chunks = list(Parser.chebi([file_path], 'chebi'))
        self.assertEqual(len(chunks), 1)
        results = chunks[0]
        self.assertEqual(len(results), 2)

        expected = {'inchi': 'InChI=1S/C15H14O6/c16-8-4-11(18)9-6-13(20)15(21-14(9)5-8)7-1-2-10(17)12(19)3-7/h1-5,13,15-20H,6H2/t13-,15-/m1/s1',
                    'inchikey': 'PFTAWBLQPZVEMU-UKRRQHHQSA-N',
                    'smiles': 'Oc1cc(O)c2c(c1)O[C@H](c1ccc(O)c(O)c1)[C@H](O)C2',
                    'src_id': 'CHEBI:90',
                    'molrepo_name': 'chebi'}
        self.assertDictEqual(expected, results[0])

    #@pytest.mark.skip(reason="RDKit is not available on test enviroment")
    def test_ctd(self):
        file_path = os.path.join(self.data_dir, 'CTD_chemicals_diseases.tsv')
        self.assertTrue(os.path.isfile(file_path))
        chunks = list(Parser.ctd([file_path], 'ctd'))
        self.assertEqual(len(chunks), 1)
        results = chunks[0]
        self.assertEqual(len(results), 2)

        expected = {'inchi': 'InChI=1S/C26H20N2O/c29-25-21-5-1-3-7-23(21)26(17-19-9-13-27-14-10-19,18-20-11-15-28-16-12-20)24-8-4-2-6-22(24)25/h1-16H,17-18H2',
                    'inchikey': 'KHJFBUUFMUBONL-UHFFFAOYSA-N',
                    'molrepo_name': 'ctd',
                    'smiles': 'C1=CC=C2C(=C1)C(=O)C3=CC=CC=C3C2(CC4=CC=NC=C4)CC5=CC=NC=C5',
                    'src_id': 'C112297'}
        self.assertDictEqual(expected, results[0])

    #@pytest.mark.skip(reason="RDKit is not available on test enviroment")
    def test_drugbank(self):
        file_path = os.path.join(self.data_dir, 'drugbank.xml')
        self.assertTrue(os.path.isfile(file_path))
        chunks = list(Parser.drugbank([file_path], 'drugbank'))
        self.assertEqual(len(chunks), 1)
        results = chunks[0]
        self.assertEqual(len(results), 1)

        expected = {'inchi': 'InChI=1S/C98H138N24O33/c1-5-52(4)82(96(153)122-39-15-23-70(122)92(149)114-60(30-34-79(134)135)85(142)111-59(29-33-78(132)133)86(143)116-64(43-55-24-26-56(123)27-25-55)89(146)118-67(97(154)155)40-51(2)3)119-87(144)61(31-35-80(136)137)112-84(141)58(28-32-77(130)131)113-88(145)63(42-54-18-10-7-11-19-54)117-90(147)66(45-81(138)139)110-76(129)50-107-83(140)65(44-71(100)124)109-75(128)49-106-73(126)47-104-72(125)46-105-74(127)48-108-91(148)68-21-13-38-121(68)95(152)62(20-12-36-103-98(101)102)115-93(150)69-22-14-37-120(69)94(151)57(99)41-53-16-8-6-9-17-53/h6-11,16-19,24-27,51-52,57-70,82,123H,5,12-15,20-23,28-50,99H2,1-4H3,(H2,100,124)(H,104,125)(H,105,127)(H,106,126)(H,107,140)(H,108,148)(H,109,128)(H,110,129)(H,111,142)(H,112,141)(H,113,145)(H,114,149)(H,115,150)(H,116,143)(H,117,147)(H,118,146)(H,119,144)(H,130,131)(H,132,133)(H,134,135)(H,136,137)(H,138,139)(H,154,155)(H4,101,102,103)/t52-,57+,58-,59-,60-,61-,62-,63-,64-,65-,66-,67-,68-,69-,70-,82-/m0/s1',
                    'inchikey': 'OIRCOABEOLEUMC-GEJPAHFPSA-N',
                    'molrepo_name': 'drugbank',
                    'smiles': 'CC[C@H](C)[C@H](NC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)[C@H](CC(O)=O)NC(=O)CNC(=O)[C@H](CC(N)=O)NC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(N)=N)NC(=O)[C@@H]1CCCN1C(=O)[C@H](N)CC1=CC=CC=C1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(O)=O)C(=O)N[C@@H](CCC(O)=O)C(=O)N[C@@H](CC1=CC=C(O)C=C1)C(=O)N[C@@H](CC(C)C)C(O)=O',
                    'src_id': 'DB00006'}
        self.assertDictEqual(expected, results[0])

    #@pytest.mark.skip(reason="RDKit is not available on test enviroment")
    def test_kegg(self):
        file_path = os.path.join(self.data_dir, 'kegg.br')
        self.assertTrue(os.path.isfile(file_path))
        mol_dir = os.path.join(self.data_dir, 'mols')
        self.assertFalse(os.path.isdir(mol_dir))
        chunks = list(Parser.kegg([file_path], 'kegg'))
        self.assertTrue(os.path.isdir(mol_dir))
        self.assertEqual(len(chunks), 1)
        results = chunks[0]
        self.assertEqual(len(results), 1)

        expected = {'inchi': '',
                    'inchikey': '',
                    'molrepo_name': 'kegg',
                    'smiles': '[Na+].[F-]',
                    'src_id': 'D00943'}

        self.assertDictEqual(expected, results[0])
        shutil.rmtree(mol_dir)

    #@pytest.mark.skip(reason="RDKit is not available on test enviroment")
    def test_lincs(self):
        file_path1 = os.path.join(self.data_dir, 'lincs_GSE70138.txt')
        self.assertTrue(os.path.isfile(file_path1))
        file_path2 = os.path.join(self.data_dir, 'lincs_GSE92742.txt')
        self.assertTrue(os.path.isfile(file_path2))
        chunks = list(Parser.lincs([file_path1, file_path2], 'lincs'))
        self.assertEqual(len(chunks), 1)
        results = chunks[0]
        self.assertEqual(len(results), 9)

        expected = {'inchi': 'InChI=1S/C20H25ClN2O/c1-3-22(4-2)13-7-8-14-23-17-9-5-6-10-19(17)24-20-12-11-16(21)15-18(20)23/h5-6,9-12,15H,3-4,7-8,13-14H2,1-2H3',
                    'inchikey': 'GYBXAGDWMCJZJK-UHFFFAOYSA-N',
                    'molrepo_name': 'lincs',
                    'smiles': 'CCN(CC)CCCCN1c2ccccc2Oc2ccc(Cl)cc12',
                    'src_id': 'BRD-K70792160'}
        self.assertDictEqual(expected, results[0])

    #@pytest.mark.skip(reason="RDKit is not available on test enviroment")
    def test_smpdb(self):
        file_path = os.path.join(self.data_dir, 'smpdb_structures')
        chunks = list(Parser.smpdb([file_path], 'smpdb'))
        self.assertEqual(len(chunks), 1)
        results = chunks[0]
        self.assertEqual(len(results), 2)

        expected = {'inchi': 'InChI=1S/C71H138O17P2/c1-61(2)47-39-31-23-17-14-12-10-9-11-13-15-19-28-37-45-53-70(75)87-66(58-82-69(74)52-44-36-29-21-25-33-41-49-63(5)6)59-85-89(77,78)83-55-65(72)56-84-90(79,80)86-60-67(88-71(76)54-46-38-30-22-26-34-42-50-64(7)8)57-81-68(73)51-43-35-27-20-16-18-24-32-40-48-62(3)4/h61-67,72H,9-60H2,1-8H3,(H,77,78)(H,79,80)/t65-,66+,67+/m0/s1',
                    'smiles': 'O[C@H](COP(=O)(OC[C@H](OC(=O)CCCCCCCCCC(C)C)COC(=O)CCCCCCCCCCCC(C)C)O)COP(=O)(OC[C@H](OC(=O)CCCCCCCCCCCCCCCCCC(C)C)COC(=O)CCCCCCCCCC(C)C)O',
                    'inchikey': 'CBGQZUCVOHIHAR-OHKZLATASA-N',
                    'src_id': "1'-[1-11-methyldodecanoyl,2-19-methyleicosanoyl-sn-glycero-3-phospho],3'-[1-13-methyltetradecanoyl,2-11-methyldodecanoyl-sn-glycero-3-phospho]-sn-glycerol CL(i-13:0/i-21:0/i-15:0/i-13:0)",
                    'molrepo_name': 'smpdb'}
        self.assertDictEqual(expected, results[0])
