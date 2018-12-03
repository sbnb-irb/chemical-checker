import os
import shutil
import unittest

from chemicalchecker.util import Downloader


class TestConfig(unittest.TestCase):

    def setUp(self):
        # path for test data
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(test_dir, 'data')
        os.environ["CC_CONFIG"] = os.path.join(
            self.data_dir, 'download_config.json')
        self.tmp_path = os.path.join(self.data_dir, 'test_download_tmp')
        os.makedirs(self.tmp_path)
        self.dest_path = os.path.join(self.data_dir, 'test_download_dest')

    def tearDown(self):
        shutil.rmtree(self.tmp_path)
        shutil.rmtree(self.dest_path)

    def test_download(self):
        url = 'http://www.disgenet.org/ds/DisGeNET/' + \
            'results/disease_mappings.tsv.gz'
        downloader = Downloader(url, self.dest_path, self.tmp_path)
        downloader.download()
        self.assertTrue(os.path.isdir(self.dest_path))
        self.assertTrue(os.path.isfile(
            os.path.join(self.dest_path, 'disease_mappings.tsv')))
