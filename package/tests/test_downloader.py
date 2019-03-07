import os
import shutil
import pytest
import unittest
import functools

from chemicalchecker.util.download import Downloader


def skip_if_import_exception(function):
    """Assist in skipping tests failing because of missing dependencies."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ImportError as err:
            pytest.skip(str(err))
    return wrapper


class TestDownloader(unittest.TestCase):

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

    @skip_if_import_exception
    def test_download(self):
        url = 'http://www.disgenet.org/static/disgenet_ap1/files/downloads/disease_mappings.tsv.gz'
        downloader = Downloader(url, self.dest_path, self.tmp_path)
        downloader.download()
        self.assertTrue(os.path.isdir(self.dest_path))
        self.assertTrue(os.path.isfile(
            os.path.join(self.dest_path, 'disease_mappings.tsv')))

    @skip_if_import_exception
    def test_wildchar(self):
        url = 'ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz'
        downloader = Downloader(url, self.dest_path, self.tmp_path)
        downloader.download()
        self.assertTrue(os.path.isdir(self.dest_path))
        self.assertTrue(os.path.isfile(
            os.path.join(self.dest_path, 'goa_human.gaf')))

        url = 'ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/*.gz'
        with self.assertRaises(RuntimeError):
            Downloader.validate_url(url)
        url = 'ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/*asdfasdf.gz'
        with self.assertRaises(RuntimeError):
            Downloader.validate_url(url)

    @skip_if_import_exception
    def test_file(self):
        url = 'file://' + os.path.realpath(__file__)
        downloader = Downloader(url, self.dest_path, self.tmp_path)
        downloader.download()
        self.assertTrue(os.path.isdir(self.dest_path))
        self.assertTrue(os.path.isfile(
            os.path.join(self.dest_path,
                         os.path.basename(os.path.realpath(__file__)))))
