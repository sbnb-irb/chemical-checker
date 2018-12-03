"""Utility for downloading raw data files.

Basically every Chemical Checker dataset require one or more files to be
downloaded from external repositories. This class performs the following:

1. create a temporary directory where to handle the download
2. check that the url is reachable
3. download the file
4. unzip, tar or whatever
5. copy to the local data path
"""
import os
import wget
import shutil
import tempfile
from ftplib import FTP
from pyunpack import Archive, PatoolError
from six.moves.urllib.parse import urlparse

from chemicalchecker.util import logged


@logged
class Downloader():
    """Systematize download and decompression from external repositories."""

    def __init__(self, url, data_path, tmp_dir):
        """Initialize the download object.

        Download from url, unpack in tmp_dir, and copy to data_path.
        We allow wild-char in url only for FTP links. We resolve the link and
        make sure there is only one file matching the regex.

        Args:
            url(str): The external link to a file.
            data_path(str): Final destination for downloaded stuff.
        """
        self.__log.debug('%s to %s', url, data_path)
        self.data_path = data_path
        self.tmp_dir = tmp_dir
        # validating url
        parsed = urlparse(url)
        if '*' in parsed.path:
            if parsed.scheme == 'ftp':
                self.__log.debug('Wild-char `*` in FTP url, resolving.')
                f = FTP(parsed.netloc)
                f.login()
                files = f.nlst(parsed.path)
                if len(files) > 1:
                    raise RuntimeError('Url resulting in multiple files.')
                if len(files) == 0:
                    raise RuntimeError('Url resulting no file.')
                parsed = parsed._replace(path=files[0])
                self.url = parsed.geturl()
                self.__log.debug('Resolved to as: %s', self.url)
            else:
                raise RuntimeError('Wild-char `*` in url not accepted.')
        else:
            self.url = url

    def download(self):
        """Perform the download."""
        # create temp dir
        tmp_dir = tempfile.mkdtemp(
            prefix='tmp_', dir=self.tmp_dir)
        self.__log.debug('temp download dir %s', tmp_dir)
        # download
        tmp_file = os.path.join(tmp_dir, wget.detect_filename(self.url))
        wget.download(self.url, tmp_dir)
        # unzip
        tmp_unzip_dir = os.path.join(tmp_dir, 'unzip')
        os.mkdir(tmp_unzip_dir)
        self.__log.debug('temp unzip dir %s', tmp_unzip_dir)
        # not a clear way to check if file is compressed, just try
        try:
            Archive(tmp_file).extractall(tmp_unzip_dir)
        except PatoolError as err:
            self.__log.warning('problem uncompressing %s', tmp_file)
            self.__log.warning('error was: %s', str(err))
            shutil.move(tmp_file, tmp_unzip_dir)
        # move to final destination
        shutil.move(tmp_unzip_dir, self.data_path)
        self.__log.debug('downloaded to  %s', self.data_path)
        # remove temp dir
        shutil.rmtree(tmp_dir)
        self.__log.debug('removed temp dir %s', tmp_dir)
