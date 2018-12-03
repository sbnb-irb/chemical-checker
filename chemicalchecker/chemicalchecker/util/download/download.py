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
from time import sleep
from ftplib import FTP
from pyunpack import Archive, PatoolError
from six.moves import urllib
from six.moves.urllib.parse import urlparse
from six.moves.urllib import request
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
        self.url = Downloader.validate_url(url)

    @staticmethod
    def validate_url(url):
        """Validate and check if url is working."""
        # validating url
        parsed = urlparse(url)
        if parsed.scheme == 'ftp':
            f = FTP(parsed.netloc)
            f.login()
            files = f.nlst(parsed.path)
            if len(files) > 1:
                raise RuntimeError('Url resolving to multiple files.')
            if len(files) == 0:
                raise RuntimeError('Url resolving to no file.')
            parsed = parsed._replace(path=files[0])
            new_url = parsed.geturl()
            Downloader.__log.debug('Resolved to as: %s', new_url)
        elif parsed.scheme == 'http' or parsed.scheme == 'https':
            if '*' in parsed.path:
                raise RuntimeError('Wild-char `*` in url not accepted.')
            req = request.Request(url)
            req.get_method = lambda: 'HEAD'
            try:
                request.urlopen(req, timeout=60)
                new_url = url
            except urllib.error.HTTPError:
                raise RuntimeError('Url resolving to no file.')
        elif parsed.scheme == 'file':
            if os.path.isfile(parsed.path):
                new_url = url
            else:
                raise RuntimeError('Url resolving to no file.')
        else:
            raise RuntimeError('Unrecognized URL protocol.')
        return new_url

    def download(self):
        """Perform the download."""
        # create temp dir
        tmp_dir = tempfile.mkdtemp(
            prefix='tmp_', dir=self.tmp_dir)
        self.__log.debug('temp download dir %s', tmp_dir)
        # download
        tmp_file = os.path.join(tmp_dir, wget.detect_filename(self.url))
        if self.url.startswith('file'):
            # file can be on local filesystem
            parsed = urlparse(self.url)
            shutil.copy(parsed.path, tmp_dir)
        else:
            # or has to be downloaded, in case try several times
            attempts = 0
            downloaded = False
            while attempts < 5:
                try:
                    wget.download(self.url, tmp_dir)
                    downloaded = True
                    break
                except Exception as err:
                    attempts += 1
                    self.__log.warning('Attempt failed: %s', str(err))
                    request.urlcleanup()
                    sleep(5)
            if not downloaded:
                raise Exception('All attempts to download failed.')
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
