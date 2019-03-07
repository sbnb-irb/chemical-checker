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
import sys
import shutil
import tempfile
from glob import glob
from time import sleep
from ftplib import FTP
from subprocess import call
from six.moves import urllib
from six.moves.urllib import request
from six.moves.urllib.parse import urlparse

from chemicalchecker.util import logged
from chemicalchecker.util import Config
from chemicalchecker.util import psql


@logged
class Downloader():
    """Systematize download and decompression from external repositories."""

    def __init__(self, url, data_path, tmp_dir=None, dbname=None, dbfile=None):
        """Initialize the download object.

        Download from url, unpack in tmp_dir, and copy to data_path.
        We allow wild-char in url only for FTP links. We resolve the link and
        make sure there is only one file matching the regex.

        Args:
            url(str): The external link to a file.
            data_path(str): Final destination for downloaded stuff.
            tmp_dir(str): Temp download path, from config by default.
        """
        self.__log.debug('%s to %s', url, data_path)
        self.data_path = data_path
        self.dbname = dbname
        self.dbfile = dbfile
        if not tmp_dir:
            tmp_dir = Config().PATH.CC_TMP
        self.tmp_dir = tmp_dir
        try:
            self.url = Downloader.validate_url(url)
        except Exception as err:
            self.__log.warning("Cannot validate url: %s", str(err))
            self.url = url

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
                raise RuntimeError('Url not resolving to file.')
            parsed = parsed._replace(path=files[0])
            new_url = parsed.geturl()
            Downloader.__log.debug('Resolved to as: %s', new_url)
        elif parsed.scheme == 'http' or parsed.scheme == 'https':
            if '*' in parsed.path:
                raise RuntimeError('Wild-char `*` in url not accepted.')
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = request.Request(url, headers=headers)
            req.get_method = lambda: 'HEAD'
            try:
                request.urlopen(req, timeout=60)
                new_url = url
            except urllib.error.HTTPError:
                raise RuntimeError('Url not resolving to file.')
        elif parsed.scheme == 'file':
            if os.path.isfile(parsed.path):
                new_url = url
            else:
                raise RuntimeError(
                    'Url not resolving to file: %s.' % parsed.path)
        else:
            raise RuntimeError('Unrecognized URL protocol.')
        return new_url

    def download(self):
        """Perform the download."""
        try:
            import wget
        except ImportError:
            raise ImportError("requires wget " +
                              "http://bitbucket.org/techtonik/python-wget/src")
        try:
            import patoolib
        except ImportError:
            raise ImportError("requires patoolib " +
                              "http://wummel.github.io/patool/")
        # create temp dir
        tmp_dir = tempfile.mkdtemp(
            prefix='tmp_', dir=self.tmp_dir)
        self.__log.debug('temp download dir %s', tmp_dir)
        # set wget user agent

        class MyOpener(urllib.request.FancyURLopener):
            version = 'Mozilla/5.0'
        wget.ulib.urlretrieve = MyOpener().retrieve
        # determine file name
        parsed = urlparse(self.url)
        tmp_file = os.path.join(tmp_dir, wget.detect_filename(self.url))
        # download
        if parsed.scheme == 'file':
            # file can be on local filesystem
            shutil.copy(parsed.path, tmp_dir)
        else:
            # or has to be downloaded, in case try several times
            attempts = 0
            downloaded = False
            while attempts < 5:
                try:
                    wget.download(self.url, tmp_file)
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
        mime, compression = patoolib.util.guess_mime(tmp_file)
        self.__log.debug("MIME %s COMPRESSION %s", mime, compression)
        if mime not in patoolib.ArchiveMimetypes:
            self.__log.debug('no need to uncompress %s, copying', tmp_file)
            shutil.move(tmp_file, tmp_unzip_dir)
        else:
            try:
                patoolib.extract_archive(tmp_file, outdir=tmp_unzip_dir)
            except patoolib.util.PatoolError as err:
                self.__log.error('problem uncompressing %s', tmp_file)
                self.__log.error('error was: %s', str(err))
                raise err
        if self.dbname is not None:
            file_path = os.path.join(tmp_unzip_dir, self.dbfile)
            if '*' in self.dbfile:
                # resolve the path
                paths = glob(file_path)
                if len(paths) > 1:
                    raise Exception("`*` in %s db_file is ambigous.", self)
                file_path = paths[0]
            cmd2run = 'dropdb --if-exists -h ' + Config().DB.host + ' ' + \
                self.dbname + ' && '
            cmd2run += "createdb -h " + Config().DB.host + " " + self.dbname + " && "
            cmd2run += 'pg_restore -h ' + Config().DB.host + '  -d ' + self.dbname + \
                ' ' + file_path

            try:
                self.__log.debug('calling script: ' + cmd2run)
                retcode = call(cmd2run, shell=True)
                self.__log.debug("FINISHED! " + cmd2run +
                                 (" returned code %d" % retcode))

            except OSError as e:
                self.__log.critical("Execution failed: %s" % e)
                sys.exit(1)

            R = psql.qstring("SELECT pg_size_pretty(pg_database_size('" +
                             self.dbname + "'))", Config().DB.database)

            size = R[0][0].split(" ")

            self.__log.debug("Size of the new DB in: " + size[1])

            if size[1] == 'kB':
                raise RuntimeError(
                    'DB created seems to be empty. Please check.')

        # move to final destination
        shutil.move(tmp_unzip_dir, self.data_path)
        self.__log.debug('downloaded to  %s', self.data_path)
        # remove temp dir
        shutil.rmtree(tmp_dir)
        self.__log.debug('removed temp dir %s', tmp_dir)
