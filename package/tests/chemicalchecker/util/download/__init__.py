"""Download raw data files.

Systematize download and decompression from external repositories.
Basically every :mod:`~chemicalchecker.database.dataset` require one or more
files to be downloaded from external repositories.
This class performs the following:

   1. Create a temporary directory where to handle the download.
   2. Check that the url is reachable.
   3. Download the file.
   4. Unzip, tar or whatever.
   5. Copy to the local data path.
"""
from .download import Downloader
