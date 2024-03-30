import os
import shutil
from chemicalchecker.util.download import Downloader

dest_path = './dest'
tmp_path = './tmp'

if( os.path.isdir(dest_path) ):
    shutil.rmtree(dest_path)
if( os.path.isdir(tmp_path) ):
    shutil.rmtree(tmp_path)

os.mkdir(dest_path)
os.mkdir(tmp_path)

url = 'http://www.disgenet.org/static/disgenet_ap1/files/downloads/disease_mappings.tsv.gz'
downloader = Downloader(url, dest_path, tmp_path)
downloader.download()
