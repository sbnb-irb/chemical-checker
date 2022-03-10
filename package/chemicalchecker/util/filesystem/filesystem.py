import os
from chemicalchecker.util import logged

@logged
class FileSystem():
    """FileSystem class"""

    def __init__(self):
        self.cwd = os.getcwd()

    def check_dir_existance_create(dir_path, additional_path=None):
        """Args:
            dir_path(str): root path
            additional_path(list) : list of strings including additional
                                    path parts to append to the root path
        """
        path = dir_path
        if additional_path:
            for element in additional_path:
                path = os.path.join(path, element)
        if not os.path.isdir(path):
            original_umask = os.umask(0)
            os.makedirs(path, 0o775)
            os.umask(original_umask)
        return path

    def check_file_existance_create(file_path):
        """
            This method create an empty file if it doesn't exist already
        """
        if not os.path.isfile(file_path):
            with open(file_path, 'w'):
                pass
