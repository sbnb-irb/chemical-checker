"""Wrapper for Mol2svg tool.

"""
import os
import subprocess
from chemicalchecker.util import logged


@logged
class Mol2svg():
    """Wrapper to run hotnet."""

    def __init__(self):
        self.exec_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "files")

    def mol2svg(self, output_path):
        """Convert molecule to SVG

        Args:
            output_path:Output data path
        """
        # check input

        cmd = "%s/mol2svg --bgcolor=220,218,219 --color=%s/black.conf %s/2d.mol > %s/2d.svg" % (
            self.exec_path, self.exec_path, output_path, output_path)

        # run process
        process = subprocess.Popen(cmd, shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        # stream output as get generated
        for line in iter(process.stdout.readline, ''):
            self.__log.info(line.strip())

        for line in iter(process.stderr.readline, ''):
            self.__log.error(line.strip())
