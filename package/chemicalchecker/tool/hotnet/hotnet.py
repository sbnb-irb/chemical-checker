"""Wrapper for HOTNET subnetworks search.

   https://github.com/raphael-group/hotnet2
"""
import os
import subprocess
from chemicalchecker.util import logged


@logged
class Hotnet():
    """Wrapper to run hotnet."""

    def __init__(self, cpu=1):
        self.exec_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "hotnet-src")
        self.cpu = cpu

    def choose_beta(self, i, o, **kwargs):
        """Call external python with given parameters.

        Args:
            i:Input Edge list filename
            o:Output data path
        """
        # check input
        if not os.path.isfile(i):
            raise Exception("Input file not found.")

        executable = os.path.join(self.exec_path, "choose_beta.py")

        # prepare arguments
        args = [
            "-i%s" % i,
            "-o%s" % o,
            "-c%s" % self.cpu
        ]

        # log command
        self.__log.info(executable + ' '.join(args))

        # run process
        process = subprocess.Popen(["python", executable] + args,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        # stream output as get generated
        for line in iter(process.stdout.readline, ''):
            self.__log.info(line.strip())

        for line in iter(process.stderr.readline, ''):
            self.__log.error(line.strip())

    def create_similarity_matrix(self, i, o, **kwargs):
        """Call external python with given parameters.

        Args:
            i:Input Edge list filename
            o:Output data path (default:'similarity_matrix.h5')
            b:Beta (Restart probability). Default is 128 (default:0.0)

        """
        # check input
        if not os.path.isfile(i):
            raise Exception("Input file not found.")

        # get arguments or default values
        b = kwargs.get("b", 0.0)

        executable = os.path.join(
            self.exec_path, "create_similarity_matrix.py")

        # prepare arguments
        args = [
            "-i%s" % i,
            "-o%s" % o,
            "-b%.2f" % b,
            "-c%s" % self.cpu
        ]

        # log command
        self.__log.info(' '.join(args))

        # run process
        process = subprocess.Popen(["python", executable] + args,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        # stream output as get generated
        for line in iter(process.stdout.readline, ''):
            self.__log.info(line.strip())

        for line in iter(process.stderr.readline, ''):
            self.__log.error(line.strip())
