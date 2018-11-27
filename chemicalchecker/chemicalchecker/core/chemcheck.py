"""This class simplify and standardize access to the Chemical Checker.
Main task of this lass are:
1. Check and enforce the directory structure.
2. Serve signatures to users.
"""

import os
import itertools

from .data import DataFactory
from chemicalchecker.util import logged


@logged
class ChemicalChecker():
    """Explore the Chemical Checker."""

    def __init__(self, cc_root):
        """Initialize the Chemical Checker.

        If the CC_ROOT directory is empty a skeleton of CC is initialized.

        Args:
            cc_root(str): The Chemical Checker root directory.
        """
        self.cc_root = cc_root
        self.__log.debug("ChemicalChecker with root: %s", cc_root)
        if not os.path.isdir(cc_root):
            self.__log.warning("Empty root directory, creating new one")
            for coord in self.coordinates:
                new_dir = os.path.join(cc_root, coord[:1], coord[:2])
                self.__log.debug("Creating %s", new_dir)
                os.makedirs(new_dir)

    @property
    def coordinates(self):
        """Iterator on Chemical Checker coordinates."""
        for name, code in itertools.product("ABCDE", "12345"):
            yield name + code

    @property
    def datasets(self, exemplary_only=False):
        """Iterator on Chemical Checker datasets.

        TODO This should be an iterator on the dataset db table.
        """
        for name, code in itertools.product("ABCDE", "12345"):
            yield name + code + ".001"

    def get_data_path(self, datatype, dataset):
        """Return the path to signature file for the given dataset.

        This should be the only place where we define the directory structure.
        The signature type directly map to a HDF5 file.
        """
        filename = '{}.h5'.format(datatype)
        data_path = os.path.join(self.cc_root, dataset[:1],
                                 dataset[:2], dataset, filename)
        self.__log.debug("signature path: %s", data_path)
        return data_path

    def get_data(self, datatype, dataset):
        """Return the full signature for the given dataset."""
        data_path = self.get_data_path(datatype, dataset)
        # initialize a data object factory feeding the type and the path
        data_factory = DataFactory()
        # the factory will spit the data in the right class
        data = data_factory.make_data(datatype, data_path)
        return data
