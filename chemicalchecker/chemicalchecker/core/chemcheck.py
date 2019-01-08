"""This class simplify and standardize access to the Chemical Checker.

Main tasks of this class are:

1. Check and enforce the directory structure.
2. Serve signatures to users or pipelines.
"""

import os
import itertools

from .data import DataFactory
from chemicalchecker.util import logged
from chemicalchecker.database import Dataset


@logged
class ChemicalChecker():
    """Explore the Chemical Checker."""

    def __init__(self, cc_root):
        """Initialize the Chemical Checker.

        If the CC_ROOT directory is empty a skeleton of CC is initialized.

        Args:
            cc_root(str): The Chemical Checker root directory. It's version
                dependendent.
        """
        self.cc_root = cc_root
        self.basic_molsets = ['reference', 'full']
        self.__log.debug("ChemicalChecker with root: %s", cc_root)
        if not os.path.isdir(cc_root):
            self.__log.warning("Empty root directory, creating dataset dirs")
            for molset in self.basic_molsets:
                for dataset in self.datasets:
                    new_dir = os.path.join(
                        cc_root, molset, dataset[:1], dataset[:2], dataset)
                    self.__log.debug("Creating %s", new_dir)
                    os.makedirs(new_dir)

    @property
    def coordinates(self):
        """Iterator on Chemical Checker coordinates."""
        for name, code in itertools.product("ABCDE", "12345"):
            yield name + code

    @property
    def datasets(self, exemplary_only=False):
        """Iterator on Chemical Checker datasets."""
        for dataset in Dataset.get():
            yield dataset.code

    def get_data_path(self, cctype, molset, dataset):
        """Return the signature data path for the given dataset.

        This should be the only place where we define the directory structure.
        The signature directory tipically contain the signature HDF5 file.

        Args:
            cctype(str): The Chemical Checker datatype i.e. one of the sign*.
            molset(str): The molecule set name.
            dataset(str): The dataset of the Chemical Checker.
        Returns:
            data_path(str): The signature data path.
        """
        data_path = os.path.join(self.cc_root, molset, dataset[:1],
                                 dataset[:2], dataset, cctype)
        self.__log.debug("signature path: %s", data_path)
        return data_path

    def get_signature(self, cctype, molset, dataset, **params):
        """Return the signature for the given dataset.

        Args:
            cctype(str): The Chemical Checker datatype i.e. one of the sign*.
            molset(str): The molecule set name.
            dataset(str): The dataset code of the Chemical Checker.
            params(dict): Optional. The set of parameters to initialize and
                compute the signature. If the signature is already initialized
                this argument will be ignored.
        Returns:
            data(Signature): A `Signature` object, the specific type depends
                on the cctype passed.
        """
        dataset_info = Dataset.get(dataset)
        if dataset_info is None:
            self.__log.warning(
                'Code %s returns no dataset', dataset)
            raise Exception("No dataset for code: " + dataset)
        data_path = self.get_data_path(cctype, molset, dataset)
        # initialize a data object factory feeding the type and the path
        data_factory = DataFactory()
        # the factory will return the signature with the right class
        data = data_factory.make_data(
            cctype, data_path, dataset_info, **params)
        return data
