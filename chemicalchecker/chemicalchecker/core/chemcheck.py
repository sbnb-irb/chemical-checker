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
            cc_root(str): The Chemical Checker root directory.
        """
        self.cc_root = cc_root
        self.__log.debug("ChemicalChecker with root: %s", cc_root)
        if not os.path.isdir(cc_root):
            self.__log.warning("Empty root directory, creating new one")
            for dataset in self.datasets:
                new_dir = os.path.join(
                    cc_root, dataset[:1], dataset[:2], dataset, 'models')
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

    def get_data_path(self, cctype, dataset):
        """Return the path to signature file for the given dataset.

        This should be the only place where we define the directory structure.
        The signature type directly map to a HDF5 file.

        Args:
            cctype(str): The Chemical Checker datatype i.e. one of the sign*.
            dataset(str): The dataset of the Chemical Checker.
        Returns:
            data_path(str): The path to an .h5 file.
        """
        filename = '{}.h5'.format(cctype)
        data_path = os.path.join(self.cc_root, dataset[:1],
                                 dataset[:2], dataset, filename)
        self.__log.debug("data path: %s", data_path)
        return data_path

    def get_model_path(self, cctype, dataset):
        """Return the path to model file for the given dataset.

        This should be the only place where we define the directory structure.
        The signature type directly map to a persistent model directory.

        Args:
            cctype(str): The Chemical Checker datatype i.e. one of the sign*.
            dataset(str): The dataset of the Chemical Checker.
        Returns:
            model_path(str): The path to an persistent model directory.
        """
        # filename = '{}.pkl'.format(cctype)
        model_path = os.path.join(self.cc_root, dataset[:1],
                                  dataset[:2], dataset, 'models')
        self.__log.debug("model path: %s", model_path)
        return model_path

    def get_stats_path(self, cctype, dataset):
        """Return the path to statistics directory for the given dataset.

        This should be the only place where we define the directory structure.
        The signature type directly map to a persistent stats directory.

        Args:
            cctype(str): The Chemical Checker datatype i.e. one of the sign*.
            dataset(str): The dataset of the Chemical Checker.
        Returns:
            stats_path(str): The path to the stats directory.
        """
        # filename = '{}.pkl'.format(cctype)
        stats_path = os.path.join(self.cc_root, dataset[:1],
                                  dataset[:2], dataset, 'stats')
        self.__log.debug("model path: %s", stats_path)
        return stats_path

    def get_data(self, cctype, dataset, **params):
        """Return the full signature for the given dataset.

        Args:
            cctype(str): The Chemical Checker datatype i.e. one of the sign*.
            dataset(str): The dataset of the Chemical Checker.
        Returns:
            data(Signature): A `Signature` object, the specific type depends
                on the cctype passed.
        """
        dataset_info = Dataset.get(dataset)
        if dataset_info is None:
            self.__log.warning(
                'Code %s returns no dataset', dataset)
            raise Exception("No dataset for code: " + dataset)
        data_path = self.get_data_path(cctype, dataset)
        model_path = self.get_model_path(cctype, dataset)
        stats_path = self.get_plots_path(cctype, dataset)
        # initialize a data object factory feeding the type and the path
        data_factory = DataFactory()
        # the factory will spit the data in the right class
        data = data_factory.make_data(
            cctype, data_path, model_path, stats_path, dataset_info, **params)
        return data
