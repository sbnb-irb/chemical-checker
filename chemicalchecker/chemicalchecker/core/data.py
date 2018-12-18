"""This class initialize and serve the different Chemical Checker data types.

It is a factory of different signatures (including along with clusters and
neighbors). Is the place where the classes implementing such data types are
imported and initialized.
"""
from .sign0 import sign0
from .sign1 import sign1
from .clst1 import clst1
from .neig1 import neig1
from .sign2 import sign2
from .proj1 import proj1

from chemicalchecker.util import logged


@logged
class DataFactory():

    def make_data(self, cctype, data_path, model_path, stats_path, dataset_info, **params):
        if cctype in globals():
            self.__log.debug("initializing object %s", cctype)
            return eval(cctype)(data_path, model_path, stats_path, dataset_info, **params)
        else:
            raise Exception("Data type %s not available" % cctype)
