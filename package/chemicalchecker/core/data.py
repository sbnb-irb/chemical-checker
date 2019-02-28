"""This class initialize and serve the different Chemical Checker data types.

It is a factory of different signatures (including along with clusters and
neighbors). Is the place where the classes implementing such data types are
imported and initialized.
"""
import os
import h5py

from .sign0 import sign0
from .sign1 import sign1
from .clus1 import clus1
from .neig1 import neig1
from .sign2 import sign2
from .proj1 import proj1

from chemicalchecker.util import logged


@logged
class DataFactory():

    def make_data(self, cctype, signature_path, validation_path, dataset, **params):
        if cctype in globals():
            self.__log.debug("initializing object %s", cctype)
            return eval(cctype)(signature_path, validation_path, dataset, **params)
        else:
            raise Exception("Data type %s not available" % cctype)

    @staticmethod
    def signaturize(cctype, signature_path, matrix, keys=None, dataset=None):
        """Signaturize a given matrix.

        Produce a signature-like structure for the given input.
        Given a matrix of anonymous molucules add keys to mantain the order.
        """
        data_path = os.path.join(signature_path, '%s.h5' % cctype)
        if not keys:
            keys = ["{0:027d}".format(n) for n in range(len(matrix))]
        if not dataset:
            dataset = "XX.001"
        with h5py.File(data_path, 'w') as hf:
            hf.create_dataset("keys", data=keys)
            hf.create_dataset("V", data=matrix)
            hf.create_dataset("shape", data=matrix.shape)
        return eval(cctype)(signature_path, signature_path, dataset)
