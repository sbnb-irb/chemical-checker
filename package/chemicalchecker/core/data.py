"""This class initialize and serve the different Chemical Checker data types.

It is a factory of different signatures (including along with clusters and
neighbors). Is the place where the classes implementing such data types are
imported and initialized.
"""
import os
import h5py

from chemicalchecker.util import logged


@logged
class DataFactory():

    @staticmethod
    def make_data(cctype, signature_path, validation_path, dataset_code, **params):
        from .sign0 import sign0
        from .sign1 import sign1
        from .clus1 import clus1
        from .neig1 import neig1
        from .sign2 import sign2
        from .sign3 import sign3
        from .proj1 import proj1

        try:
            DataFactory.__log.debug("initializing object %s", cctype)
            args = (signature_path, validation_path, dataset_code)
            return eval(cctype)(*args, **params)
        except Exception as ex:
            raise Exception("Data type %s not available: %s" % (cctype, ex))

    @staticmethod
    def signaturize(cctype, signature_path, matrix, keys=None, dataset_code=None):
        """Signaturize a given matrix.

        Produce a signature-like structure for the given input.
        Given a matrix of anonymous molucules add keys to mantain the order.
        """
        from .sign0 import sign0
        from .sign1 import sign1
        from .clus1 import clus1
        from .neig1 import neig1
        from .sign2 import sign2
        from .sign3 import sign3
        from .proj1 import proj1

        data_path = os.path.join(signature_path, '%s.h5' % cctype)
        if not keys:
            keys = ["{0:027d}".format(n) for n in range(len(matrix))]
        if not dataset_code:
            dataset_code = "XX.001"
        with h5py.File(data_path, 'w') as hf:
            hf.create_dataset("keys", data=keys)
            hf.create_dataset("V", data=matrix)
            hf.create_dataset("shape", data=matrix.shape)
        try:
            return eval(cctype)(signature_path, signature_path, dataset_code)
        except Exception as ex:
            raise Exception("Data type %s not available: %s" % (cctype, ex))
