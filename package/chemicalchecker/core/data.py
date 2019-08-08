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
    def make_data(cctype, *args, **kwargs):
        from .sign0 import sign0
        from .sign1 import sign1
        from .sign2 import sign2
        from .sign3 import sign3

        from .clus import clus
        from .neig import neig
        from .proj import proj

        DataFactory.__log.debug("initializing object %s", cctype)
        if cctype[:4] in ['clus', 'neig', 'proj']:
            return eval(cctype[:4])(*args, **kwargs)
        else:
            return eval(cctype)(*args, **kwargs)

    @staticmethod
    def signaturize(cctype, signature_path, matrix, keys=None, dataset_code=None):
        """Signaturize a given matrix.

        Produce a signature-like structure for the given input.
        Given a matrix of anonymous molucules add keys to mantain the order.
        """
        from .sign0 import sign0
        from .sign1 import sign1
        from .sign2 import sign2
        from .sign3 import sign3

        from .clus import clus
        from .neig import neig
        from .proj import proj

        data_path = os.path.join(signature_path, '%s.h5' % cctype)
        if not keys:
            keys = ["{0:027d}".format(n) for n in range(len(matrix))]
        if not dataset_code:
            dataset_code = "XX.001"
        with h5py.File(data_path, 'w') as hf:
            hf.create_dataset("keys", data=keys)
            hf.create_dataset("V", data=matrix)
            hf.create_dataset("shape", data=matrix.shape)
        if cctype[:4] in ['clus', 'neig', 'proj']:
            return eval(cctype[:4])(signature_path, dataset_code)
        else:
            return eval(cctype)(signature_path, dataset_code)
