"""Remove redundant rows in a data matrix.

Removes duplicates or near-duplicates using the
`Faiss library <https://github.com/facebookresearch/faiss>`_
"""
import os
import h5py
import pickle
import random
import numpy as np
from collections import defaultdict

from chemicalchecker.util import logged
from chemicalchecker.core.signature_data import DataSignature


@logged
class RNDuplicates():
    """RNDuplicates class."""

    def __init__(self, nbits=128, only_duplicates=False, cpu=1):
        """Initialize a RNDuplicates instance.

        Args:
            nbits (int): Number of bits to use to quantize.
            only_duplicates (boolean): Remove only exact duplicates.
            cpu (int): Number of cores to use.
        """
        self.nbits = nbits
        self.only_duplicates = only_duplicates
        self.cpu = cpu
        self.threshold = 100000
        self.chunk = 1000
        self.data_file = ''
        self.__log.debug('RNDuplicates to use ' + str(self.nbits) + " bits")

    def remove(self, data, keys=None, save_dest=None, just_mappings=False):
        """Remove redundancy from data.

        Args:
            data (array): The data to remove duplicates from. It can be a numpy
                array or a file path to a ``HDF5`` file with dataset ``V``.
            keys (array): Array of keys for the input data. If `None`, keys are
                taken from ``HDF5`` dataset ``keys``.
            save_dest (str): If the result needs to be saved in a file,
                the path to the file. (default: None)
            just_mappings (bool): Just return the mappings. Only applies if
                save_dest is None. (default=False)
        Returns:
            keys (array):
            data (array):
            mappings (dictionary):

        """
        try:
            import faiss
        except ImportError:
            raise ImportError("requires faiss " +
                              "https://github.com/facebookresearch/faiss")

        faiss.omp_set_num_threads(self.cpu)
        self.__log.info("Removing near duplicates.")

        if type(data) == str:
            self.__log.debug("Data input is: " + data)
            if os.path.isfile(data):
                dh5 = h5py.File(data, 'r')
                if "keys" not in dh5.keys() or "V" not in dh5.keys():
                    raise Exception(
                        "H5 file does not contain datasets 'keys' and 'V'")
                data_size = dh5["V"].shape
                if (data_size[0] < self.threshold and data_size[1] < self.threshold) or self.only_duplicates:
                    self.data = np.array(dh5["V"][:], dtype=float)
                else:
                    self.data = None
                    self.data_file = data
                self.data_type = dh5["V"].dtype
                self.keys = dh5["keys"][:]
                dh5.close()

            else:
                raise Exception("This module only accepts .h5 files")

        else:
            self.data = data
            data_size = self.data.shape
            self.data_type = data.dtype
            if keys is None:
                self.keys = np.array(range(len(data)))
            else:
                self.keys = np.array(keys)

        self.__log.info("Size before removing: " + str(data_size[0]))

        self.final_ids = list()
        self.mappings = dict()

        if self.only_duplicates:
            indexl2 = faiss.IndexFlatL2(self.data.shape[1])

            indexl2.add( np.array( self.data, dtype='float32' ) )

            self.__log.debug("Done adding in L2 space")

            D, I = indexl2.search( np.array(self.data, dtype='float32'), 1000)

            self.__log.debug("Done searching in L2 space")

            done = set()

            for i in range(len(D)):
                if i in done:
                    continue
                indexes = []
                for j in range(1000):
                    if i == I[i][j]:
                        continue
                    if D[i][j] <= 0.0:
                        done.add(I[i][j])
                        indexes.append(I[i][j])
                    else:
                        if len(indexes) > 0:
                            chosen = random.choice(indexes)
                            self.final_ids.append(chosen)
                            for v in indexes:
                                self.mappings[v] = self.keys[chosen]
                        else:
                            self.final_ids.append(i)
                            self.mappings[self.keys[i]] = self.keys[i]

                        break

        else:

            indexlsh = faiss.IndexLSH(data_size[1], self.nbits)

            if self.data is None:

                starts = range(0, data_size[0], self.chunk)

                dh5 = h5py.File(self.data_file, 'r')

                for start in starts:

                    indexlsh.add(
                        np.array(dh5["V"][start:start + self.chunk], dtype='float32') )
                dh5.close()

            else:

                indexlsh.add( np.array( self.data, dtype='float32' ) )

            indexes = faiss.vector_to_array(
                indexlsh.codes).reshape(-1, int(indexlsh.nbits / 8))

            buckets = defaultdict(list)

            for i in range(len(indexes)):
                buckets[indexes[i].tobytes()].append(i)

            for key, value in buckets.items():
                if(len(value) > 1):
                    chosen = random.choice(value)
                    self.final_ids.append(chosen)
                    for v in value:
                        self.mappings[self.keys[v]] = self.keys[chosen]
                else:
                    self.final_ids.append(value[0])
                    self.mappings[self.keys[value[0]]] = self.keys[value[0]]

        self.final_ids.sort()

        self.__log.info("Size after removing: " + str(len(self.final_ids)))
        if save_dest is not None:
            self.save(save_dest)
        else:
            if just_mappings:
                return self.mappings
            else:
                if self.data is None:
                    dh5 = h5py.File(self.data_file, "r")
                    self.data = dh5["V"][:]
                return self.keys[np.array(self.final_ids)], np.array(self.data[np.array(self.final_ids)], dtype=self.data_type), self.mappings

    def save(self, destination):
        """Save non-redundant data.

        Save non-redundant data to a ``HDF5`` file.

        Returns:
            destination (str): The destination file path.
        """

        dirpath = os.path.dirname(destination)

        self.__log.info("Saving removed duplicates to: " + destination)
        list_maps = sorted(self.mappings.items())
        with h5py.File(destination, 'w') as hf:
            keys = self.keys[np.array(self.final_ids)]
            hf.create_dataset("keys", data=np.array(keys, DataSignature.string_dtype()))
            if self.data is None:
                dh5 = h5py.File(self.data_file, 'r')
                hf.create_dataset("V", (len(self.final_ids), dh5["V"].shape[1]), dtype=self.data_type)
                for count, i in enumerate(self.final_ids):
                    hf["V"][count] = dh5["V"][i]
            else:
                V = np.array(
                    self.data[np.array(self.final_ids)], dtype=self.data_type)
                hf.create_dataset("V", data=V)

            hf.create_dataset("shape", data=hf["V"].shape)
            hf.create_dataset("mappings",
                              data=np.array(list_maps,
                                            DataSignature.string_dtype()))
        self.__log.debug("Writing mappings to: " + dirpath)
        with open(os.path.join(dirpath, "mappings"), 'wb') as fh:
            pickle.dump(self.mappings, fh)

