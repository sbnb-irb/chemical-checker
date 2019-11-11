"""Utility for removing near duplicates from matrix data.

It removes duplicates or near-duplicates by using Faiss library,
which the data vectors according their similarity.
"""
import os
import h5py
import pickle
import random
import numpy as np
import collections
from collections import defaultdict

from chemicalchecker.util import logged
from chemicalchecker.core.signature_data import DataSignature


@logged
class RNDuplicates():
    """Removes near duplicates from matrix data."""

    def __init__(self, nbits=128, only_duplicates=False, cpu=1):
        """Initialize the RNDuplicates object.

        Args:
            nbits(int): Number of bits to use to quantize
            only_duplicates(boolean): Remove only exact duplicates
            cpu(int): Number of cores to use
        """
        self.nbits = nbits
        self.only_duplicates = only_duplicates
        self.cpu = cpu
        self.threshold = 100000
        self.chunk = 1000
        self.data_file = ''
        self.__log.debug('RNDuplicates to use ' + str(self.nbits) + " bits")

    def remove(self, data, keys=None, save_dest=None):
        """Remove near duplicates from data.

        Args:
            data(array): The data to remove duplicates from. It can be a numpy array
                         or a file path to a .h5 file
            keys(array): Array of keys for the input data
            save_dest(str): If the resulyt needs to be saved in a file, the path to the file( default: None)
        Returns:
            keys(array):
            data(array):
            mappings(dictionary):

        """
        try:
            import faiss
        except ImportError:
            raise ImportError("requires faiss " +
                              "https://github.com/facebookresearch/faiss")
        faiss.omp_set_num_threads(self.cpu)

        if type(data) == str:
            self.__log.debug("Data input is: " + data)
            if os.path.isfile(data) and data[-3:] == ".h5":
                dh5 = h5py.File(data, 'r')
                if "keys" not in dh5.keys() or "V" not in dh5.keys():
                    raise Exception(
                        "H5 file " + data + " does not contain datasets 'keys' and 'V'")
                data_size = dh5["V"].shape
                if data_size[0] < self.threshold or self.only_duplicates:
                    self.data = np.array(dh5["V"][:], dtype=np.float32)
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

            indexl2.add(self.data)

            self.__log.debug("Done adding in L2 space")

            D, I = indexl2.search(self.data, 1000)

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

            if data_size[0] > self.threshold:

                starts = range(0, data_size[0], self.chunk)

                dh5 = h5py.File(data, 'r')

                for start in starts:

                    indexlsh.add(
                        np.array(dh5["V"][start:start + self.chunk], dtype=np.float32))
                dh5.close()

            else:

                indexlsh.add(self.data)

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
            return self.keys[np.array(self.final_ids)], np.array(self.data[np.array(self.final_ids)], dtype=self.data_type), self.mappings

    def save(self, destination):
        """Save data after removing to a h5 file.

        Returns:
            destination(str): The destination file(.h5 file) where to save the new data after removing.

        """
        if destination[-3:] != ".h5":
            raise Exception("The destination file needs to be a .h5 file")

        dirpath = os.path.dirname(destination)

        self.__log.info("Saving removed duplicates to : " + destination)
        list_maps = sorted(self.mappings.items())
        self.__log.info("Starting to write to : " + destination)
        with h5py.File(destination, 'w') as hf:
            hf.create_dataset("keys", data=self.keys[np.array(self.final_ids)])
            if self.data is None:
                dh5 = h5py.File(self.data_file, 'r')
                V = np.array(
                    [dh5["V"][i] for i in self.final_ids], dtype=self.data_type)
            else:
                V = np.array(
                    self.data[np.array(self.final_ids)], dtype=self.data_type)
            hf.create_dataset("V", data=V)
            hf.create_dataset("shape", data=V.shape)
            hf.create_dataset("mappings",
                              data=np.array(list_maps,
                                            DataSignature.string_dtype()))
        self.__log.info("Writing mappings to " + dirpath)
        with open(os.path.join(dirpath, "mappings"), 'wb') as fh:
            pickle.dump(self.mappings, fh)
