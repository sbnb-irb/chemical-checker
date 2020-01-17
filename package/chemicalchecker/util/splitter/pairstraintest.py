import h5py
import faiss
import itertools
import numpy as np
from tqdm import tqdm

from chemicalchecker.util import logged
from chemicalchecker.util.remove_near_duplicates import RNDuplicates


@logged
class PairTraintest(object):
    """Convenience batch reader from HDF5 files of pairs.

    This class allow creation and access to HDF5 train-test sets and expose
    the generator functions which tensorflow likes.
    """

    def __init__(self, hdf5_file, split, nr_neig=10, replace_nan=None):
        """Initialize the traintest object.

        We assume the file is containing diffrent splits.
        e.g. "x_train", "y_train", "x_test", ...
        """
        self._file = hdf5_file
        self._f = None
        self.replace_nan = replace_nan
        self.nr_neig = nr_neig
        if split is None:
            self.p_name = "p"
            self.x_name = "x"
            self.y_name = "y"
        else:
            self.p_name = "p_%s" % split
            self.x_name = "x"
            self.y_name = "y_%s" % split
            available_splits = self.get_split_names()
            if split not in available_splits:
                raise Exception("Split '%s' not found in %s!" %
                                (split, str(available_splits)))

    def valid(self):
        '''Check if nr of neighbors correspond to want is desired.'''
        self.open()
        if "nr_neig" in self._f:
            if self._f["nr_neig"] == self.nr_neig:
                self.close()
                return True
        self.close()
        return False

    def get_xy_shapes(self):
        """Return the shpaes of X an Y."""
        self.open()
        x_shape = self._f[self.x_name].shape
        y_shape = self._f[self.y_name].shape
        self.close()
        return x_shape, y_shape

    def get_split_names(self):
        """Return the name of the splits."""
        self.open()
        if "split_names" in self._f:
            split_names = [a.decode() for a in self._f["split_names"]]
        else:
            split_names = ['train', 'test']
        self.close()
        combos = itertools.combinations_with_replacement(split_names, 2)
        return ['_'.join(x) for x in combos]

    def open(self):
        """Open the HDF5."""
        self._f = h5py.File(self._file, 'r')
        self.__log.info("HDF5 open %s", self._file)

    def close(self):
        """Close the HDF5."""
        try:
            self._f.close()
            self.__log.info("HDF5 close %s", self._file)
        except AttributeError:
            self.__log.error('HDF5 file is not open yet.')

    def get_py(self, beg_idx, end_idx):
        """Get a batch of X and Y."""
        features = self._f[self.p_name][beg_idx: end_idx]
        # handle NaNs
        if self.replace_nan is not None:
            features[np.where(np.isnan(features))] = self.replace_nan
        labels = self._f[self.y_name][beg_idx: end_idx]
        return features, labels

    def get_p(self, beg_idx, end_idx):
        """Get a batch of X."""
        features = self._f[self.p_name][beg_idx: end_idx]
        # handle NaNs
        if self.replace_nan is not None:
            features[np.where(np.isnan(features))] = self.replace_nan
        return features

    def get_y(self, beg_idx, end_idx):
        """Get a batch of Y."""
        features = self._f[self.y_name][beg_idx: end_idx]
        # handle NaNs
        if self.replace_nan is not None:
            features[np.where(np.isnan(features))] = self.replace_nan
        return features

    def get_all_x(self):
        """Get full X."""
        features = self._f[self.x_name][:]
        # handle NaNs
        if self.replace_nan is not None:
            features[np.where(np.isnan(features))] = self.replace_nan
        return features

    def get_all_p(self):
        """Get full X."""
        features = self._f[self.p_name][:]
        # handle NaNs
        if self.replace_nan is not None:
            features[np.where(np.isnan(features))] = self.replace_nan
        return features

    def get_all_y(self):
        """Get full Y."""
        labels = self._f[self.y_name][:]
        return labels

    @staticmethod
    def create_signature_file(sign_from, sign_to, out_filename):
        """Create the HDF5 file with both X and Y, train and test."""
        # get type1
        with h5py.File(sign_from, 'r') as fh:
            X = fh['V'][:]
            check_X = fh['keys'][:]
        X = np.asarray(X, dtype=np.float32)
        # get type2
        with h5py.File(sign_to, 'r') as fh:
            Y = fh['V'][:]
            check_Y = fh['keys'][:]
        assert(np.array_equal(check_X, check_Y))
        # train test validation splits
        PairTraintest.create(X, Y, out_filename)

    @staticmethod
    def get_split_indeces(rows, fractions):
        """Get random indeces for different splits."""
        if not sum(fractions) == 1.0:
            raise Exception("Split fractions should sum to 1.0")
        # shuffle indeces
        idxs = list(range(rows))
        np.random.shuffle(idxs)
        # from frequs to indices
        splits = np.cumsum(fractions)
        splits = splits[:-1]
        splits *= len(idxs)
        splits = splits.round().astype(np.int)
        return np.split(idxs, splits)

    @staticmethod
    def create(X, out_file, neigbors_matrix=None, neigbors=10,
               mean_center_x=True, shuffle=True,
               split_names=['train', 'test'], split_fractions=[.8, .2],
               x_dtype=np.float32, y_dtype=np.float32, debug=False):
        """Create the HDF5 file with validation splits for both X and Y.

        Args:
            X(numpy.ndarray): features to train from.
            Y(numpy.ndarray): labels to predict.
            out_file(str): path of the h5 file to write.
            split_names(list(str)): names for the split of data.
            split_fractions(list(float)): fraction of data in each split.
            x_dtype(type): numpy data type for X.
            y_dtype(type): numpy data type for Y (np.float32 for regression,
                np.int32 for classification.
        """
        PairTraintest.__log.debug(
            "{:<20} shape: {:>10}".format("input X", str(X.shape)))
        # train test validation splits
        if len(split_names) != len(split_fractions):
            raise Exception("Split names and fraction should be same amount.")
        #split_names = [s.encode() for s in split_names]

        # override parameters for debug
        if debug:
            neigbors = 10
            split_names = ['train', 'test']
            split_fractions = [.8, .2]

        # the neigbors_matrix is optional
        if neigbors_matrix is None:
            neigbors_matrix = X
        else:
            if len(neigbors_matrix) != len(X):
                raise Exception("neigbors_matrix shuold be same length as X.")

        # reduce redundancy, keep full-ref mapping
        rnd = RNDuplicates(cpu=10)
        _, ref_matrix, full_ref_map = rnd.remove(neigbors_matrix)
        ref_full_map = np.array(rnd.final_ids)
        rows = ref_matrix.shape[0]

        if debug:
            # we'll use this to later check that the mapping went fine
            test = faiss.IndexFlatL2(neigbors_matrix.shape[1])
            test.add(neigbors_matrix)
            tmp = dict()
            for key, value in full_ref_map.items():
                tmp.setdefault(value, list()).append(key)
            max_repeated = max([len(x) for x in tmp.values()])
            _, test_neig = test.search(neigbors_matrix, max_repeated + 1)

        # split chunks, get indeces of chunks for each split
        chunk_size = np.floor(rows / 100)
        split_chunk_idx = PairTraintest.get_split_indeces(
            int(np.floor(rows / chunk_size)) + 1,
            split_fractions)

        # split ref matrix, keep ref-split mapping
        nr_matrix = dict()
        split_ref_map = dict()
        for split_name, chunks in zip(split_names, split_chunk_idx):
            # need total size and mapping of chunks
            src_dst = list()
            total_size = 0
            for dst, src in enumerate(sorted(chunks)):
                # source chunk start-end
                src_start = src * chunk_size
                src_end = (src * chunk_size) + chunk_size
                # check current chunk size to avoid overflowing
                curr_chunk_size = chunk_size
                if src_end > ref_matrix.shape[0]:
                    src_end = ref_matrix.shape[0]
                    curr_chunk_size = src_end - src_start
                # update total size
                total_size += curr_chunk_size
                # destination start-end
                dst_start = dst * chunk_size
                dst_end = (dst * chunk_size) + curr_chunk_size
                src_slice = (int(src_start), int(src_end))
                dst_slice = (int(dst_start), int(dst_end))
                src_dst.append((src_slice, dst_slice))
            # create chunk matrix
            cols = ref_matrix.shape[1]
            nr_matrix[split_name] = np.zeros((int(total_size), cols),
                                             dtype=ref_matrix.dtype)
            split_ref_map[split_name] = dict()
            ref_idxs = np.arange(ref_matrix.shape[0])
            for src_slice, dst_slice in tqdm(src_dst):
                src_chunk = slice(*src_slice)
                dst_chunk = slice(*dst_slice)
                PairTraintest.__log.debug(
                    "writing src: %s  to dst: %s" % (src_slice, dst_slice))
                ref_src_chunk = ref_idxs[src_chunk]
                ref_dst_chunk = ref_idxs[dst_chunk]
                for src_id, dst_id in zip(ref_src_chunk, ref_dst_chunk):
                    split_ref_map[split_name][dst_id] = src_id
                nr_matrix[split_name][dst_chunk] = ref_matrix[src_chunk]
            PairTraintest.__log.debug(
                "nr_matrix %s %s", split_name, nr_matrix[split_name].shape)

        # for each split generate NN
        NN = dict()
        for split_name in split_names:
            # create faiss index
            NN[split_name] = faiss.IndexFlatL2(nr_matrix[split_name].shape[1])
            # add data
            NN[split_name].add(nr_matrix[split_name])

        # mean centering columns
        if mean_center_x == True:
            mean = np.nanmean(X, axis=1)
            X = X - mean[:, np.newaxis]

        # create dataset
        PairTraintest.__log.info('Traintest saving to %s', out_file)
        with h5py.File(out_file, "w") as fh:
            fh.create_dataset('x', data=X)

            # for each split combo generate pairs and ys
            #combos = itertools.combinations_with_replacement(split_names, 2)
            combos = [('train', 'train'), ('train', 'test'), ('test', 'test')]
            for split1, split2 in combos:
                # handle case where we ask more neig then molecules
                if neigbors > nr_matrix[split2].shape[0]:
                    combo_neig = nr_matrix[split2].shape[0]
                    PairTraintest.__log.warning(
                        'split %s is small, reducing neigbors to %i' %
                        (split2, combo_neig))
                else:
                    combo_neig = neigbors
                # remove self neighbors when splits are the same
                if split1 == split2:
                    # search NN
                    dists, neig_idxs = NN[split1].search(nr_matrix[split2],
                                                         combo_neig + 1)
                    # the nearest neig between same groups is the molecule
                    # itself
                    assert(all(neig_idxs[:, 0] ==
                               np.arange(0, len(neig_idxs))))
                    neig_idxs = neig_idxs[:, 1:]
                else:
                    _, neig_idxs = NN[split1].search(
                        nr_matrix[split2], combo_neig)
                if debug:
                    _, neig_idxs = NN[split1].search(
                        nr_matrix[split2], combo_neig)
                # get positive pairs
                # get first pair element idxs
                idxs1 = np.repeat(
                    np.arange(nr_matrix[split2].shape[0]), combo_neig)
                # get second pair elements idxs
                idxs2_1 = neig_idxs.flatten()
                assert(len(idxs1) == len(idxs2_1))
                # map back to reference
                idxs1_ref = np.array([split_ref_map[split2][x] for x in idxs1])
                idxs2_1_ref = np.array(
                    [split_ref_map[split1][x] for x in idxs2_1])
                # map back to full
                idxs1_full = ref_full_map[idxs1_ref]
                idxs2_1_full = ref_full_map[idxs2_1_ref]
                if debug:
                    # train ~= full
                    total = 0
                    ok = 0
                    for t1, t2 in zip(idxs1_full, idxs2_1_full):
                        total += 1
                        if t2 in test_neig[t1]:
                            ok += 1
                    print(split1, split2, ok / total, ok, combo_neig)
                # get negative pairs
                idxs2_0 = list()
                for row in neig_idxs:
                    no_neig = set(range(nr_matrix[split2].shape[0])) - set(row)
                    smpl = np.random.choice(
                        list(no_neig), combo_neig, replace=False)
                    idxs2_0.extend(smpl)
                idxs2_0 = np.array(idxs2_0)
                # map
                idxs2_0_ref = np.array(
                    [split_ref_map[split1][x] for x in idxs2_0])
                idxs2_0_full = ref_full_map[idxs2_0_ref]

                # stack pairs and ys
                pairs_1 = np.vstack((idxs1_full, idxs2_1_full)).T
                y_1 = np.ones((1, pairs_1.shape[0]))
                pairs_0 = np.vstack((idxs1_full, idxs2_0_full)).T
                y_0 = np.zeros((1, pairs_0.shape[0]))
                all_pairs = np.vstack((pairs_1, pairs_0))
                all_ys = np.hstack((y_1, y_0)).T

                # shuffling
                shuffle_idxs = np.arange(all_ys.shape[0])
                if shuffle:
                    np.random.shuffle(shuffle_idxs)

                # save to h5
                ds_name = "p_%s_%s" % (split1, split2)
                PairTraintest.__log.info(
                    'writing %s %s %s', ds_name, pairs_1.shape, pairs_0.shape)
                fh.create_dataset(ds_name, data=all_pairs[shuffle_idxs])
                ds_name = "y_%s_%s" % (split1, split2)
                PairTraintest.__log.info(
                    'writing %s %s', ds_name, all_ys.shape)
                fh.create_dataset(ds_name, data=all_ys[shuffle_idxs])

        PairTraintest.__log.info('PairTraintest saved to %s', out_file)

    @staticmethod
    def generator_fn(file_name, split, batch_size=None, only_x=False):
        """Return the generator function that we can query for batches."""
        reader = PairTraintest(file_name, split)
        reader.open()
        # read shapes
        x_shape = reader._f[reader.x_name].shape
        y_shape = reader._f[reader.y_name].shape
        # read data types
        x_dtype = reader._f[reader.x_name].dtype
        y_dtype = reader._f[reader.y_name].dtype
        # no batch size -> return everything
        p_shape = reader._f[reader.p_name].shape
        if not batch_size:
            batch_size = p_shape[0]
        # keep X in memory for resolving pairs quickly
        X = reader.get_all_x()

        def example_generator_fn():
            # generator function yielding data
            epoch = 0
            beg_idx, end_idx = 0, batch_size
            total = reader._f[reader.x_name].shape[0]
            while True:
                if beg_idx >= total:
                    PairTraintest.__log.debug("EPOCH completed")
                    beg_idx = 0
                    epoch += 1
                    return
                if only_x:
                    pairs = reader.get_p(beg_idx, end_idx)
                    x1 = X[pairs[:, 0]]
                    x2 = X[pairs[:, 1]]
                    yield x1, x2
                else:
                    pairs, y = reader.get_py(beg_idx, end_idx)
                    x1 = X[pairs[:, 0]]
                    x2 = X[pairs[:, 1]]
                    yield x1, x2, y
                beg_idx, end_idx = beg_idx + batch_size, end_idx + batch_size

        shapes = (x_shape, x_shape, y_shape)
        dtypes = (x_dtype, x_dtype, y_dtype)
        return shapes, dtypes, example_generator_fn
