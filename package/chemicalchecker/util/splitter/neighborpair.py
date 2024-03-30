"""Splitter on Neighbor pairs."""
import os
import h5py
import pickle
import itertools
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import RobustScaler

from chemicalchecker.util import logged
from chemicalchecker.util.remove_near_duplicates import RNDuplicates


@logged
class NeighborPairTraintest(object):
    """NeighborPairTraintest class."""

    def __init__(self, hdf5_file, split, nr_neig=10, replace_nan=None):
        """Initialize a NeighborPairTraintest instnace.

        We assume the file is containing diffrent splits.
        e.g. "x_train", "y_train", "x_test", ...
        """
        self._file = hdf5_file
        self._f = None
        self.replace_nan = replace_nan
        self.nr_neig = nr_neig
        self.x_name = "x"
        if split is None:
            self.p_name = "p"
            self.y_name = "y"
        else:
            self.p_name = "p_%s" % split
            self.y_name = "y_%s" % split
            available_splits = self.get_split_names()
            if split not in available_splits:
                raise Exception("Split '%s' not found in %s!" %
                                (split, str(available_splits)))

    def get_py_shapes(self):
        """Return the shpaes of X an Y."""
        self.open()
        p_shape = self._f[self.p_name].shape
        y_shape = self._f[self.y_name].shape
        self.close()
        return p_shape, y_shape

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
        #self.__log.info("HDF5 open %s", self._file)

    def close(self):
        """Close the HDF5."""
        try:
            self._f.close()
            #self.__log.info("HDF5 close %s", self._file)
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
        splits = splits.round().astype(int)
        return np.split(idxs, splits)

    @staticmethod
    def create(X, out_file, neigbors_matrix=None, pos_neighbors=10,
               neg_neighbors=100, scaler_dest=None,
               mean_center_x=True, shuffle=True,
               check_distances=True,
               split_names=['train', 'test'], split_fractions=[.8, .2],
               x_dtype=float, y_dtype=float, debug_test=False):
        """Create the HDF5 file with validation splits.

        Args:
            X(numpy.ndarray): features to train from.
            out_file(str): path of the h5 file to write.
            neigbors_matrix(numpy.ndarray): matrix for computing neighbors.
            neigbors(int): Number of positive neighbors to include.
            mean_center_x(bool): center each feature on its mean?
            shuffle(bool): Shuffle positive and negatives.
            split_names(list(str)): names for the split of data.
            split_fractions(list(float)): fraction of data in each split.
            x_dtype(type): numpy data type for X.
            y_dtype(type): numpy data type for Y (float for regression,
                int32 for classification.
        """
        try:
            import faiss
        except ImportError as err:
            raise err
        NeighborPairTraintest.__log.debug(
            "{:<20} shape: {:>10}".format("input X", str(X.shape)))
        # train test validation splits
        if len(split_names) != len(split_fractions):
            raise Exception("Split names and fraction should be same amount.")

        # override parameters for debug
        if debug_test:
            pos_neigbors = 10
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

        if debug_test:
            # we'll use this to later check that the mapping went fine
            test = faiss.IndexFlatL2(neigbors_matrix.shape[1])
            test.add( np.array(neigbors_matrix, dtype='float32') )
            tmp = dict()
            for key, value in full_ref_map.items():
                tmp.setdefault(value, list()).append(key)
            max_repeated = max([len(x) for x in tmp.values()])
            _, test_neig = test.search( np.array(neigbors_matrix, dtype='float32'), max_repeated + 1)

        # split chunks, get indeces of chunks for each split
        chunk_size = np.floor(rows / 100)
        split_chunk_idx = NeighborPairTraintest.get_split_indeces(
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
                NeighborPairTraintest.__log.debug(
                    "writing src: %s  to dst: %s" % (src_slice, dst_slice))
                ref_src_chunk = ref_idxs[src_chunk]
                ref_dst_chunk = ref_idxs[dst_chunk]
                for src_id, dst_id in zip(ref_src_chunk, ref_dst_chunk):
                    split_ref_map[split_name][dst_id] = src_id
                nr_matrix[split_name][dst_chunk] = ref_matrix[src_chunk]
            NeighborPairTraintest.__log.debug(
                "nr_matrix %s %s", split_name, nr_matrix[split_name].shape)

        # for each split generate NN
        NN = dict()
        for split_name in split_names:
            # create faiss index
            NN[split_name] = faiss.IndexFlatL2(nr_matrix[split_name].shape[1])
            # add data
            NN[split_name].add( np.array( nr_matrix[split_name], dtype='float32') )

        # mean centering columns
        if mean_center_x:
            scaler = RobustScaler()
            X = scaler.fit_transform(X)
            if scaler_dest is None:
                scaler_dest = os.path.split(out_file)[0]
            scaler_file = os.path.join(scaler_dest, 'scaler.pkl')
            pickle.dump(scaler, open(scaler_file, 'wb'))

        # create dataset
        NeighborPairTraintest.__log.info('Traintest saving to %s', out_file)
        with h5py.File(out_file, "w") as fh:
            fh.create_dataset('x', data=X)
            # for each split combo generate pairs and ys
            combos = itertools.combinations_with_replacement(split_names, 2)
            #combos = [('train', 'train'), ('train', 'test'), ('test', 'test')]
            for split1, split2 in combos:
                # handle case where we ask more neig then molecules
                if pos_neighbors > nr_matrix[split2].shape[0]:
                    combo_neig = nr_matrix[split2].shape[0]
                    NeighborPairTraintest.__log.warning(
                        'split %s is small, reducing pos_neighbors to %i' %
                        (split2, combo_neig))
                else:
                    combo_neig = pos_neighbors
                # remove self neighbors when splits are the same
                if split1 == split2:
                    # search NN
                    dists, neig_idxs = NN[split1].search( np.array(nr_matrix[split2], dtype='float32'),
                                                         combo_neig + 1)
                    # the nearest neig between same groups is the molecule
                    # itself
                    assert(all(neig_idxs[:, 0] ==
                               np.arange(0, len(neig_idxs))))
                    neig_idxs = neig_idxs[:, 1:]
                else:
                    _, neig_idxs = NN[split1].search(
                        np.array( nr_matrix[split2], dtype='float32'), combo_neig)
                if debug_test:
                    _, neig_idxs = NN[split1].search(
                        np.array( nr_matrix[split2], dtype='float32'), combo_neig)
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
                # oversample the positives
                neg_pos_ratio = np.floor(neg_neighbors / combo_neig)
                idxs1_full = np.repeat(idxs1_full, neg_pos_ratio)
                idxs2_1_full = np.repeat(idxs2_1_full, neg_pos_ratio)
                if debug_test:
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
                for idx, row in enumerate(neig_idxs):
                    no_neig = set(range(nr_matrix[split2].shape[0])) - set(row)
                    # avoid fetching itself as negative!
                    if split1 == split2:
                        no_neig = no_neig - set([idx])
                    smpl = np.random.choice(
                        list(no_neig), neg_neighbors, replace=False)
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

                if check_distances:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    d1 = list()
                    d0 = list()
                    for idx in range(len(all_ys))[:500]:
                        dist = euclidean(
                            neigbors_matrix[all_pairs[shuffle_idxs][idx][0]],
                            neigbors_matrix[all_pairs[shuffle_idxs][idx][1]])
                        if all_ys[shuffle_idxs][idx] == 1:
                            d1.append(dist)
                        else:
                            d0.append(dist)
                    name = "%s_%s" % (split1, split2)
                    plot_file = os.path.join(os.path.split(out_file)[0],
                                             'dist_%s.png' % name)
                    sns.distplot(d1, label='1')
                    sns.distplot(d0, label='0')
                    plt.legend()
                    plt.savefig(plot_file)
                    plt.close()

                # save to h5
                ds_name = "p_%s_%s" % (split1, split2)
                NeighborPairTraintest.__log.info(
                    'writing %s %s %s', ds_name, pairs_1.shape, pairs_0.shape)
                fh.create_dataset(ds_name, data=all_pairs[shuffle_idxs])
                ds_name = "y_%s_%s" % (split1, split2)
                NeighborPairTraintest.__log.info(
                    'writing %s %s', ds_name, all_ys.shape)
                fh.create_dataset(ds_name, data=all_ys[shuffle_idxs])

        NeighborPairTraintest.__log.info(
            'NeighborPairTraintest saved to %s', out_file)

    @staticmethod
    def generator_fn(file_name, split, batch_size=None, only_x=False,
                     replace_nan=None, augment_scale=1,
                     augment_fn=None, augment_kwargs={},
                     mask_fn=None, shuffle=True,
                     sharedx=None):
        """Return the generator function that we can query for batches.

        file_name(str): The H5 generated via `create`
        split(str): One of 'train_train', 'train_test', or 'test_test'
        batch_size(int): Size of a batch of data.
        only_x(bool): Usually when predicting only X are useful.
        replace_nan(bool): Value used for NaN replacement.
        augment_scale(int): Augment the train size by this factor.
        augment_fn(func): Function to augment data.
        augment_kwargs(dict): Parameters for the aument functions.
        """
        reader = NeighborPairTraintest(file_name, split)
        reader.open()
        # read shapes
        x_shape = reader._f[reader.x_name].shape
        y_shape = reader._f[reader.y_name].shape
        p_shape = reader._f[reader.p_name].shape
        # read data types
        x_dtype = reader._f[reader.x_name].dtype
        y_dtype = reader._f[reader.y_name].dtype
        # no batch size -> return everything
        if not batch_size:
            batch_size = p_shape[0]
        # keep X in memory for resolving pairs quickly
        if sharedx is not None:
            X = sharedx
        else:
            NeighborPairTraintest.__log.debug('Reading X in memory')
            X = reader.get_all_x()
        # default mask is not mask
        if mask_fn is None:
            def mask_fn(*data):
                return data
        batch_beg_end = np.zeros((int(np.ceil(p_shape[0] / batch_size)), 2))
        last = 0
        for row in batch_beg_end:
            row[0] = last
            row[1] = last + batch_size
            last = row[1]
        batch_beg_end = batch_beg_end.astype(int)
        NeighborPairTraintest.__log.debug('Generator ready')

        def example_generator_fn():
            # generator function yielding data
            epoch = 0
            batch_idx = 0
            while True:
                if batch_idx == len(batch_beg_end):
                    batch_idx = 0
                    epoch += 1
                    if shuffle:
                        np.random.shuffle(batch_beg_end)
                    # Traintest.__log.debug('EPOCH %i (caller: %s)', epoch,
                    #                      inspect.stack()[1].function)
                beg_idx, end_idx = batch_beg_end[batch_idx]
                pairs, y = reader.get_py(beg_idx, end_idx)
                x1 = X[pairs[:, 0]]
                x2 = X[pairs[:, 1]]
                if augment_fn is not None:
                    tmp_x1 = list()
                    tmp_x2 = list()
                    tmp_y = list()
                    for i in range(augment_scale):
                        tmp_x1.append(augment_fn(
                            x1, **augment_kwargs))
                        tmp_x2.append(augment_fn(
                            x2, **augment_kwargs))
                        tmp_y.append(y)
                    x1 = np.vstack(tmp_x1)
                    x2 = np.vstack(tmp_x2)
                    y = np.vstack(tmp_y)
                x1, x2, y = mask_fn(x1, x2, y)
                if replace_nan is not None:
                    x1[np.where(np.isnan(x1))] = replace_nan
                    x2[np.where(np.isnan(x2))] = replace_nan
                if only_x:
                    yield [x1, x2]
                else:
                    yield [x1, x2], y
                batch_idx += 1

        pair_shape = (p_shape[0] * augment_scale, x_shape[1])
        shapes = (pair_shape, pair_shape, y_shape)
        dtypes = (x_dtype, x_dtype, y_dtype)
        return shapes, dtypes, example_generator_fn
