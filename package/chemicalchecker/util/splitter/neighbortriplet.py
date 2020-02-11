import os
import h5py
import pickle
import itertools
import numpy as np
from tqdm import tqdm
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler

from chemicalchecker.util import logged
from chemicalchecker.util.remove_near_duplicates import RNDuplicates


@logged
class NeighborTripletTraintest(object):
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
        splits = splits.round().astype(np.int)
        return np.split(idxs, splits)

    @staticmethod
    def create(X, out_file, neigbors_matrix=None, F=1000, T=100,
               mean_center_x=True, shuffle=True,
               check_distances=True,
               split_names=['train', 'test'], split_fractions=[.8, .2],
               x_dtype=np.float32, y_dtype=np.float32, debug_test=False):
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
            y_dtype(type): numpy data type for Y (np.float32 for regression,
                np.int32 for classification.
        """
        try:
            import faiss
        except ImportError as err:
            raise err
        NeighborTripletTraintest.__log.debug(
            "{:<20} shape: {:>10}".format("input X", str(X.shape)))
        # train test validation splits
        if len(split_names) != len(split_fractions):
            raise Exception("Split names and fraction should be same amount.")

        # F is 10% of the size of NN shape (1000 maximum)
        F = int(min([F, 0.1 * len(neigbors_matrix)]))

        # T is 1% of the size of NN shape (100 maximum)
        T = int(min([T, 0.01 * len(neigbors_matrix)]))

        # the neigbors_matrix is optional
        if len(neigbors_matrix) != len(X):
            raise Exception("neigbors_matrix shuold be same length as X.")

        # reduce redundancy, keep full-ref mapping
        rnd = RNDuplicates(cpu=10)
        _, ref_matrix, full_ref_map = rnd.remove(neigbors_matrix)
        ref_full_map = np.array(rnd.final_ids)
        rows = ref_matrix.shape[0]

        # split chunks, get indeces of chunks for each split
        chunk_size = np.floor(rows / 100)
        split_chunk_idx = NeighborTripletTraintest.get_split_indeces(
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
                NeighborTripletTraintest.__log.debug(
                    "writing src: %s  to dst: %s" % (src_slice, dst_slice))
                ref_src_chunk = ref_idxs[src_chunk]
                ref_dst_chunk = ref_idxs[dst_chunk]
                for src_id, dst_id in zip(ref_src_chunk, ref_dst_chunk):
                    split_ref_map[split_name][dst_id] = src_id
                nr_matrix[split_name][dst_chunk] = ref_matrix[src_chunk]
            NeighborTripletTraintest.__log.debug(
                "nr_matrix %s %s", split_name, nr_matrix[split_name].shape)

        # for each split generate NN
        NN = dict()
        for split_name in split_names:
            # create faiss index
            NN[split_name] = faiss.IndexFlatL2(nr_matrix[split_name].shape[1])
            # add data
            NN[split_name].add(nr_matrix[split_name])

        # mean centering columns
        if mean_center_x:
            scaler = RobustScaler()
            X = scaler.fit_transform(X)
            scaler_file = os.path.join(os.path.split(out_file)[0],
                                       'scaler.pkl')
            pickle.dump(scaler, open(scaler_file, 'wb'))

        # create dataset
        NeighborTripletTraintest.__log.info('%s', F)
        NeighborTripletTraintest.__log.info('%s', T)
        NeighborTripletTraintest.__log.info('Traintest saving to %s', out_file)
        with h5py.File(out_file, "w") as fh:
            fh.create_dataset('x', data=X)
            # for each split combo generate triplets where [anchor, positive,
            # negative]
            combos = itertools.combinations_with_replacement(split_names, 2)
            #combos = [('train', 'train'), ('train', 'test'), ('test', 'test')]

            dists = []

            for split1, split2 in combos:
                # remove self neighbors when splits are the same
                if split1 == split2:
                    # search NN
                    _, neig_idxs = NN[split1].search(
                        nr_matrix[split2], int(F + 1))
                    # the nearest neig between same groups is the molecule
                    # itself
                    assert(all(neig_idxs[:, 0] ==
                               np.arange(0, len(neig_idxs))))
                    neig_idxs = neig_idxs[:, 1:]
                else:
                    _, neig_idxs = NN[split1].search(
                        nr_matrix[split2], F)

                # get probabilities for T
                t_prob = ((np.arange(T + 1)[::-1]) /
                          np.sum(np.arange(T + 1)))[:T]
                assert(sum(t_prob) == 1.0)

                anchors_lst = list()

                easy_p_lst = list()
                easy_n_lst = list()

                medium_p_lst = list()
                medium_n_lst = list()

                hard_p_lst = list()
                hard_n_lst = list()

                num_triplets = 10

                # Idx comes from split2, all else comes from split1
                for idx, row in enumerate(neig_idxs):

                    no_F = set(range(len(neig_idxs))) - set(row)

                    # avoid fetching itself as negative!
                    if split1 == split2:
                        no_F = no_F - set([idx])

                    no_F = list(no_F)

                    p_indexes = np.random.choice(
                        T, num_triplets, replace=True, p=t_prob)

                    anchors = [idx] * num_triplets
                    anchors_lst.extend(anchors)

                    positives = neig_idxs[idx][:T][p_indexes]
                    easy_p_lst.extend(positives)
                    medium_p_lst.extend(positives)
                    hard_p_lst.extend(positives)

                    easy_n = np.random.choice(
                        no_F, num_triplets, replace=False)
                    easy_n_lst.extend(easy_n)

                    medium_n = np.random.choice(
                        neig_idxs[idx][T:], num_triplets, replace=False)
                    medium_n_lst.extend(medium_n)

                    hard_n = [np.random.choice(neig_idxs[idx][p_i:T], 1, p=t_prob[
                                               p_i:] / sum(t_prob[p_i:]))[0] for p_i in p_indexes]
                    hard_n_lst.extend(hard_n)

                anchors_lst = ref_full_map[
                    np.array([split_ref_map[split1][x] for x in anchors_lst])]
                print(split_ref_map)
                easy_p_lst = ref_full_map[
                    np.array([split_ref_map[split2][x] for x in easy_p_lst])]
                easy_n_lst = ref_full_map[
                    np.array([split_ref_map[split2][x] for x in easy_n_lst])]

                medium_p_lst = ref_full_map[
                    np.array([split_ref_map[split2][x] for x in medium_p_lst])]
                medium_n_lst = ref_full_map[
                    np.array([split_ref_map[split2][x] for x in medium_n_lst])]

                hard_p_lst = ref_full_map[
                    np.array([split_ref_map[split2][x] for x in hard_p_lst])]
                hard_n_lst = ref_full_map[
                    np.array([split_ref_map[split2][x] for x in hard_n_lst])]

                easy_triplets = np.vstack(
                    (anchors_lst, easy_p_lst, easy_n_lst)).T
                medium_triplets = np.vstack(
                    (anchors_lst, medium_p_lst, medium_n_lst)).T
                hard_triplets = np.vstack(
                    (anchors_lst, hard_p_lst, hard_n_lst)).T

                if check_distances:
                    import matplotlib.pyplot as plt
                    import seaborn as sns

                    def subplot(triplets_lst, tiplet_name, limit=1000):
                        plt.title('Distances %s' % tiplet_name)
                        anchors = neigbors_matrix[triplets_lst[:, 0][:limit]]
                        positives = neigbors_matrix[triplets_lst[:, 1][:limit]]
                        negatives = neigbors_matrix[triplets_lst[:, 2][:limit]]

                        dis_ap = [np.linalg.norm(a - p)
                                  for a, p in zip(anchors, positives)]
                        dis_an = [np.linalg.norm(a - n)
                                  for a, n in zip(anchors, negatives)]

                        row.append([dis_ap, dis_an])
                    dists.append(row)

                # shuffling
                triplets = np.hstack(
                    (easy_triplets, medium_triplets, hard_triplets))
                shuffle_idxs = np.arange(triplets.shape[1])
                if shuffle:
                    np.random.shuffle(shuffle_idxs)

                # save to h5
                ds_name = "p_%s_%s" % (split1, split2)
                NeighborTripletTraintest.__log.info(
                    'writing %s %s %s %s', ds_name, easy_triplets.shape, medium_triplets.shape, hard_triplets.shape)
                fh.create_dataset(ds_name, data=triplets[shuffle_idxs])

            if check_distances:
                import matplotlib.pyplot as plt
                import seaborn as sns

                plt.figure(figsize=(10, 10))
                plot_file = os.path.join(
                    os.path.split(out_file)[0], 'distances.png')
                i = 0
                j = 0
                categs = ['easy', 'medium', 'hard']
                combos = ['train_train', 'train_test', 'test_test']
                for row in dists:
                    split = combos[i]
                    for pair in row:
                        plt.subplot(3, 3, j + 1)
                        plt.title('%s %s' % (split, categs[j % 3]))
                        sns.distplot(pair[0], label='AP')
                        sns.distplot(pair[1], label='AN')
                        plt.legend()
                        plt.xlim(0, 5)
                        j += 1
                    i += 1

                plt.savefig(plot_file)
                plt.close()

        NeighborTripletTraintest.__log.info(
            'NeighborTripletTraintest saved to %s', out_file)

    @staticmethod
    def generator_fn(file_name, split, batch_size=None,
                     replace_nan=None, augment_scale=1,
                     augment_fn=None, augment_kwargs={},
                     mask_fn=None, shuffle=True,
                     sharedx=None):
        """Return the generator function that we can query for batches.

        file_name(str): The H5 generated via `create`
        split(str): One of 'train_train', 'train_test', or 'test_test'
        batch_size(int): Size of a batch of data.
        replace_nan(bool): Value used for NaN replacement.
        augment_scale(int): Augment the train size by this factor.
        augment_fn(func): Function to augment data.
        augment_kwargs(dict): Parameters for the aument functions.
        """
        reader = NeighborTripletTraintest(file_name, split)
        reader.open()
        # read shapes
        x_shape = reader._f[reader.x_name].shape
        p_shape = reader._f[reader.p_name].shape
        # read data types
        x_dtype = reader._f[reader.x_name].dtype
        # no batch size -> return everything
        if not batch_size:
            batch_size = p_shape[0]
        # keep X in memory for resolving pairs quickly
        if sharedx is not None:
            X = sharedx
        else:
            NeighborTripletTraintest.__log.debug('Reading X in memory')
            X = reader.get_all_x()
        # default mask is not mask
        if mask_fn is None:
            def mask_fn(*data):
                return data
        batch_beg_end = np.zeros((int(np.ceil(x_shape[0] / batch_size)), 2))
        last = 0
        for row in batch_beg_end:
            row[0] = last
            row[1] = last + batch_size
            last = row[1]
        batch_beg_end = batch_beg_end.astype(int)
        NeighborTripletTraintest.__log.debug('Generator ready')

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
                pairs = reader.get_p(beg_idx, end_idx)
                # @PAU FIX generator
                x1 = X[pairs[:, 0]]
                x2 = X[pairs[:, 1]]
                x3 = X[pairs[:, 2]]
                if augment_fn is not None:
                    tmp_x1 = list()
                    tmp_x2 = list()
                    tmp_x3 = list()
                    for i in range(augment_scale):
                        tmp_x1.append(augment_fn(
                            x1, **augment_kwargs))
                        tmp_x2.append(augment_fn(
                            x2, **augment_kwargs))
                        tmp_x3.append(augment_fn(
                            x3, **augment_kwargs))
                    x1 = np.vstack(tmp_x1)
                    x2 = np.vstack(tmp_x2)
                    x3 = np.vstack(tmp_x3)
                x1, x2, x3 = mask_fn(x1, x2, x3)
                if replace_nan is not None:
                    x1[np.where(np.isnan(x1))] = replace_nan
                    x2[np.where(np.isnan(x2))] = replace_nan
                    x3[np.where(np.isnan(x3))] = replace_nan
                yield [x1, x2, x3]
                batch_idx += 1

        pair_shape = (p_shape[0] * augment_scale, x_shape[1])
        shapes = (pair_shape, pair_shape, pair_shape)
        dtypes = (x_dtype, x_dtype, x_dtype)
        return shapes, dtypes, example_generator_fn
