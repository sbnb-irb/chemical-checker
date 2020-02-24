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
            self.t_name = "t"
            self.y_name = "y"
        else:
            self.t_name = "t_%s" % split
            self.y_name = "y_%s" % split
            available_splits = self.get_split_names()
            if split not in available_splits:
                raise Exception("Split '%s' not found in %s!" %
                                (split, str(available_splits)))

    def get_ty_shapes(self):
        """Return the shpaes of X an Y."""
        self.open()
        t_shape = self._f[self.t_name].shape
        y_shape = self._f[self.y_name].shape
        self.close()
        return t_shape, y_shape

    def get_xy_shapes(self):
        """Return the shpaes of X an Y."""
        self.open()
        x_shape = self._f[self.x_name].shape
        self.close()
        return x_shape

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
        # self.__log.info("HDF5 open %s", self._file)

    def close(self):
        """Close the HDF5."""
        try:
            self._f.close()
            # self.__log.info("HDF5 close %s", self._file)
        except AttributeError:
            self.__log.error('HDF5 file is not open yet.')

    def get_t(self, beg_idx, end_idx):
        """Get a batch of X."""
        features = self._f[self.t_name][beg_idx: end_idx]
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
        features = self._f[self.t_name][:]
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
    def precomputed_triplets(X, triplets, out_file,
                             mean_center_x=True,
                             shuffle=True,
                             split_names=['train', 'test'],
                             split_fractions=[.8, .2],
                             suffix='eval',
                             x_dtype=np.float32, y_dtype=np.float32):

        # mean centering columns
        if mean_center_x:
            scaler = RobustScaler()
            X = scaler.fit_transform(X)
            if suffix is None:
                scaler_file = os.path.join(os.path.split(out_file)[0],
                                           'scaler.pkl')
            else:
                scaler_file = os.path.join(os.path.split(out_file)[0],
                                           'scaler_%s.pkl' % suffix)
            pickle.dump(scaler, open(scaler_file, 'wb'))

        # shuffling
        shuffle_idxs = np.arange(triplets.shape[0])
        if shuffle:
            np.random.shuffle(shuffle_idxs)
        triplets = np.array(triplets)[shuffle_idxs]

        # do traintest split for triplets (np.unique of indeces)
        split_idxs = NeighborTripletTraintest.get_split_indeces(
            len(triplets), split_fractions)
        '''
        split_idxs = dict(zip(split_names, split_idxs))
        # find triplets having test-test train-trani and train-test
        combos = itertools.combinations_with_replacement(split_names, 2)
        for split1, split2 in combos:
            split1_idxs = split_idxs[split1]
            split2_idxs = split_idxs[split2]
            if split1 != split2:
                split1_mask = ~np.all(np.isin(triplets, split1_idxs), axis=1)
                split2_mask = ~np.all(np.isin(triplets, split2_idxs), axis=1)
                combo_mask = np.logical_and(split1_mask, split2_mask)
            else:
                combo_mask = np.all(np.isin(triplets, split1_idxs), axis=1)
            print(split1, split2, np.count_nonzero(combo_mask),
            combo_mask.shape)
        '''
        # create dataset
        NeighborTripletTraintest.__log.info('Traintest saving to %s', out_file)
        with h5py.File(out_file, "w") as fh:
            fh.create_dataset('x', data=X, dtype=x_dtype)
            for split_name, split_idx in zip(split_names, split_idxs):
                split_triplets = triplets[split_idx]
                fh.create_dataset('t_%s' % split_name,
                                  data=split_triplets)
                fh.create_dataset('y_%s' % split_name,
                                  data=np.zeros((len(split_triplets), 1)))
        NeighborTripletTraintest.__log.info(
            'NeighborTripletTraintest saved to %s', out_file)

    @staticmethod
    def create(X, out_file, neigbors_sign, f_per=0.1, t_per=0.01,
               mean_center_x=True, shuffle=True,
               check_distances=True,
               split_names=['train', 'test'], split_fractions=[.8, .2],
               suffix='eval',
               x_dtype=np.float32, y_dtype=np.float32, num_triplets=1e6, limit=100000):
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
            from chemicalchecker.core.signature_data import DataSignature
        except ImportError as err:
            raise err
        # train test validation splits
        if len(split_names) != len(split_fractions):
            raise Exception("Split names and fraction should be same amount.")

        # Limit total X size
        neigbors_matrix = neigbors_sign[:]
        shuffle_idx = np.arange(neigbors_matrix.shape[0])
        np.random.shuffle(shuffle_idx)

        neigbors_matrix = neigbors_matrix[shuffle_idx[:limit]]
        X = X.get_h5_dataset('x')[shuffle_idx[:limit]]
        X_inks = np.array(neigbors_sign.keys)[shuffle_idx[:limit]]

        if len(neigbors_matrix) != len(X):
            raise Exception("neigbors_matrix should be same length as X.")

        NeighborTripletTraintest.__log.debug(
            "{:<20} shape: {:>10}".format("input X", str(X.shape)))

        # reduce redundancy, keep full-ref mapping
        rnd = RNDuplicates(cpu=10)
        _, ref_matrix, full_ref_map = rnd.remove(
            neigbors_matrix.astype(np.float32))
        ref_full_map = dict()
        for key, value in full_ref_map.items():
            ref_full_map.setdefault(value, list()).append(key)
        full_refid_map = dict(
            zip(rnd.final_ids, np.arange(len(rnd.final_ids))))
        refid_full_map = {full_refid_map[k]: v
                          for k, v in ref_full_map.items()}
        # ref_full_all_map = np.array(rnd.final_ids)

        # Set triplet_factor
        triplet_per_mol = int(
            np.ceil((num_triplets / 3) / ref_matrix.shape[0]))

        # split chunks, get indeces of chunks for each split
        chunk_size = np.floor(ref_matrix.shape[0] / 100)
        split_chunk_idx = NeighborTripletTraintest.get_split_indeces(
            int(np.floor(ref_matrix.shape[0] / chunk_size)) + 1,
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
            if suffix is None:
                scaler_file = os.path.join(os.path.split(out_file)[0],
                                           'scaler.pkl')
            else:
                scaler_file = os.path.join(os.path.split(out_file)[0],
                                           'scaler_%s.pkl' % suffix)
            pickle.dump(scaler, open(scaler_file, 'wb'))

        # create dataset
        NeighborTripletTraintest.__log.info('Traintest saving to %s', out_file)
        combo_dists = dict()
        with h5py.File(out_file, "w") as fh:
            fh.create_dataset('x', data=X)
            # for each split combo generate triplets where [anchor, positive,
            # negative]
            combos = itertools.combinations_with_replacement(split_names, 2)
            for split1, split2 in combos:
                combo = '_'.join([split1, split2])
                # define F and T according to the split that is being used
                F = np.clip(f_per * nr_matrix[split2].shape[0], 100, 1000)
                F = int(min(F, (nr_matrix[split2].shape[0] - 1)))
                T = int(np.clip(t_per * nr_matrix[split2].shape[0], 5, 100))
                NeighborTripletTraintest.__log.info("F and T: %s %s" % (F, T))
                assert(T < F)

                # remove self neighbors when splits are the same
                if split1 == split2:
                    # search NN
                    _, neig_idxs = NN[split1].search(
                        nr_matrix[split2], F + 1)
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
                assert(sum(t_prob) > 0.99)

                # save list of split indeces
                anchors_split = list()
                easy_p_split = list()
                easy_n_split = list()
                medi_p_split = list()
                medi_n_split = list()
                hard_p_split = list()
                hard_n_split = list()

                # idx refere split2, all else to split1
                for idx, row in enumerate(neig_idxs):
                    # anchors are repeated num_triplets times
                    anchors = [idx] * triplet_per_mol
                    anchors_split.extend(anchors)
                    # positives are sampled from top T NNs for each category
                    p_indexes = np.random.choice(
                        T, triplet_per_mol, replace=True, p=t_prob)
                    positives = neig_idxs[idx][:T][p_indexes]
                    easy_p_split.extend(positives)
                    medi_p_split.extend(positives)
                    hard_p_split.extend(positives)

                    # easy negatives are sampled from outside NN
                    no_nn = set(range(neig_idxs.shape[0])) - set(row)
                    # avoid fetching itself as negative!
                    if split1 == split2:
                        no_nn = no_nn - set([idx])
                    no_nn = list(no_nn)
                    easy_n = np.random.choice(
                        no_nn, triplet_per_mol, replace=True)
                    easy_n_split.extend(easy_n)

                    # medium negatives are from F (in NN but not T)
                    medi_n = np.random.choice(
                        neig_idxs[idx][T:], triplet_per_mol, replace=True)
                    medi_n_split.extend(medi_n)

                    # hard negatives are from T
                    hard_n = [np.random.choice(
                        neig_idxs[idx][p_i + 1:T + 1], 1,
                        p=(t_prob[p_i:] / sum(t_prob[p_i:]))[::-1])[0]
                        for p_i in p_indexes]
                    hard_n_split.extend(hard_n)

                # get reference ids
                anchors_ref = [split_ref_map[split2][x] for x in anchors_split]
                easy_p_ref = [split_ref_map[split1][x] for x in easy_p_split]
                easy_n_ref = [split_ref_map[split1][x] for x in easy_n_split]
                medi_p_ref = [split_ref_map[split1][x] for x in medi_p_split]
                medi_n_ref = [split_ref_map[split1][x] for x in medi_n_split]
                hard_p_ref = [split_ref_map[split1][x] for x in hard_p_split]
                hard_n_ref = [split_ref_map[split1][x] for x in hard_n_split]

                # choose random from full analogs
                anchors_full = np.array(
                    [np.random.choice(refid_full_map[x]) for x in anchors_ref])
                easy_p_full = np.array(
                    [np.random.choice(refid_full_map[x]) for x in easy_p_ref])
                easy_n_full = np.array(
                    [np.random.choice(refid_full_map[x]) for x in easy_n_ref])
                medi_p_full = np.array(
                    [np.random.choice(refid_full_map[x]) for x in medi_p_ref])
                medi_n_full = np.array(
                    [np.random.choice(refid_full_map[x]) for x in medi_n_ref])
                hard_p_full = np.array(
                    [np.random.choice(refid_full_map[x]) for x in hard_p_ref])
                hard_n_full = np.array(
                    [np.random.choice(refid_full_map[x]) for x in hard_n_ref])

                # stack triplets
                easy_triplets = np.vstack(
                    (anchors_full, easy_p_full, easy_n_full)).T
                medium_triplets = np.vstack(
                    (anchors_full, medi_p_full, medi_n_full)).T
                hard_triplets = np.vstack(
                    (anchors_full, hard_p_full, hard_n_full)).T
                triplets = np.vstack(
                    (easy_triplets, medium_triplets, hard_triplets))
                # stack categories
                y = np.hstack((
                    np.full((easy_triplets.shape[0],), 0),
                    np.full((medium_triplets.shape[0],), 1),
                    np.full((hard_triplets.shape[0],), 2)))

                # shuffling
                shuffle_idxs = np.arange(triplets.shape[0])
                if shuffle:
                    np.random.shuffle(shuffle_idxs)
                unique_ids = np.unique(triplets)
                NeighborTripletTraintest.__log.info(
                    'Using %s molecules in triplets' %
                    len(unique_ids))

                # get inchikeys of test or train molecules
                if split1 == split2:
                    ink_ids = np.array(sorted(unique_ids))
                    split_inks = np.array(np.array(X_inks[ink_ids]),
                                          DataSignature.string_dtype())
                    ds_name = "keys_%s" % split1
                    fh.create_dataset(ds_name, data=split_inks)
                # save to h5
                ds_name = "t_%s_%s" % (split1, split2)
                ys_name = "y_%s_%s" % (split1, split2)
                NeighborTripletTraintest.__log.info(
                    'writing %s %s %s %s %s %s', ds_name, easy_triplets.shape,
                    medium_triplets.shape, hard_triplets.shape, triplets.shape,
                    y.shape)
                fh.create_dataset(ds_name, data=triplets[shuffle_idxs])
                fh.create_dataset(ys_name, data=y[shuffle_idxs])

                if check_distances:
                    import matplotlib.pyplot as plt
                    import seaborn as sns

                    limit = min(10000, len(shuffle_idxs))
                    dists = np.empty((limit, 3))
                    for idx, row in enumerate(shuffle_idxs[:limit]):
                        anchor = neigbors_matrix[triplets[row][0]]
                        positive = neigbors_matrix[triplets[row][1]]
                        negative = neigbors_matrix[triplets[row][2]]
                        category = y[row]

                        dis_ap = euclidean(anchor, positive)
                        dis_an = euclidean(anchor, negative)
                        dists[idx] = [dis_ap, dis_an, category]
                        if (dis_ap > dis_an):
                            NeighborTripletTraintest.__log.warning(
                                'DIST ERROR %s %.2f %.2f %i' %
                                (triplets[row], dis_ap, dis_an, category))
                    assert(len(np.unique(dists[:, 2])) == 3)
                    combo_dists[combo] = dists

        if check_distances:
            fig, axes = plt.subplots(
                3, 3, sharex=True, sharey=False, figsize=(10, 10))
            ax_idx = 0
            cat_names = ['easy', 'medium', 'hard']
            for combo, dists in combo_dists.items():
                for cat_id in [0, 1, 2]:
                    cat_mask = dists[:, 2] == cat_id
                    ax = axes.flatten()[ax_idx]
                    ax.set_title('%s %s' % (combo, cat_names[cat_id]))
                    sns.distplot(dists[cat_mask, 0], label='AP', ax=ax)
                    sns.distplot(dists[cat_mask, 1], label='AN', ax=ax)
                    ax.legend()
                    ax_idx += 1

            if suffix is None:
                plot_file = os.path.join(
                    os.path.split(out_file)[0], 'distances.png')
            else:
                plot_file = os.path.join(
                    os.path.split(out_file)[0], 'distances_%s.png' % suffix)
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
        t_shape = reader._f[reader.t_name].shape
        # read data types
        x_dtype = reader._f[reader.x_name].dtype
        # no batch size -> return everything
        if not batch_size:
            batch_size = t_shape[0]
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
        batch_beg_end = np.zeros((int(np.ceil(t_shape[0] / batch_size)), 2))
        last = 0
        for row in batch_beg_end:
            row[0] = last
            row[1] = last + batch_size
            last = row[1]
        batch_beg_end = batch_beg_end.astype(int)
        for idx, row in enumerate(batch_beg_end):
            beg_idx, end_idx = batch_beg_end[idx]
            pairs = reader.get_t(beg_idx, end_idx)
            # print(beg_idx, end_idx, 'batch_beg_end', idx, pairs.shape)
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
                # print('EPOCH %i' % epoch)
                # print('batch_idx %i' % batch_idx)
                beg_idx, end_idx = batch_beg_end[batch_idx]
                pairs = reader.get_t(beg_idx, end_idx)
                y = reader.get_y(beg_idx, end_idx)
                x1 = X[pairs[:, 0]]
                x2 = X[pairs[:, 1]]
                x3 = X[pairs[:, 2]]
                if augment_fn is not None:
                    tmp_x1 = list()
                    tmp_x2 = list()
                    tmp_x3 = list()
                    tmp_y = list()
                    for i in range(augment_scale):
                        tmp_x1.append(augment_fn(
                            x1, **augment_kwargs))
                        tmp_x2.append(augment_fn(
                            x2, **augment_kwargs))
                        tmp_x3.append(augment_fn(
                            x3, **augment_kwargs))
                        tmp_y.append(y)
                    x1 = np.vstack(tmp_x1)
                    x2 = np.vstack(tmp_x2)
                    x3 = np.vstack(tmp_x3)
                    y = np.hstack(tmp_y)
                x1, x2, x3 = mask_fn(x1, x2, x3)
                if replace_nan is not None:
                    x1[np.where(np.isnan(x1))] = replace_nan
                    x2[np.where(np.isnan(x2))] = replace_nan
                    x3[np.where(np.isnan(x3))] = replace_nan
                # print(x1.shape, x2.shape, x3.shape, y.shape)
                yield [x1, x2, x3], y
                batch_idx += 1

        pair_shape = (t_shape[0] * augment_scale, x_shape[1])
        shapes = (pair_shape, pair_shape, pair_shape)
        dtypes = (x_dtype, x_dtype, x_dtype)
        return shapes, dtypes, example_generator_fn
