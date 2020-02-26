import os
import h5py
import itertools
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import robust_scale

from chemicalchecker.util import logged
from chemicalchecker.util.remove_near_duplicates import RNDuplicates


@logged
class NeighborErrorTraintest(object):
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
    def create(to_predict, out_file, predict_fn, subsample_fn,
               neigbors_matrix=None, shuffle=True, max_x=10000,
               split_names=['train', 'test'], split_fractions=[.8, .2],
               suffix='eval', x_dtype=np.float32, y_dtype=np.float32):
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

        def row_wise_correlation(X, Y):
            var1 = (X.T - np.mean(X, axis=1)).T
            var2 = (Y.T - np.mean(Y, axis=1)).T
            cov = np.mean(var1 * var2, axis=1)
            return cov / (np.std(X, axis=1) * np.std(Y, axis=1))

        NeighborErrorTraintest.__log.debug(
            "{:<20} shape: {:>10}".format("input to_predict",
                                          to_predict))
        # train test validation splits
        if len(split_names) != len(split_fractions):
            raise Exception("Split names and fraction should be same amount.")

        # generate predictions and save coverage as X
        with h5py.File(to_predict, "r") as features:
            tot_x = features['x'].shape[0]
            if len(neigbors_matrix) != tot_x:
                raise Exception("neigbors_matrix should be same length as X.")
            tot_x = min(max_x, tot_x)
            tot_feat = features['x'].shape[1]
            chunk_size = int(np.floor(tot_x / 100))
            X = np.zeros((tot_x, int(tot_feat / 128)))
            Y = np.zeros((tot_x, 1))
            preds_noself = np.zeros((tot_x, 128), dtype=np.float32)
            preds_all = np.zeros((tot_x, 128), dtype=np.float32)

            # read input in chunks
            for idx in tqdm(range(0, tot_x, chunk_size), desc='Predicting'):
                src_chunk = slice(idx, idx + chunk_size)
                if idx + chunk_size > tot_x:
                    src_chunk = slice(idx, tot_x)
                feat = features['x'][src_chunk]
                feat_all = subsample_fn(feat)
                preds_all[src_chunk] = predict_fn(feat_all)
                feat_noself = subsample_fn(
                    feat, p_only_self=0.0, p_self=0.0)
                preds_noself[src_chunk] = predict_fn(feat_noself)
                X[src_chunk] = (
                    ~np.isnan(feat_all[:, ::128])).astype(int)
                print([int(x) for x in X[src_chunk][0]])
            Y = np.expand_dims(row_wise_correlation(
                robust_scale(preds_all),
                robust_scale(preds_noself)), 1)

        # reduce redundancy, keep full-ref mapping
        rnd = RNDuplicates(cpu=10)
        _, ref_matrix, full_ref_map = rnd.remove(neigbors_matrix[:max_x])
        ref_full_map = dict()
        for key, value in full_ref_map.items():
            ref_full_map.setdefault(value, list()).append(key)
        full_refid_map = dict(
            zip(rnd.final_ids, np.arange(len(rnd.final_ids))))
        refid_full_map = {full_refid_map[k]: v
                          for k, v in ref_full_map.items()}

        # split chunks, get indeces of chunks for each split
        chunk_size = np.floor(ref_matrix.shape[0] / 100)
        split_chunk_idx = NeighborErrorTraintest.get_split_indeces(
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
                NeighborErrorTraintest.__log.debug(
                    "writing src: %s  to dst: %s" % (src_slice, dst_slice))
                ref_src_chunk = ref_idxs[src_chunk]
                ref_dst_chunk = ref_idxs[dst_chunk]
                for src_id, dst_id in zip(ref_src_chunk, ref_dst_chunk):
                    split_ref_map[split_name][dst_id] = src_id
                nr_matrix[split_name][dst_chunk] = ref_matrix[src_chunk]
            NeighborErrorTraintest.__log.debug(
                "nr_matrix %s %s", split_name, nr_matrix[split_name].shape)

        # create dataset
        NeighborErrorTraintest.__log.info('Traintest saving to %s', out_file)
        with h5py.File(out_file, "w") as fh:
            # for each split
            for split_name in split_names:
                # map from split to ref to full expanding to all redundant
                ref_idxs = split_ref_map[split_name].values()
                ys = [Y[x] for x in ref_idxs]
                full_idxs = [refid_full_map[x] for x in ref_idxs]
                full_idxs_flat = list()
                full_y = list()
                assert(len(full_idxs) == len(ys))
                for sublist, y in zip(full_idxs, ys):
                    for item in sublist:
                        full_idxs_flat.append(item)
                        full_y.append(y)
                assert(len(full_idxs_flat) == len(full_y))
                full_idxs_flat = np.array(full_idxs_flat)
                full_y = np.expand_dims(np.array(full_y), 1)

                shuffle_idxs = np.arange(len(full_idxs_flat))
                if shuffle:
                    np.random.shuffle(shuffle_idxs)

                NeighborErrorTraintest.__log.info(
                    'X shape %s', X[full_idxs_flat[shuffle]][0].shape)
                NeighborErrorTraintest.__log.info(
                    'Y shape %s', full_y[shuffle][0].shape)
                # save to h5
                xs_name = "x_%s" % split_name
                ys_name = "y_%s" % split_name
                fh.create_dataset(xs_name, data=X[full_idxs_flat[shuffle]][0],
                                  dtype=x_dtype)
                fh.create_dataset(ys_name, data=full_y[shuffle][0],
                                  dtype=y_dtype)

        NeighborErrorTraintest.__log.info(
            'NeighborErrorTraintest saved to %s', out_file)

    @staticmethod
    def generator_fn(file_name, split, batch_size=None,
                     replace_nan=None, augment_scale=1,
                     augment_fn=None, augment_kwargs={},
                     mask_fn=None, shuffle=True,
                     return_on_epoch=True,
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
        reader = NeighborErrorTraintest(file_name, split)
        reader.open()
        # read shapes
        x_shape = reader._f[reader.x_name].shape
        y_shape = reader._f[reader.y_name].shape
        # read data types
        x_dtype = reader._f[reader.x_name].dtype
        y_dtype = reader._f[reader.y_name].dtype
        # no batch size -> return everything
        if not batch_size:
            batch_size = x_shape[0]
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
        NeighborErrorTraintest.__log.debug('Generator ready')

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
                    if return_on_epoch:
                        return
                # print('EPOCH %i' % epoch)
                # print('batch_idx %i' % batch_idx)
                beg_idx, end_idx = batch_beg_end[batch_idx]
                y = reader.get_y(beg_idx, end_idx)
                x = reader.get_x(beg_idx, end_idx)
                if augment_fn is not None:
                    tmp_x = list()
                    tmp_y = list()
                    for i in range(augment_scale):
                        tmp_x.append(augment_fn(
                            x, **augment_kwargs))
                        tmp_y.append(y)
                    x1 = np.vstack(tmp_x)
                    y = np.hstack(tmp_y)
                x = mask_fn(x)
                if replace_nan is not None:
                    x[np.where(np.isnan(x))] = replace_nan
                # print(x1.shape, x2.shape, x3.shape, y.shape)
                yield x1, y
                batch_idx += 1

        x_shape = (x_shape[0] * augment_scale, x_shape[1])
        y_shape = (y_shape[0] * augment_scale, y_shape[1])
        shapes = (x_shape, y_shape)
        dtypes = (x_dtype, y_dtype)
        print('SHAPES', shapes)
        return shapes, dtypes, example_generator_fn
