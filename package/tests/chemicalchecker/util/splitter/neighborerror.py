"""Splitter on Neighbor error."""
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
    """NeighborErrorTraintest class."""

    def __init__(self, hdf5_file, split, nr_neig=10, replace_nan=None):
        """Initialize a NeighborErrorTraintest instance.

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
        splits = splits.round().astype(int)
        return np.split(idxs, splits)

    @staticmethod
    def create(to_predict, out_file, predict_fn, subsample_fn, max_x=10000,
               split_names=['train', 'test'], split_fractions=[.8, .2],
               suffix='eval', x_dtype=float, y_dtype=float):
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
        NeighborErrorTraintest.__log.debug(
            "{:<20} shape: {:>10}".format("input to_predict",
                                          to_predict))
        # train test validation splits
        if len(split_names) != len(split_fractions):
            raise Exception("Split names and fraction should be same amount.")

        # generate predictions and save coverage as X
        with h5py.File(to_predict, "r") as features:
            tot_x = features['x'].shape[0]
            tot_feat = features['x'].shape[1]
            X = np.zeros((max_x, int(tot_feat / 128)))
            Y = np.zeros((max_x, 1))
            # prepare X and Y in chunks
            chunk_size = int(np.floor(tot_x / 100))
            reached_max = False
            for i in range(0, int(np.ceil(max_x / tot_x))):
                for idx in tqdm(range(0, tot_x, chunk_size), desc='Preds'):
                    # check if enought
                    if reached_max:
                        break
                    # define source chunk
                    src_start = idx
                    src_end = idx + chunk_size
                    if src_end > tot_x:
                        src_end = tot_x
                    # define destination chunk
                    dst_start = src_start + (int(tot_x) * i)
                    dst_end = src_end + (tot_x * i)
                    if dst_end > max_x:
                        dst_end = max_x
                        reached_max = True
                        src_end = dst_end - (int(tot_x) * i)
                    src_chunk = slice(src_start, src_end)
                    dst_chunk = slice(dst_start, dst_end)
                    # get only-self and not-self predictions
                    feat = features['x'][src_chunk]
                    feat_onlyself = subsample_fn(feat, p_only_self=1.0)
                    preds_onlyself = predict_fn(feat_onlyself)
                    feat_notself = subsample_fn(feat)
                    preds_noself = predict_fn(feat_notself)
                    # the error is only-self vs not-self predictions
                    delta = preds_onlyself - preds_noself
                    log_mse = np.log10(1e-6 + np.mean((delta**2), axis=1))
                    Y[dst_chunk] = np.expand_dims(log_mse, 1)
                    # the X is the dataset presence in the not-self
                    presence = ~np.isnan(feat_notself[:, ::128])
                    X[dst_chunk] = presence.astype(int)

        # split chunks, get indeces of chunks for each split
        split_idxs = NeighborErrorTraintest.get_split_indeces(
            X.shape[0], split_fractions)

        # create dataset
        NeighborErrorTraintest.__log.info('Traintest saving to %s', out_file)
        with h5py.File(out_file, "w") as fh:
            # for each split
            for split_idxs, split_name in zip(split_idxs, split_names):

                NeighborErrorTraintest.__log.info(
                    'X shape %s', X[split_idxs].shape)
                NeighborErrorTraintest.__log.info(
                    'Y shape %s', Y[split_idxs].shape)
                # save to h5
                xs_name = "x_%s" % split_name
                ys_name = "y_%s" % split_name
                fh.create_dataset(xs_name, data=X[split_idxs],
                                  dtype=x_dtype)
                fh.create_dataset(ys_name, data=Y[split_idxs],
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
