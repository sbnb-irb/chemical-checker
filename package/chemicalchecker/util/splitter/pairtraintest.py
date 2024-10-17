"""Splitter on pairs."""
import h5py
import numpy as np

from chemicalchecker.util import logged


@logged
class PairTraintest(object):
    """PairTraintest class."""

    def __init__(self, hdf5_file, split, nr_neig=10, replace_nan=None):
        """Initialize a PairTraintest class.

        We assume the file is containing diffrent splits.
        e.g. "x_train", "y_train", "x_test", ...
        """
        self._file = hdf5_file
        self._f = None
        self.replace_nan = replace_nan
        self.nr_neig = nr_neig
        self.x1_name = "x1"
        self.x2_name = "x2"
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

    def get_pos_neg(self):
        y = self.get_all_y()
        return len(y[y == 1]), len(y[y == 0])

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
        x1_shape = self._f[self.x1_name].shape
        x2_shape = self._f[self.x2_name].shape
        y_shape = self._f[self.y_name].shape
        self.close()
        return x1_shape, x2_shape, y_shape

    def get_split_names(self):
        """Return the name of the splits."""
        self.open()
        if "split_names" in self._f:
            split_names = [a.decode() for a in self._f["split_names"]]
        else:
            split_names = ['train', 'test']
            self.__log.info("Using default split names %s" % split_names)
        self.close()
        return split_names

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

    def get_all_x1(self):
        """Get full X."""
        features = self._f[self.x1_name][:]
        # handle NaNs
        if self.replace_nan is not None:
            features[np.where(np.isnan(features))] = self.replace_nan
        return features

    def get_all_x2(self):
        """Get full X."""
        features = self._f[self.x2_name][:]
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
    def create(X1, X2, pairs, split_names, out_file,
               mean_center_x=True, shuffle=True):
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
                int32 for classification.
        """
        PairTraintest.__log.debug(
            "{:<20} shape: {:>10}".format("input X1", str(X1.shape)))
        PairTraintest.__log.debug(
            "{:<20} shape: {:>10}".format("input X2", str(X2.shape)))
        # train test validation splits
        if len(split_names) != len(pairs):
            raise Exception("Split names and set of pairs must be same nr.")
        for name, Y in zip(split_names, pairs):
            PairTraintest.__log.debug(
                "{:<20} shape: {:>10}".format(name, str(Y.shape)))

        # create dataset
        PairTraintest.__log.info('Traintest saving to %s', out_file)
        with h5py.File(out_file, "w") as fh:
            fh.create_dataset('x1', data=X1)
            fh.create_dataset('x2', data=X2)
            #fh.create_dataset('split_names', data=split_names)

            for name, PY in zip(split_names, pairs):
                # shuffling
                shuffle_idxs = np.arange(PY.shape[0])
                if shuffle:
                    np.random.shuffle(shuffle_idxs)
                # save to h5
                P = PY[:, :2]
                ds_name = "p_%s" % name
                PairTraintest.__log.info(
                    'writing %s %s %s', ds_name, name, P.shape)
                fh.create_dataset(ds_name, data=P[shuffle_idxs])
                Y = PY[:, -1]
                ds_name = "y_%s" % name
                PairTraintest.__log.info(
                    'writing %s %s', ds_name, Y.shape)
                fh.create_dataset(ds_name, data=Y[shuffle_idxs])

        PairTraintest.__log.info('PairTraintest saved to %s', out_file)

    @staticmethod
    def generate_splits(X1, X2, pairs):
        # leave left out
        x1_set = list(set(pairs[:, 0]))
        x1_train_idxs = x1_set[:int(len(x1_set) * .8)]
        x1_train_mask = np.isin(pairs[:, 0], x1_train_idxs)
        x1_train, x1_test = pairs[x1_train_mask], pairs[~x1_train_mask]
        x1_train_test = pairs[x1_train_mask], pairs[~x1_train_mask]
        assert(len(set(x1_train[:, 0]) & set(x1_test[:, 0])) == 0)
        # leave right out
        x2_set = list(set(pairs[:, 1]))
        x2_train_idxs = x2_set[:int(len(x2_set) * .8)]
        x2_train_mask = np.isin(pairs[:, 1], x2_train_idxs)
        x2_train, x2_test = pairs[x2_train_mask], pairs[~x2_train_mask]
        x2_train_test = pairs[x2_train_mask], pairs[~x2_train_mask]
        assert(len(set(x2_train[:, 1]) & set(x2_test[:, 1])) == 0)
        # leave both out
        both_train_mask = np.logical_and(x1_train_mask, x2_train_mask)
        both_test_mask = np.logical_and(~x1_train_mask, ~x2_train_mask)
        both_train, both_test = pairs[both_train_mask], pairs[both_test_mask]
        both_train_test = pairs[both_train_mask], pairs[both_test_mask]
        assert(len(set(both_train[:, 0]) & set(both_test[:, 0])) == 0)
        assert(len(set(both_train[:, 1]) & set(both_test[:, 1])) == 0)
        return x1_train_test, x2_train_test, both_train_test

    @staticmethod
    def generator_fn(file_name, split, batch_size=None, only_x=False,
                     replace_nan=None, mask_fn=None):
        """Return the generator function that we can query for batches.

        file_name(str): The H5 generated via `create`
        split(str): One of 'train_train', 'train_test', or 'test_test'
        batch_size(int): Size of a batch of data.
        only_x(bool): Usually when predicting only X are useful.
        replace_nan(bool): Value used for NaN replacement.
        """
        reader = PairTraintest(file_name, split)
        reader.open()
        # read shapes
        x1_shape = reader._f[reader.x1_name].shape
        x2_shape = reader._f[reader.x2_name].shape
        y_shape = reader._f[reader.y_name].shape
        p_shape = reader._f[reader.p_name].shape
        # read data types
        x1_dtype = reader._f[reader.x1_name].dtype
        y_dtype = reader._f[reader.y_name].dtype
        # no batch size -> return everything
        if not batch_size:
            batch_size = p_shape[0]
        # keep X in memory for resolving pairs quickly
        PairTraintest.__log.debug('Loading Xs')
        X1 = reader.get_all_x1()
        X2 = reader.get_all_x2()
        # default mask is not mask
        if mask_fn is None:
            def mask_fn(*data):
                return data

        def example_generator_fn():
            # generator function yielding data
            epoch = 0
            beg_idx, end_idx = 0, batch_size
            total = reader._f[reader.p_name].shape[0]
            while True:
                if beg_idx >= total:
                    beg_idx, end_idx = 0, batch_size
                    epoch += 1
                    #PairTraintest.__log.debug('EPOCH %i', epoch)
                pairs, y = reader.get_py(beg_idx, end_idx)
                x1 = X1[pairs[:, 0]]
                x2 = X2[pairs[:, 1]]
                x1, x2, y = mask_fn(x1, x2, y)
                if only_x:
                    yield np.hstack((x1, x2))
                else:
                    yield np.hstack((x1, x2)), y
                beg_idx, end_idx = beg_idx + batch_size, end_idx + batch_size

        shapes = (y_shape[0], x1_shape[1] + x2_shape[1]), y_shape
        dtypes = (x1_dtype, y_dtype)
        return shapes, dtypes, example_generator_fn
