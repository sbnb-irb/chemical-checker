"""Basic train-test splitter."""
import h5py
import numpy as np
from tqdm import tqdm

from chemicalchecker.util import logged


@logged
class Traintest(object):
    """Traintest class."""

    def __init__(self, hdf5_file, split, replace_nan=None):
        """Initialize a Traintest instance.

        We assume the file is containing diffrent splits.
        e.g. "x_train", "y_train", "x_test", ...
        """
        self._file = hdf5_file
        self._f = None
        self.replace_nan = replace_nan
        if split is None:
            self.x_name = "x"
            self.y_name = "y"
            self.sw_name = "sw"
        else:
            self.x_name = "x_%s" % split
            self.y_name = "y_%s" % split
            self.sw_name = "sw_%s" % split
            '''
            available_splits = self.get_split_names()
            if split not in available_splits:
                raise Exception("Split '%s' not found in %s!" %
                                (split, str(available_splits)))
            '''

    def get_x_shapes(self):
        """Return the shpaes of X."""
        self.open()
        x_shape = self._f[self.x_name].shape
        self.close()
        return x_shape

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
            split_names = ['train', 'test', 'validation']
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

    def get_sw(self, beg_idx, end_idx):
        """Get a batch of X."""

        features = self._f[self.sw_name][beg_idx: end_idx]
        # handle NaNs
        if self.replace_nan is not None:
            features[np.where(np.isnan(features))] = self.replace_nan
        return features

    def get_xy(self, beg_idx, end_idx):
        """Get a batch of X and Y."""

        features = self._f[self.x_name][beg_idx: end_idx]
        # handle NaNs
        if self.replace_nan is not None:
            features[np.where(np.isnan(features))] = self.replace_nan
        labels = self._f[self.y_name][beg_idx: end_idx]
        return features, labels

    def get_x(self, beg_idx, end_idx):
        """Get a batch of X."""

        features = self._f[self.x_name][beg_idx: end_idx]
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

    def get_all_x_columns(self, columns):
        """Get all the X.

        Args:
            colums(tuple(int,int)): start, stop indexes.
        """
        features = self._f[self.x_name][:, slice(*columns)]
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
        Traintest.create(X, Y, out_filename)

    @staticmethod
    def get_split_indeces(rows, fractions, random_state=None):
        """Get random indeces for different splits."""
        if not sum(fractions) == 1.0:
            raise Exception("Split fractions should sum to 1.0")
        # shuffle indeces
        idxs = list(range(rows))
        np.random.seed(random_state)
        np.random.shuffle(idxs)
        # from frequs to indices
        splits = np.cumsum(fractions)
        splits = splits[:-1]
        splits *= len(idxs)
        splits = splits.round().astype(int)
        return np.split(idxs, splits)

    @staticmethod
    def create(X, Y, out_file, split_names=['train', 'test', 'validation'],
               split_fractions=[.8, .1, .1], x_dtype=np.float32,
               y_dtype=np.float32, chunk_size=10000):
        """Create the HDF5 file with validation splits for both X and Y.

        Args:
            X(numpy.ndarray): features to train from.
            Y(numpy.ndarray): labels to predict.
            out_file(str): path of the h5 file to write.
            split_names(list(str)): names for the split of data.
            split_fractions(list(float)): fraction of data in each split.
            x_dtype(type): numpy data type for X.
            y_dtype(type): numpy data type for Y (np.float32 for regression,
                int32 for classification.
        """
        # Force number of dimension to 2 (reshape Y)
        if Y.ndim == 1:
            Traintest.__log.debug("We need Y as a column vector, reshaping.")
            Y = np.reshape(Y, (len(Y), 1))
        Traintest.__log.debug(
            "{:<20} shape: {:>10}".format("input X", str(X.shape)))
        Traintest.__log.debug(
            "{:<20} shape: {:>10}".format("input Y", str(Y.shape)))
        # train test validation splits
        if len(split_names) != len(split_fractions):
            raise Exception("Split names and fraction should be same amount.")
        split_names = [s.encode() for s in split_names]
        split_idxs = Traintest.get_split_indeces(
            Y.shape[0], split_fractions)

        # create dataset
        Traintest.__log.info('Traintest saving to %s', out_file)
        with h5py.File(out_file, "w") as fh:
            fh.create_dataset('split_names', data=split_names)
            fh.create_dataset('split_fractions', data=split_fractions)

            for name, idxs in zip(split_names, split_idxs):
                ds_name = "x_%s" % name.decode()                    # NS added decode() otherwise--> x_b'train'
                fh.create_dataset(ds_name, (len(idxs), X.shape[1]), dtype=x_dtype)

                for i in range(0, len(idxs), chunk_size):
                    chunk = slice(i, i + chunk_size)
                    fh[ds_name][chunk] = X[idxs[chunk]]

                Traintest.__log.debug("Written: {:<20} shape: {:>10}".format(ds_name, str(fh[ds_name].shape)))
                ds_name = "y_%s" % name.decode()                  # NS added decode() otherwise--> y_b'train'
                fh.create_dataset(ds_name, (len(idxs), Y.shape[1]), dtype=y_dtype)

                for i in range(0, len(idxs), chunk_size):
                    chunk = slice(i, i + chunk_size)
                    fh[ds_name][chunk] = Y[idxs[chunk]]
                    
                Traintest.__log.debug("Written: {:<20} shape: {:>10}".format(
                    ds_name, str(fh[ds_name].shape)))
        Traintest.__log.info('Traintest saved to %s', out_file)

    @staticmethod
    def split_h5(in_file, out_file,
                 split_names=['train', 'test', 'validation'],
                 split_fractions=[.8, .1, .1], chunk_size=1000):
        """Create the HDF5 file with validation splits from an input file.

        Args:
            in_file(str): path of the h5 file to read from.
            out_file(str): path of the h5 file to write.
            split_names(list(str)): names for the split of data.
            split_fractions(list(float)): fraction of data in each split.
        """
        with h5py.File(in_file, 'r') as hf_in:
            # log input datasets and shapes
            for k in hf_in.keys():
                Traintest.__log.debug(
                    "{:<20} shape: {:>10}".format(k, str(hf_in[k].shape)))
                rows = hf_in[k].shape[0]

            # train test validation splits
            if len(split_names) != len(split_fractions):
                raise Exception(
                    "Split names and fraction should be same amount.")
            split_names = [s.encode() for s in split_names]
            split_idxs = Traintest.get_split_indeces(rows, split_fractions)

            Traintest.__log.info('Traintest saving to %s', out_file)
            with h5py.File(out_file, "w") as hf_out:
                # create fixed datasets
                hf_out.create_dataset(
                    'split_names', data=np.array(split_names))
                hf_out.create_dataset(
                    'split_fractions', data=np.array(split_fractions))

                for name, idxs in zip(split_names, split_idxs):
                    # for each original dataset
                    for k in hf_in.keys():
                        # create all splits
                        ds_name = "%s_%s" % (k, name.decode())
                        hf_out.create_dataset(ds_name,
                                              (len(idxs), hf_in[k].shape[1]),
                                              dtype=hf_in[k].dtype)
                        # fill-in by chunks
                        for i in range(0, len(idxs), chunk_size):
                            chunk = slice(i, i + chunk_size)
                            sorted_idxs = sorted(list(idxs[chunk]))
                            hf_out[ds_name][chunk] = hf_in[k][sorted_idxs]
                        Traintest.__log.debug(
                            "Written: {:<20} shape: {:>10}".format(
                                ds_name, str(hf_out[ds_name].shape)))
        Traintest.__log.info('Traintest saved to %s', out_file)

    @staticmethod
    def split_h5_blocks(in_file, out_file,
                        split_names=['train', 'test', 'validation'],
                        split_fractions=[.8, .1, .1], block_size=1000,
                        datasets=None):
        """Create the HDF5 file with validation splits from an input file.

        Args:
            in_file(str): path of the h5 file to read from.
            out_file(str): path of the h5 file to write.
            split_names(list(str)): names for the split of data.
            split_fractions(list(float)): fraction of data in each split.
            block_size(int): size of the block to be used.
            dataset(list): only split the given dataset and ignore others.
        """
        with h5py.File(in_file, 'r') as hf_in:
            # log input datasets and get shapes
            for k in hf_in.keys():
                Traintest.__log.debug(
                    "{:<20} shape: {:>10}".format(k, str(hf_in[k].shape)))
                rows = hf_in[k].shape[0]
            # reduce block size if it is not adequate
            while rows / (float(block_size) * 10) <= 1:
                block_size = int(block_size / 10)
                Traintest.__log.warning(
                    "Reducing block_size to: %s", block_size)
            # train test validation splits
            if len(split_names) != len(split_fractions):
                raise Exception(
                    "Split names and fraction should be same amount.")
            split_names = [s.encode() for s in split_names]
            # get indeces of blocks for each split
            split_block_idx = Traintest.get_split_indeces(
                int(np.floor(rows / block_size)) + 1,
                split_fractions)
            if datasets is None:
                datasets = hf_in.keys()
            for dataset_name in datasets:
                if dataset_name not in hf_in.keys():
                    raise Exception(
                        "Dataset %s not found in source file." % dataset_name)
            # save to output file
            Traintest.__log.info('Traintest saving to %s', out_file)
            with h5py.File(out_file, "w") as hf_out:
                # create fixed datasets
                hf_out.create_dataset(
                    'split_names', data=np.array(split_names))
                hf_out.create_dataset(
                    'split_fractions', data=np.array(split_fractions))

                for name, blocks in zip(split_names, split_block_idx):
                    # for each original dataset
                    for k in datasets:
                        # create all splits
                        ds_name = "%s_%s" % (k, name.decode())
                        # need total size and mapping of blocks
                        src_dst = list()
                        total_size = 0
                        for dst, src in enumerate(sorted(blocks)):
                            # source block start-end
                            src_start = src * block_size
                            src_end = (src * block_size) + block_size
                            # check current block size to avoid overflowing
                            curr_block_size = block_size
                            if src_end > hf_in[k].shape[0]:
                                src_end = hf_in[k].shape[0]
                                curr_block_size = src_end - src_start
                            # update total size
                            total_size += curr_block_size
                            # destination start-end
                            dst_start = dst * block_size
                            dst_end = (dst * block_size) + curr_block_size
                            src_slice = (src_start, src_end)
                            dst_slice = (dst_start, dst_end)
                            src_dst.append((src_slice, dst_slice))
                            # Traintest.__log.debug(
                            #    "src: %s  dst: %s" % src_dst[-1])
                            # Traintest.__log.debug(
                            #    "block_size: %s" % curr_block_size)
                        # create block matrix
                        reshape = False
                        if len(hf_in[k].shape) == 1:
                            cols = 1
                            reshape = True
                        else:
                            cols = hf_in[k].shape[1]
                        hf_out.create_dataset(ds_name,
                                              (total_size, cols),
                                              dtype=hf_in[k].dtype)
                        for src_slice, dst_slice in tqdm(src_dst):
                            src_chunk = slice(*src_slice)
                            dst_chunk = slice(*dst_slice)
                            # Traintest.__log.debug(
                            #    "writing src: %s  to dst: %s" %
                            #    (src_slice, dst_slice))
                            if reshape:
                                hf_out[ds_name][dst_chunk] = np.expand_dims(
                                    hf_in[k][src_chunk], 1)
                            else:
                                hf_out[ds_name][dst_chunk] = hf_in[
                                    k][src_chunk]
                        # Traintest.__log.debug(
                        #    "Written: {:<20} shape: {:>10}".format(
                        #        ds_name, str(hf_out[ds_name].shape)))
        Traintest.__log.info('Traintest saved to %s', out_file)

    @staticmethod
    def generator_fn(file_name, split, batch_size=None, only_x=False,
                     sample_weights=False, shuffle=True,
                     return_on_epoch=False):
        """Return the generator function that we can query for batches."""
        reader = Traintest(file_name, split)
        reader.open()

        if only_x:
            x_shape = reader._f[reader.x_name].shape
            shapes = x_shape
            x_dtype = reader._f[reader.x_name].dtype
            dtypes = x_dtype
        else:
            # read shapes
            x_shape = reader._f[reader.x_name].shape
            y_shape = reader._f[reader.y_name].shape
            shapes = (x_shape, y_shape)
            # read data types
            x_dtype = reader._f[reader.x_name].dtype
            y_dtype = reader._f[reader.y_name].dtype
            dtypes = (x_dtype, y_dtype)
        # no batch size -> return everything
        if not batch_size:
            batch_size = x_shape[0]
        batch_beg_end = np.zeros((int(np.ceil(x_shape[0] / batch_size)), 2))
        last = 0
        for row in batch_beg_end:
            row[0] = last
            row[1] = last + batch_size
            last = row[1]
        batch_beg_end = batch_beg_end.astype(int)

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
                beg_idx, end_idx = batch_beg_end[batch_idx]
                if only_x:
                    if sample_weights:
                        yield reader.get_x(beg_idx, end_idx), \
                            reader.get_sw(beg_idx, end_idx)
                    else:
                        yield reader.get_x(beg_idx, end_idx)
                else:
                    if sample_weights:
                        yield reader.get_xy(beg_idx, end_idx), \
                            reader.get_sw(beg_idx, end_idx)
                    else:
                        yield reader.get_xy(beg_idx, end_idx)
                batch_idx += 1

        return shapes, dtypes, example_generator_fn
