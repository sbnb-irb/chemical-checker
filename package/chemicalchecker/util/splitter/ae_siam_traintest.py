"""Splitter for Siamese Autoencoder."""
import h5py
import numpy as np
from tqdm import tqdm

from chemicalchecker.util import logged


@logged
class AE_SiameseTraintest(object):
    """AE_SiameseTraintest class."""

    def __init__(self, hdf5_file, split, replace_nan=None):
        """Initialize a AE_SiameseTraintest instance.

        We assume the file is containing diffrent splits.
        e.g. "x_train", "y_train", "x_test", ...
        """
        self._file = hdf5_file
        self._f = None
        self.replace_nan = replace_nan
        if split is None:
            self.x_name_left = "x_left"
            self.y_name_left = "x_left"
            self.sw_name_left = "sw_left"
            self.x_name_right = "x_right"
            self.y_name_right = "x_right"
            self.sw_name_right = "sw_right"
        else:
            self.x_name_left = "x_left_%s" % split
            self.y_name_left = "x_left_%s" % split
            self.sw_name_left = "sw_left_%s" % split
            self.x_name_right = "x_right_%s" % split
            self.y_name_right = "x_right_%s" % split
            self.sw_name_right = "sw_right_%s" % split
            '''
            available_splits = self.get_split_names()
            if split not in available_splits:
                raise Exception("Split '%s' not found in %s!" %
                                (split, str(available_splits)))
            '''

    def get_x_shapes(self):
        """Return the shpaes of X."""
        self.open()
        x_shape = self._f[self.x_name_left].shape
        self.close()
        return x_shape

    def get_xy_shapes(self):
        """Return the shpaes of X an Y."""
        self.open()
        x_shape = self._f[self.x_name_left].shape
        y_shape = self._f[self.y_name_left].shape
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
        features_left = self._f[self.sw_name_left][beg_idx: end_idx]
        features_right = self._f[self.sw_name_right][beg_idx: end_idx]
        # handle NaNs
        if self.replace_nan is not None:
            features_left[
                np.where(np.isnan(features_left))] = self.replace_nan
            features_right[
                np.where(np.isnan(features_right))] = self.replace_nan
        return [features_left, features_right], [features_left, features_right]

    def get_xy(self, beg_idx, end_idx, shuffle):
        """Get a batch of X and Y."""
        features_left = self._f[self.x_name_left][beg_idx: end_idx]
        features_right = self._f[self.x_name_right][beg_idx: end_idx]
        if shuffle:
            np.random.shuffle(features_left)
            np.random.shuffle(features_right)
        # handle NaNs
        if self.replace_nan is not None:
            features_left[
                np.where(np.isnan(features_left))] = self.replace_nan
            features_right[
                np.where(np.isnan(features_right))] = self.replace_nan
        # print (features_left.shape, features_right.shape, beg_idx, end_idx)
        return [features_left, features_right], [features_left, features_right]

    def get_x(self, beg_idx, end_idx):
        """Get a batch of X."""
        features_left = self._f[self.x_name_left][beg_idx: end_idx]
        features_right = self._f[self.x_name_right][beg_idx: end_idx]
        # handle NaNs
        if self.replace_nan is not None:
            features_left[
                np.where(np.isnan(features_left))] = self.replace_nan
            features_right[
                np.where(np.isnan(features_right))] = self.replace_nan
        return [features_left, features_right]

    @staticmethod
    def get_split_indeces(rows, fractions):
        """Get random indeces for different splits."""
        if not sum(fractions) == 1.0:
            raise Exception("Split fractions should sum to 1.0")
        # shuffle indeces
        idxs = list(range(rows))
        idxs_shuflle = list(range(rows))
        np.random.shuffle(idxs_shuflle)
        # from frequs to indices
        splits = np.cumsum(fractions)
        splits = splits[:-1]
        splits *= len(idxs)
        splits = splits.round().astype(int)
        split_left = np.split(idxs, splits)
        split_right = []
        for i in range(len(split_left)):
            split_right.append(np.copy(split_left[i]))
            np.random.shuffle(split_right[i])

        final_split = []

        for i in range(len(split_left)):
            final_split.append((split_left[i], split_right[i]))

        return final_split

    @staticmethod
    def split_h5_blocks(in_file, out_file,
                        split_names=['train', 'test', 'validation'],
                        split_fractions=[.8, .1, .1], block_size=1000,
                        input_datasets=None):
        """Create the HDF5 file with validation splits from an input file.

        Args:
            in_file(str): path of the h5 file to read from.
            out_file(str): path of the h5 file to write.
            split_names(list(str)): names for the split of data.
            split_fractions(list(float)): fraction of data in each split.
            block_size(int): size of the block to be used.
            dataset(list): only split the given dataset and ignore others.
        """
        output_datasets = ['x']

        with h5py.File(in_file, 'r') as hf_in:
            # log input datasets and get shapes
            for k in hf_in.keys():
                AE_SiameseTraintest.__log.debug(
                    "{:<20} shape: {:>10}".format(k, str(hf_in[k].shape)))
                rows = hf_in[k].shape[0]
            # reduce block size if it is not adequate
            while rows / (float(block_size) * 10) <= 1:
                block_size = int(block_size / 10)
                AE_SiameseTraintest.__log.warning(
                    "Reducing block_size to: %s", block_size)
            # train test validation splits
            if len(split_names) != len(split_fractions):
                raise Exception(
                    "Split names and fraction should be same amount.")
            split_names = [s.encode() for s in split_names]
            # get indeces of blocks for each split
            split_block_idx = AE_SiameseTraintest.get_split_indeces(
                rows, split_fractions)

            if input_datasets is None:
                input_datasets = hf_in.keys()
            for dataset_name in input_datasets:
                if dataset_name not in hf_in.keys():
                    raise Exception(
                        "Dataset %s not found in source file." % dataset_name)
            if len(input_datasets) != len(output_datasets):
                raise Exception(
                    "Length of input datasets and out datasets is not the same")
            # save to output file
            AE_SiameseTraintest.__log.info('Traintest saving to %s', out_file)
            with h5py.File(out_file, "w") as hf_out:
                # create fixed datasets
                hf_out.create_dataset(
                    'split_names', data=np.array(split_names))
                hf_out.create_dataset(
                    'split_fractions', data=np.array(split_fractions))

                for name, blocks in zip(split_names, split_block_idx):
                    # for each original dataset
                    for k in range(len(input_datasets)):
                        # create all splits
                        ds_name_left = "%s_left_%s" % (
                            output_datasets[k], name.decode())
                        ds_name_right = "%s_right_%s" % (
                            output_datasets[k], name.decode())
                        # need total size and mapping of blocks
                        total_size = blocks[0].shape[0]
                        index_right = blocks[1]
                        # create block matrix
                        reshape = False
                        if len(hf_in[input_datasets[k]].shape) == 1:
                            cols = 1
                            reshape = True
                        else:
                            cols = hf_in[input_datasets[k]].shape[1]
                        hf_out.create_dataset(ds_name_left,
                                              (total_size, cols),
                                              dtype=hf_in[input_datasets[k]].dtype)
                        hf_out.create_dataset(ds_name_right,
                                              (total_size, cols),
                                              dtype=hf_in[input_datasets[k]].dtype)

                        for i in tqdm(range(0, total_size, block_size)):
                            chunk = slice(i, i + block_size)
                            dst_chunk = chunk
                            src_chunk_left = chunk
                            src_chunk_right = index_right[chunk]
                            src_data_right = np.array(
                                [hf_in[input_datasets[k]][j] for j in src_chunk_right])
                            if src_data_right.shape[0] != block_size:
                                dst_chunk = slice(
                                    i, i + src_data_right.shape[0])
                                src_chunk_left = dst_chunk
                            if reshape:
                                hf_out[ds_name_left][dst_chunk] = np.expand_dims(
                                    hf_in[input_datasets[k]][src_chunk_left], 1)
                                hf_out[ds_name_right][dst_chunk] = np.expand_dims(
                                    src_data_right, 1)
                            else:
                                hf_out[ds_name_left][dst_chunk] = hf_in[
                                    input_datasets[k]][src_chunk_left]
                                hf_out[ds_name_right][
                                    dst_chunk] = src_data_right

        AE_SiameseTraintest.__log.info('Traintest saved to %s', out_file)

    @staticmethod
    def generator_fn(file_name, split, batch_size=None, only_x=False,
                     sample_weights=False, shuffle=True,
                     return_on_epoch=False):
        """Return the generator function that we can query for batches."""
        reader = AE_SiameseTraintest(file_name, split)
        reader.open()

        x_shape = reader._f[reader.x_name_left].shape
        y_shape = reader._f[reader.y_name_left].shape
        x_dtype = reader._f[reader.x_name_left].dtype
        y_dtype = reader._f[reader.y_name_left].dtype
        shapes = (x_shape, y_shape)
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
                        yield reader.get_xy(beg_idx, end_idx, shuffle), \
                            reader.get_sw(beg_idx, end_idx)
                    else:
                        yield reader.get_xy(beg_idx, end_idx, shuffle)
                batch_idx += 1

        return shapes, dtypes, example_generator_fn
