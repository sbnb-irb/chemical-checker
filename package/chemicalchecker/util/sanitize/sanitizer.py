"""Simple sanitization of input matrices.

Garbage in, garbage out, before data can be used it needs to be cleaned.
This includes:

* Removing seldomly occurring features (columns)
* Removing molecules with few data (rows)
* Handling missing data (NaNs or infs)
* Trimming less informative features if too many are provided
"""
import os
import h5py
import uuid
import numpy as np
from scipy.stats import entropy

from chemicalchecker.core.signature_data import DataSignature
from chemicalchecker.util import Config, logged


@logged
class Sanitizer(object):
    """Sanitizer class."""

    def __init__(self, *args, impute_missing=True, trim=True,
                 max_features=10000,
                 check_features=True, min_feature_abs=5, max_feature_freq=0.8,
                 check_keys=True, min_keys_abs=1, max_keys_freq=0.8,
                 sample_size=1000, max_categories=20, zero_as_missing=True,
                 chunk_size=10000, tmp_path=None, **kwargs):
        """Initialize a Sanitizer instance.

        Args:
            impute_missing (bool): True if NaNs (and -inf/+inf) will be imputed.
                NaN will be median, -inf/+inf will be min/max of the column.
            trim (bool): Trim dataset to have a maximum number of features.
            max_features (int): Maximum number of features to keep
                (default=10000).
            check_features (bool): True if we want to drop features based on
                frequency arguments. For categorical data, 0 is considered as
                `missing`. For continuous, any non numerical value.
            min_feature_abs (int): Minimum number (counts) of occurrences
                of feature, column-wise. (default=5).
            max_feature_freq (float): Maximum proportion of occurrences of
                the feature, column-wise. (default=0.8).
            check_keys (bool): True if we want to drop keys based on
                frequency arguments. For categorical data, 0 is considered as
                `missing`. For continuous, any non numerical value.
            min_key_abs (int): Minimum number (counts) of occurrences
                of feature, row-wise. (default=1).
            max_keys_freq (float): Maximum proportion of occurrences of
                the feature, row-wise. (default=0.8).
            sample_size (int): rows used for determining data type.
            max_categories (int): Maximum number of categories we can expect.
            zero_as_missing (bool): Only applyied to categorical data (usually)
                binary where the 0 denotes a missing information. Used when
                filtering row or columns by frequency.
        """
        self.__dict__.update(locals())
        if self.tmp_path is None:
            self.tmp_path = Config().PATH.CC_TMP
        for k, v in self.__dict__.items():
            if k == 'self':
                continue
            self.__log.debug("{:<22}: {:>12}".format(str(k), str(v)))

    def transform(self, data=None, V=None, keys=None, keys_raw=None,
                  features=None, sign=None):
        """Sanitize data

        Args:
            data (str): Path to a H5 or a DataSignature (default=None).
            V (matrix): Input matrix (default=None).
            keys (array): Keys (default=None).
            keys_raw (array): Keys raw (default=None).
            features (array): Features (default=None).
            sign (DataSignature): Auxiliary data used to impute (default=None).
        """
        if data is not None and V is not None:
            raise Exception("Too many inputs! Either provide `data` or `V`.")
        if data is None:
            # create temporary h5 data structure
            was_data = False
            if V is None or keys is None or features is None:
                raise Exception("`data` not provided so "
                                "`V`, `keys`, `features` are expected.")
            if keys_raw is None:
                keys_raw = keys
            tag = str(uuid.uuid4())
            data_path = os.path.join(self.tmp_path, "%s.h5" % tag)
            data = DataSignature(data_path)
            datasets = {'V': V, 'keys': keys,
                        'keys_raw': keys, 'features': features}
            data.add_datasets(datasets)
            self.__log.debug("Saving temporary data to %s" % data_path)
            V = None
            keys = None
            keys_raw = None
            features = None
        else:
            was_data = True
            if isinstance(data, str):
                data = DataSignature(data)
            req_ds = ['V', 'keys', 'keys_raw', 'features']
            if any([k not in data.info_h5 for k in req_ds]):
                raise Exception("`data` must contain %s." % str(req_ds))
        self.data = data

        # check data type
        self.__log.debug("Data type: %s" % str(self.data[0].dtype))
        self.__log.debug("Data shape: %s" % str(self.data.shape))
        self.__log.debug("Data size: %s" % str(self.data.size))
        if self.data.size > 1e9:
            self.__log.debug("Data size exceeds 1e9, reducing `chunk_size`.")
            self.chunk_size = 1000
        cs = self.chunk_size
        vals = data[:self.sample_size].ravel()
        unique_vals, unique_counts = np.unique(vals[np.isfinite(vals)],
                                               return_counts=True)
        if len(unique_vals) <= self.max_categories:
            self.is_categorical = True
            self.categories = unique_vals
            unique_freqs = {k: v/vals.size for k,
                            v in zip(unique_vals, unique_counts)}
            self.__log.debug("Data is categorical: %s" % str(unique_vals))
            freq_str = ', '.join(['%s : %.3f' % (k, v)
                                  for k, v in unique_freqs.items()])
            self.__log.debug("Category frequency: [%s]" % freq_str)
        else:
            self.is_categorical = False
            self.__log.debug("Data is continuous.")

        # if a signature is specified make sure the new input has equal columns
        if sign is not None:
            try:
                ref_features = sign.get_h5_dataset('features')
            except Exception:
                raise Exception("`sign` should have the `features` dataset.")
            if len(set(features) & set(ref_features)) != len(set(ref_features)):
                raise Exception("`data` must contains at least all features "
                                "present in `sign`.")
            add_features = sorted(list(set(features) - set(ref_features)))
            if add_features:
                self.__log.info("Some input features are skipped as are not "
                                "present in reference: %s" % str(add_features))
            # we assume that features are in the same order
            mask = np.isin(list(features), list(ref_features))
            data.filter_h5_dataset('V', mask, axis=1)
            data.filter_h5_dataset('features', mask, axis=1)

        # check features frequencies
        if self.check_features:
            self.__log.debug('Checking features:')
            features = data.get_h5_dataset('features')
            drop_abs = np.full((data.shape[1],), False, dtype=bool)
            drop_freq = np.full((data.shape[1],), False, dtype=bool)
            for chunk, cols in data.chunk_iter('V', cs, axis=1, chunk=True):
                missing = np.sum(~np.isfinite(cols), axis=0)
                if self.is_categorical and self.zero_as_missing:
                    missing += np.sum(cols == 0, axis=0)
                present = data.shape[0] - missing
                present_freq = present / data.shape[0]
                drop_abs[chunk] = present < self.min_feature_abs
                if self.is_categorical:
                    drop_freq[chunk] = present_freq > self.max_feature_freq
            self.__log.info('Filter %s features (min_feature_abs): %s'
                            % (np.sum(drop_abs), str(features[drop_abs])))
            self.__log.info('Filter %s features (max_feature_freq): %s'
                            % (np.sum(drop_freq), str(features[drop_freq])))
            keep = ~np.logical_or(drop_abs, drop_freq)
            if np.any(~keep):
                data.filter_h5_dataset('V', keep, axis=1)
                data.filter_h5_dataset('features', keep, axis=1)
        
        print( 'Features frequency', data.shape )
        
        # check keys frequencies
        if self.check_keys:
            self.__log.debug('Checking keys:')
            keys = data.get_h5_dataset('keys')
            drop_abs = np.full((data.shape[0],), False, dtype=bool)
            drop_freq = np.full((data.shape[0],), False, dtype=bool)
            for chunk, rows in data.chunk_iter('V', cs, axis=0, chunk=True):
                missing = np.sum(~np.isfinite(rows), axis=1)
                if self.is_categorical and self.zero_as_missing:
                    missing += np.sum(rows == 0, axis=1)
                present = data.shape[1] - missing
                present_freq = present / data.shape[1]
                drop_abs[chunk] = present < self.min_keys_abs
                if self.is_categorical:
                    drop_freq[chunk] = present_freq > self.max_keys_freq
            self.__log.info('Filter %s keys (min_keys_abs): %s'
                            % (np.sum(drop_abs), str(keys[drop_abs])))
            self.__log.info('Filter %s keys (max_keys_freq): %s'
                            % (np.sum(drop_freq), str(keys[drop_freq])))
            keep = ~np.logical_or(drop_abs, drop_freq)
            if np.any(~keep):
                data.filter_h5_dataset('V', keep, axis=0)
                data.filter_h5_dataset('keys', keep, axis=0)
                data.filter_h5_dataset('keys_raw', keep, axis=0)

        # count NaN & infs
        self.__log.debug('Missing values:')
        nan_counts = dict()
        nan_counts['NaN'] = 0
        nan_counts['+inf'] = 0
        nan_counts['-inf'] = 0
        nan_counts['all'] = 0
        for chunk in data.chunk_iter('V', cs):
            nan_counts['NaN'] += np.sum(np.isnan(chunk))
            nan_counts['+inf'] += np.sum(np.isposinf(chunk))
            nan_counts['-inf'] += np.sum(np.isneginf(chunk))
            nan_counts['all'] += nan_counts['NaN']
            nan_counts['all'] += nan_counts['+inf']
            nan_counts['all'] += nan_counts['-inf']
        for k, v in nan_counts.items():
            if k == 'all':
                continue
            self.__log.debug("{:<5}: {:>12}".format(str(k), str(v)))
    
        # impute NaN & infs
        if self.impute_missing and nan_counts['all'] != 0:
            self.__log.info('Imputing missing values.')
            #hf = h5py.File(data.data_path, 'a')
            for chunk, cols in data.chunk_iter('V', cs, axis=1, chunk=True):
                # get values for replacements
                ref_cols = cols
                if sign is not None:
                    ref_cols = sign[:, chunk]
                nan_vals = np.nanmedian(ref_cols, axis=0)
                if self.is_categorical:
                    # if categorical replace NaNs with most frequent instead
                    count = np.zeros((cols.shape[1], len(self.categories)))
                    for idx, cat in enumerate(self.categories):
                        count[:, idx] = np.sum(ref_cols == cat, axis=0)
                    nan_vals = np.argmax(count, axis=1)
                cols_masked = np.ma.masked_array(
                    ref_cols, mask=~np.isfinite(ref_cols))
                posinf_vals = np.max(cols_masked, axis=0).data
                neginf_vals = np.min(cols_masked, axis=0).data
                # replace
                idxs = np.where(np.isnan(cols))
                cols[idxs] = np.take(nan_vals, idxs[1])
                idxs = np.where(np.isposinf(cols))
                cols[idxs] = np.take(posinf_vals, idxs[1])
                idxs = np.where(np.isneginf(cols))
                cols[idxs] = np.take(neginf_vals, idxs[1])
                data.set_data_h5_dataset( 'V', chunk, cols, 0 )
                #data['V'][:, chunk] = cols
            #hf.close()


        # count NaN & infs
        if nan_counts['all'] != 0:
            self.__log.debug('Missing values:')
            nan_counts = dict()
            nan_counts['NaN'] = 0
            nan_counts['+inf'] = 0
            nan_counts['-inf'] = 0
            cs = self.chunk_size
            for chunk in data.chunk_iter('V', cs):
                nan_counts['NaN'] += np.sum(np.isnan(chunk))
                nan_counts['+inf'] += np.sum(np.isposinf(chunk))
                nan_counts['-inf'] += np.sum(np.isneginf(chunk))
            for k, v in nan_counts.items():
                if k == 'all':
                    continue
                self.__log.debug("{:<5}: {:>12}".format(str(k), str(v)))
        
        print( 'Flter nans and inf', data.shape )
        
        # trim if there are too many features
        if self.trim and data.shape[1] > self.max_features:
            self.__log.debug("More than %d features, trimming the "
                             "least informative ones." % self.max_features)
            if self.is_categorical:
                entropy_vals = np.zeros((data.shape[1],))
                for chunk, cols in data.chunk_iter('V', cs, axis=1, chunk=True):
                    entropy_vals[chunk] = entropy(cols, axis=0)
                features_rank = np.argsort(entropy_vals)[::-1]
            else:
                std_vals = np.zeros((data.shape[1],))
                for chunk, cols in data.chunk_iter('V', cs, axis=1, chunk=True):
                    std_vals[chunk] = np.std(cols, axis=0)
                features_rank = np.argsort(std_vals)[::-1]
            keep = np.full((data.shape[1], ), False, dtype=bool)
            keep[features_rank[: self.max_features]] = True
            filtered = data.get_h5_dataset('features', mask=~keep)
            data.filter_h5_dataset('V', keep, axis=1)
            data.filter_h5_dataset('features', keep, axis=1)
            self.__log.info("Removed %i features (max_features): %s"
                            % (len(filtered), str(filtered)))
        
        print( 'Filter too many features', data.shape )
        
        self.__log.info("Sanitized data shape: %s" % str(self.data.shape))

        # return if input was raw data
        if not was_data:
            with h5py.File(data.data_path, "r") as hf:
                V = hf["V"][:]
                keys = hf["keys"][:].astype(str)
                keys_raw = hf["keys_raw"][:].astype(str)
                features = hf["features"][:].astype(str)
            os.remove(data.data_path)
            return V, keys, keys_raw, features
