"""Data aggregation classes."""
import collections
import numpy as np

from chemicalchecker.util import logged


@logged
class AggregateAsMatrix(object):
    """AggregateAsMatrix class.

    Given a matrix with keys with potential duplicates, aggregate them.
    """

    def __init__(self, method):
        """Initialize a AggregateAsMatrix instance.

        Args:
            method(str): The aggregation method to be used. Must be one of:
                -first: The first occurrence of the signature is kept.
                -last: The last occurrence of the signature is kept.
                -average: The average of the signature is kept.
        """
        self.method = method

    def get_agg_func(self):
        """Get aggregation function."""
        self.__log.debug("Aggregation method = %s" % self.method)
        if self.method not in ['first', 'last', 'average']:
            raise Exception("Aggregate 'method' must be one of: "
                            "'first', 'last', 'average'.")

        def first(V, idxs):
            return V[idxs[0], :]

        def last(V, idxs):
            return V[idxs[-1], :]

        def average(V, idxs):
            if len(idxs) == 1:
                return V[idxs[0], :]
            else:
                return np.mean(V[idxs, :], axis=0)
        return eval(self.method)


@logged
class AggregateAsPairs(object):
    """AggregateAsPairs class.

    Given a matrix with potential duplicates, aggregate them.
    """

    def __init__(self, method):
        """Initialize a AggregateAsPairs instance.

        Args:
            method(str): The aggregation method to be used. Must be one of:
                -first: The first occurrence of the signature is kept.
                -last: The last occurrence of the signature is kept.
                -average: The average of the signature is kept.
        """
        self.method = method

    def get_agg_func(self):
        """Get aggregation function."""
        self.__log.debug("Aggregation method = %s" % self.method)
        if self.method not in ['first', 'last', 'average']:
            raise Exception("Aggregate 'method' must be one of: "
                            "'first', 'last', 'average'.")

        def first(V, idxs):
            if len(idxs) == 1:
                return V[idxs[0], :]
            else:
                v0 = V[idxs[0], :]
                for idx in idxs[1:]:
                    v1 = V[idx, :]
                    zero_idxs = np.where(
                        np.logical_and(v0 == 0, v1 != 0))[0]
                    for zidx in zero_idxs:
                        v0[zidx] = v1[zidx]
                return v0

        def last(V, idxs):
            if len(idxs) == 1:
                return V[idxs[0], :]
            else:
                idxs = np.array(idxs)[::-1]
                v0 = V[idxs[0], :]
                for idx in idxs[1:]:
                    v1 = V[idx, :]
                    zero_idxs = np.where(
                        np.logical_and(v0 == 0, v1 != 0))[0]
                    for zidx in zero_idxs:
                        v0[zidx] = v1[zidx]
                return v0

        def average(V, idxs):
            if len(idxs) == 1:
                return V[idxs[0], :]
            else:
                idxs = np.array(idxs)[::-1]
                num = np.sum(V[idxs, :], axis=0)
                den = np.sum(V[idxs, :] != 0, axis=0)
                mask = den > 0
                v = np.zeros(len(num))
                v[mask] = num[mask] / den[mask]
                return v
        return eval(self.method)


@logged
class Aggregate(object):
    """Aggregate class.

    Aggregate samples.
    """

    def __init__(self, method, input_type):
        """Initialize a Aggregate instance.

        Args:
            method(str): The aggregation method to be used. Must be one of:
                -first: The first occurrence of the signature is kept.
                -last: The last occurrence of the signature is kept.
                -average: The average of the signature is kept.
            input_type(str): One of 'pairs' or 'matrix'.
        """
        if input_type not in ['pairs', 'matrix']:
            raise Exception("Input type must be 'pairs' or 'matrix'")
        if input_type == 'pairs':
            self.agg = AggregateAsPairs(method=method)
        else:
            self.agg = AggregateAsMatrix(method=method)

    def transform(self, V, keys, keys_raw):
        """Do the aggregation.

        Args:
            V(matrix): The signatures matrix.
            keys(array): The keys.
            keys_raw(array): The raw keys (default=None).

        Returns a (V, keys, keys_raw) tuple.
        """
        if np.isnan(np.sum(V)):
            raise Exception("V matrix cannot have NaN values")
        if np.isinf(np.sum(V)):
            raise Exception("V matrix cannot have inf values")
        if len(keys) == len(set(keys)):
            self.__log.debug("Matrix does not need aggregation")
            return V, keys, keys_raw
        self.__log.debug("Looking for duplicated keys")
        keys_idxs = collections.defaultdict(list)
        for i, k in enumerate(keys):
            keys_idxs[k] += [i]
        keys_ = np.array(sorted(keys_idxs.keys()))
        if keys_raw is not None:
            keys_raw_ = []
        else:
            keys_raw_ = None
        V_ = np.zeros((len(keys_idxs), V.shape[1]), dtype=V.dtype)
        self.__log.debug("Applying aggregation method")
        agg_func = self.agg.get_agg_func()
        for i, k in enumerate(keys_):
            idxs = keys_idxs[k]
            V_[i, :] = agg_func(V, idxs)
            if keys_raw is not None:
                keys_raw_ += ["|".join(keys_raw[idxs])]
        if keys_raw_ is not None:
            keys_raw_ = np.array(keys_raw_)
        return V_, keys_, keys_raw_
