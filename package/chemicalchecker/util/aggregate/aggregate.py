"""Given duplicated keys, aggregate"""
import collections

from chemicalchecker.util import logged

@logged
class Aggregate(object):
    """Given a matrix with keys with potential duplicates, aggregate them"""
    def __init__(self, method):
        """Initialize class
            
            Args:
                method(str): The aggregation method to be used. Must be one of:
                    -first: The first occurrence of the signature is kept.
                    -last: The last occurrence of the signature is kept.
                    -average: The average of the signature is kept.
        """
        self.method = method

    def get_agg_func(self):
        if self.method == "first":
            def agg_func(V, idxs):
                return V[idxs[0],:]
        if self.method == "last":
            def agg_func(V, idxs):
                return V[idxs[-1],:]
        if self.method == "average":
            def agg_func(V, idxs):
                if len(idxs) == 1:
                    return V[idxs[0],:]
                else:
                    return np.mean(V[idxs,:], axis=0)
        return agg_func
        
    def transform(self, V, keys, keys_raw=None):
        """Do the aggregation.

            Args:
                V(matrix): The signatures matrix.
                keys(array): The keys.
                keys_raw(array): The raw keys (default=None).

            Returns a (V, keys, keys_raw) tuple.
        """        
        self.__log.debug("Looking for duplicated keys")
        keys_idxs = collections.defaultdict(list)
        for i,k in enumerate(keys):
            keys_idxs[k] += [i]
        keys_ = np.array(sorted(keys_idxs.keys()))
        if keys_raw is not None:
            keys_raw_ = []
        V_ = np.zeros((len(keys_idxs), V.shape[1]), dtype=V.dtype())
        self.__log.debug("Applying aggregation method")
        agg_func = self.get_acc_func()
        for i,k in enumerate(keys_):
            idxs = keys_idxs[k]
            V_[i,:] = agg_func(V, idxs)
            if keys_raw is not None:
                keys_raw_ += ["|".join(keys_raw[idxs])]
        keys_raw_ = np.array(keys_raw_)
        return V_, keys_, keys_raw_