"""Diagnostics for Signatures.

Performs an array of validations and diagnostic analysis.
"""
import os
import random
import pickle
import shutil
import collections
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import normalize
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from chemicalchecker.util import logged
from chemicalchecker.util.plot import DiagnosisPlot
from chemicalchecker.util.decorator import safe_return
from chemicalchecker.util.decorator import cached_property


@logged
class Diagnosis(object):
    """Diagnosis class."""

    def __init__(self, ref_cc, sign, ref_cctype='sign0', ref_molset='full',
                 save=True, plot=True, overwrite=False, n=10000):
        """Initialize a Diagnosis instance.

        Args:
            ref_cc (ChemicalChecker): A CC instance use d as reference.
            sign (CC signature): The CC signature object to be diagnosed.
            save (bool): Whether to save results in the `diags` folder of the
                signature. (default=True)
            plot (bool): Whether to save plots in the `diags` folder of the
                signature. (default=True)
            overwrite (bool): Whether to overwrite the results of the
                diagnosis. (default=False)
            n (int): Number of molecules to sample. (default=10000)
        """
        self.ref_cc = ref_cc
        self.save = save
        self.plot = plot
        self.plotter = DiagnosisPlot(self)
        self.sign = sign
        self.ref_cctype = ref_cctype
        self.ref_molset = ref_molset
        self.subsample_n = n
        # check if reference CC has reference all cctype signatures
        available_sign = ref_cc.report_available(
            molset=ref_molset, signature=ref_cctype)
        if len(available_sign[ref_molset]) < 25:
            self.__log.warning(
                "Reference CC `%s` does not have enough `%s` `%s`" %
                (ref_cc.name, ref_cctype, ref_molset))
            if self.ref_cctype == 'sign0':
                self.ref_cctype = 'sign1'
            else:
                self.ref_cctype = 'sign0'
            self.__log.warning("Switching to `%s`" % self.ref_cctype)
        # define current diag_path
        self.name = '%s_%s_%s' % (ref_cc.name, self.ref_cctype, ref_molset)
        self.path = os.path.join(sign.diags_path, self.name)
        self.overwrite = overwrite
        if self.overwrite:
            if os.path.isdir(self.path):
                self.__log.debug("Deleting %s" % self.path)
                shutil.rmtree(self.path)
                os.mkdir(self.path)
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        if self.plot and not self.save:
            self.__log.warning(
                "Saving is necessary to plot. Setting 'save' to True.")
            self.save = True

        fn = os.path.join(self.sign.diags_path, "subsampled_data.pkl")
        if self.save:
            if not os.path.exists(fn):
                self.__log.debug("Subsampling")
                with open(fn, "wb") as f:
                    V, keys = self.sign.subsample(self.subsample_n)
                    d = {"V": V, "keys": keys}
                    pickle.dump(d, f)
        self.__log.debug("Reading subsamples")
        with open(fn, "rb") as f:
            d = pickle.load(f)
            self.V = d["V"]
            self.keys = d["keys"]

    def _todo(self, fn, inner=False):
        if os.path.exists(os.path.join(self.path, fn + ".pkl")):
            if inner:
                return False
            else:
                if self.overwrite:
                    return True
                else:
                    return False
        else:
            return True

    def _load_diagnosis_pickle(self, fn):
        with open(os.path.join(self.path, fn), "rb") as f:
            results = pickle.load(f)
        return results

    def _returner(self, results, fn, save, plot, plotter_function,
                  kw_plotter=dict()):
        for k, v in kw_plotter.items():
            self.__log.debug('kw_plotter: %s %s', str(k), str(v))
        if results is None:
            fn_ = os.path.join(self.path, fn + ".pkl")
            with open(fn_, "rb") as f:
                results = pickle.load(f)
            if plot:
                plotter_function(**kw_plotter)
            else:
                return results
        else:
            if save:
                fn_ = os.path.join(self.path, fn + ".pkl")
                with open(fn_, "wb") as f:
                    pickle.dump(results, f)
                if plot:
                    return plotter_function(**kw_plotter)
                else:
                    return fn_
            else:
                return results

    def _apply_mappings(self, sign):
        keys = sign.keys
        inchikey_mappings = dict(self.sign.mappings)
        keys = [inchikey_mappings[k] for k in keys]
        return keys

    def _paired_keys(self, my_keys, vs_keys):
        keys = sorted(set(my_keys).intersection(vs_keys))
        my_keys = keys
        vs_keys = keys
        return np.array(my_keys), np.array(vs_keys)

    def _paired_conn_layers(self, my_keys, vs_keys):
        vs_conn_set = set([ik.split("-")[0] for ik in vs_keys])
        my_conn_set = set([ik.split("-")[0] for ik in my_keys])
        common_conn = my_conn_set.intersection(vs_conn_set)
        vs_keys_conn = dict(
            (ik.split("-")[0], ik) for ik in vs_keys if ik.split("-")[0] in common_conn)
        my_keys_conn = dict(
            (ik.split("-")[0], ik) for ik in my_keys if ik.split("-")[0] in common_conn)
        common_conn = sorted(common_conn)
        my_keys = [my_keys_conn[c] for c in common_conn]
        vs_keys = [vs_keys_conn[c] for c in common_conn]
        return np.array(my_keys), np.array(vs_keys)

    def _select_datasets(self, datasets, exemplary):
        if datasets is None:
            datasets = self.ref_cc.datasets
        dss = []
        for ds in datasets:
            if exemplary:
                if ds[0] in "ABCDE" and ds[-3:] == "001":
                    dss += [ds]
            else:
                dss += [ds]
        return dss

    def _distance_distribution(self, n_pairs, metric):
        """Distance distribution. Sampled with replacement.

        Args:
            n_pairs (int): Number of pairs to sample. (default=10000)
            metric (str): 'cosine' or 'euclidean'. (default='cosine')
        """
        if metric == "cosine":
            from scipy.spatial.distance import cosine as metric_func
        elif metric == "euclidean":
            from scipy.spatial.distance import euclidean as metric_func
        else:
            raise "metric needs to be 'cosine' or 'euclidean'"
        dists = []
        n = self.V.shape[0]
        for _ in range(0, n_pairs):
            i, j = np.random.choice(n, 2)
            dists += [metric_func(self.V[i], self.V[j])]
        results = {
            "dists": np.array(sorted(dists))
        }
        return results

    @safe_return(None)
    def euclidean_distances(self, n_pairs=10000):
        """Euclidean distance distribution.

        Args:
            n_pairs (int): Number of pairs to sample. (default=10000)
        """
        self.__log.debug("Euclidean distances")
        fn = "euclidean_distances"
        if self._todo(fn):
            results = self._distance_distribution(
                n_pairs=n_pairs, metric="euclidean")
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.euclidean_distances)

    @safe_return(None)
    def cosine_distances(self, n_pairs=10000):
        """Cosine distance distribution.

        Args:
            n_pairs (int): Number of pairs to sample. (default=10000)
        """
        self.__log.debug("Cosine distances")
        fn = "cosine_distances"
        if self._todo(fn):
            results = self._distance_distribution(
                n_pairs=n_pairs, metric="cosine")
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.cosine_distances)

    @safe_return(None)
    def cross_coverage(self, sign, apply_mappings=False, try_conn_layer=False,
                       save=None, force_redo=False, **kwargs):
        """Intersection of coverages.

        Args:
            sign (signature): A CC signature object to check against.
        """
        fn = os.path.join(self.path,
                          "cross_coverage_%s" % sign.qualified_name)
        if self._todo(fn) or force_redo:
            # apply mappings if necessary
            if apply_mappings:
                my_keys = self._apply_mappings(self.sign)
                vs_keys = self._apply_mappings(sign)
            else:
                my_keys = self.sign.keys
                vs_keys = sign.keys
            # apply inchikey connectivity layer if possible
            if try_conn_layer:
                keys1, keys2 = self._paired_conn_layers(my_keys, vs_keys)
            else:
                keys1, keys2 = self._paired_keys(my_keys, vs_keys)
            results = {
                "inter": len(keys1),
                "my_overlap": len(keys1) / len(my_keys),
                "vs_overlap": len(keys1) / len(vs_keys)
            }
            if save is None:
                save = self.save
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=save,
            plot=self.plot,
            plotter_function=self.plotter.cross_coverage,
            kw_plotter={"sign": sign})

    @safe_return(None)
    def cross_roc(self, sign, n_samples=10000, n_neighbors=5, neg_pos_ratio=1,
                  apply_mappings=False, try_conn_layer=False, metric='cosine',
                  save=None, force_redo=False, **kwargs):
        """Perform validations.

        Args:
            sign (signature): A CC signature object to validate against.
            n_samples (int): Number of samples.
            apply_mappings (bool): Whether to use mappings to compute
                validation. Signature which have been redundancy-reduced
                (i.e. `reference`) have fewer molecules. The key are molecules
                from the `full` signature and values are molecules from the
                `reference` set.
            try_conn_layer (bool): Try with the inchikey connectivity layer.
                (default=False)
            metric (str): 'cosine' or 'euclidean'. (default='cosine')
            save (bool): Specific save parameter. If not specified, the global
                is set. (default=None).
        """
        fn = os.path.join(self.path, "cross_roc_%s" %
                          sign.qualified_name)
        if self._todo(fn) or force_redo:
            r = self.cross_coverage(sign, apply_mappings=apply_mappings,
                                    try_conn_layer=try_conn_layer, save=False,
                                    force_redo=force_redo)
            if r["inter"] < n_neighbors:
                self.__log.warning("Not enough shared molecules")
                return None
            # apply mappings if necessary
            if apply_mappings:
                my_keys = self._apply_mappings(self.sign)
                vs_keys = self._apply_mappings(sign)
            else:
                my_keys = self.sign.keys
                vs_keys = sign.keys
            # apply inchikey connectivity layer if possible
            if try_conn_layer:
                keys1, keys2 = self._paired_conn_layers(my_keys, vs_keys)
            else:
                keys1, keys2 = self._paired_keys(my_keys, vs_keys)
            # Get fraction of shared samples
            if n_samples < len(keys1):
                idxs = sorted(np.random.choice(
                    len(keys1), n_samples, replace=False))
                keys1 = keys1[idxs]
                keys2 = keys2[idxs]
            # extract matrices
            my_vectors = self.sign.get_vectors(keys1)[1]
            vs_vectors = sign.get_vectors(keys2)[1]
            # do nearest neighbors
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
            nn.fit(vs_vectors)
            neighs = nn.kneighbors(vs_vectors)[1][:, 1:]
            # sample positive and negative pairs
            pos_pairs = set()
            neg_pairs = set()
            for i in range(0, len(neighs)):
                for j in neighs[i]:
                    pair = [i, j]
                    pair = sorted(pair)
                    pos_pairs.update([(pair[0], pair[1])])
            for _ in range(0, int(len(pos_pairs) * 10)):
                pair = np.random.choice(len(keys2), 2, replace=False)
                pair = sorted(pair)
                pair = (pair[0], pair[1])
                if pair in pos_pairs:
                    continue
                neg_pairs.update([pair])
                if len(neg_pairs) > len(pos_pairs) * neg_pos_ratio:
                    break
            # do distances
            if metric == "cosine":
                from scipy.spatial.distance import cosine as metric
            if metric == "euclidean":
                from scipy.spatial.distance import euclidean as metric
            y_t = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))
            pairs = list(pos_pairs) + list(neg_pairs)
            y_p = []
            for pair in pairs:
                y_p += [metric(my_vectors[pair[0]], my_vectors[pair[1]])]
            # convert to similarity-respected order
            y_p = -np.abs(np.array(y_p))
            # roc space
            fpr, tpr, _ = roc_curve(y_t, y_p)
            results = {
                "fpr": fpr,
                "tpr": tpr,
                "auc": auc(fpr, tpr),
            }
            if save is None:
                save = self.save
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=save,
            plot=self.plot,
            plotter_function=self.plotter.cross_roc,
            kw_plotter={"sign": sign})

    @safe_return(None)
    def projection(self, keys=None, focus_keys=None, max_keys=10000,
                   perplexity=None, max_pca=100):
        """TSNE projection of CC signatures.

        Args:
            keys (list): Keys to be projected. If None specified, keys are
                randomly sampled. (default=None)
            focus_keys (list): Keys to be highlighted in the projection.
                 (default=None).
            max_keys (int): Maximum number of keys to include in the
                projection. (default=10000)
        """
        self.__log.debug("Projection")
        from MulticoreTSNE import MulticoreTSNE as TSNE
        fn = "projection"
        if self._todo(fn):
            if focus_keys is not None:
                focus_keys = list(set(focus_keys).intersection(self.sign.keys))
                self.__log.debug("%d focus keys found" % len(focus_keys))
                focus_keys = sorted(random.sample(
                    focus_keys, np.min([max_keys, len(focus_keys)])))
                self.__log.debug("Fetching focus signatures")
                X_focus = self.sign.get_vectors(focus_keys)[1]
            else:
                X_focus = None
            if keys is None:
                X = self.V
                keys = self.keys
                if focus_keys is not None:
                    fk = set(focus_keys)
                    idxs = [i for i, k in enumerate(keys) if k not in fk]
                    X = self.V[idxs]
                    keys = np.array(self.keys)[idxs]
            else:
                keys = set(keys).intersection(self.sign.keys)
                if focus_keys is not None:
                    keys = list(keys.difference(focus_keys))
                self.__log.debug("%d keys found" % len(keys))
                keys = sorted(random.sample(
                    keys, np.min([max_keys, len(keys)])))
                if len(keys) == 0:
                    raise Exception("There are no non-focus keys")
                self.__log.debug("Fetching signatures")
                X = self.sign.get_vectors(keys)[1]
            if focus_keys is not None:
                X = np.vstack((X, X_focus))
            self.__log.debug("Fit-transforming t-SNE")
            if X.shape[1] > max_pca:
                self.__log.debug("First doing a PCA")
                X = PCA(n_components=max_pca).fit_transform(X)
            if perplexity is None:
                perp = int(np.sqrt(X.shape[0]))
                perp = np.max([5, perp])
                perp = np.min([50, perp])
            self.__log.debug("Chosen perplexity %d" % perp)
            tsne = TSNE(perplexity=perp)
            P_ = tsne.fit_transform(X)
            P = P_[:len(keys)]
            if focus_keys is not None:
                P_focus = P_[len(keys):]
            else:
                P_focus = None
            results = {
                "P": P,
                "keys": keys,
                "P_focus": P_focus,
                "focus_keys": focus_keys
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.projection)

    @safe_return(None)
    def image(self, keys=None, max_keys=100, shuffle=False):
        self.__log.debug("Image")
        fn = "image"
        if self._todo(fn):
            if keys is None:
                X = self.V
                keys = self.keys
                if shuffle:
                    idxs = np.random.choice(len(keys), np.min(
                        [max_keys, len(keys)]), replace=False)
                    X = X[idxs]
                    keys = np.array(keys)[idxs]
                else:
                    X = X[:max_keys]
                    keys = keys[:max_keys]
            else:
                keys = set(keys).intersection(self.sign.keys)
                self.__log.debug("%d keys found" % len(keys))
                keys = np.array(sorted(random.sample(
                    keys, np.min([max_keys, len(keys)]))))
                X = self.sign.get_vectors(keys)[1]
            # Sort features if available
            features = self.sign.features
            idxs = np.argsort(features)
            X = X[:, idxs]
            results = {
                "X": X,
                "keys": keys
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.image)

    def _iqr(self, axis, keys, max_keys):
        if keys is None:
            X = self.V
            keys = self.keys
            idxs = np.random.choice(len(keys), np.min(
                [max_keys, len(keys)]), replace=False)
            X = X[idxs]
            keys = np.array(keys)[idxs]
        else:
            keys = set(keys).intersection(self.sign.keys)
            self.__log.debug("%d keys found" % len(keys))
            keys = np.array(sorted(random.sample(
                keys, np.min([max_keys, len(keys)]))))
            X = self.sign.get_vectors(keys)[1]
        p25 = np.percentile(X, 25, axis=axis)
        p50 = np.percentile(X, 50, axis=axis)
        p75 = np.percentile(X, 75, axis=axis)
        results = {
            "p25": p25,
            "p50": p50,
            "p75": p75
        }
        return results

    @safe_return(None)
    def features_iqr(self, keys=None, max_keys=10000):
        self.__log.debug("Features IQR")
        fn = "features_iqr"
        if self._todo(fn):
            results = self._iqr(axis=0, keys=keys, max_keys=max_keys)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.features_iqr)

    @safe_return(None)
    def keys_iqr(self, keys=None, max_keys=1000):
        self.__log.debug("Keys IQR")
        fn = "keys_iqr"
        if self._todo(fn):
            results = self._iqr(axis=1, keys=keys, max_keys=max_keys)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.keys_iqr)

    def _bins(self, axis, keys, max_keys, n_bins):
        if keys is None:
            X = self.V
            keys = self.keys
            if len(keys) > max_keys:
                idxs = np.random.choice(len(keys), max_keys, replace=False)
                X = X[idxs]
        else:
            keys = set(keys).intersection(self.sign.keys)
            self.__log.debug("%d keys found" % len(keys))
            keys = np.array(sorted(random.sample(
                keys, np.min([max_keys, len(keys)]))))
            X = self.sign.get_vectors(keys)[1]
        bins = np.linspace(np.min(X), np.max(X), n_bins)
        if axis == 0:
            H = np.zeros((n_bins - 1, X.shape[1]))
            for j in range(0, X.shape[1]):
                H[:, j] = np.histogram(X[:, j], bins=bins)[0]
        elif axis == 1:
            H = np.zeros((n_bins - 1, X.shape[0]))
            for i in range(0, X.shape[0]):
                H[:, i] = np.histogram(X[i, :], bins=bins)[0]
        else:
            raise Exception("Wrong axis, must be 0 (features) or 1 (keys)")
        results = {
            "H": H,
            "bins": bins,
            "p50": np.median(X, axis=axis)
        }
        return results

    @safe_return(None)
    def features_bins(self, keys=None, max_keys=10000, n_bins=100):
        self.__log.debug("Features bins")
        fn = "features_bins"
        if self._todo(fn):
            results = self._bins(
                axis=0, keys=keys, max_keys=max_keys, n_bins=n_bins)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.features_bins)

    @safe_return(None)
    def keys_bins(self, keys=None, max_keys=1000, n_bins=100):
        self.__log.debug("Keys bins")
        fn = "keys_bins"
        if self._todo(fn):
            results = self._bins(
                axis=1, keys=keys, max_keys=max_keys, n_bins=n_bins)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.keys_bins)

    @safe_return(None)
    def values(self, max_values=10000):
        self.__log.debug("Values")
        fn = "values"
        if self._todo(fn):
            V = self.V.ravel()
            idxs = np.random.choice(len(V), min(max_values, len(V)),
                                    replace=False)
            V = V[idxs]
            kernel = gaussian_kde(V)
            positions = np.linspace(np.min(V), np.max(V), 1000)
            values = kernel(positions)
            results = {
                "x": positions,
                "y": values
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.values)

    def _projection_binned(self, P, scores, n_bins):
        # check nans
        mask = ~np.isnan(scores)
        P = P[mask, ]
        scores = scores[mask]
        self.__log.debug("Binning projection")
        bins_x = np.linspace(np.min(P[:, 0]), np.max(P[:, 0]) + 1e-6, n_bins)
        bins_y = np.linspace(np.min(P[:, 1]), np.max(P[:, 1]) + 1e-6, n_bins)
        H = np.zeros((len(bins_y), len(bins_x)))
        S = np.zeros(H.shape)
        for j in range(0, len(bins_x) - 1):
            min_x = bins_x[j]
            max_x = bins_x[j + 1]
            maskx = np.logical_and(P[:, 0] >= min_x, P[:, 0] < max_x)
            P_ = P[maskx]
            scores_ = scores[maskx]
            for i in range(0, len(bins_y) - 1):
                min_y = bins_y[i]
                max_y = bins_y[i + 1]
                masky = np.logical_and(P_[:, 1] >= min_y, P_[:, 1] < max_y)
                ss = scores_[masky]
                if len(ss) > 0:
                    H[i, j] = len(ss)
                    S[i, j] = np.mean(ss)
        bins_x = np.array([(bins_x[i] + bins_x[i + 1]) /
                           2 for i in range(0, len(bins_x) - 1)])
        bins_y = np.array([(bins_y[i] + bins_y[i + 1]) /
                           2 for i in range(0, len(bins_y) - 1)])
        results = {
            "H": H,
            "S": S,
            "bins_x": bins_x,
            "bins_y": bins_y,
            "scores": np.array([np.max(scores), np.min(scores)]),
            "lims": np.array([[np.min(P[:, 0]), np.min(P[:, 1])],
                              [np.max(P[:, 0]), np.max(P[:, 1])]])
        }
        return results

    @safe_return(None)
    def confidences(self):
        self.__log.debug("Confidences")
        fn = "confidences"
        if self._todo(fn):
            V = self.V
            keys = self.keys
            mask = np.isin(
                list(self.sign.keys), list(keys), assume_unique=True)
            confidences = self.sign.get_h5_dataset('confidence')[mask]
            confidences = np.array(confidences)
            kernel = gaussian_kde(confidences)
            positions = np.linspace(
                np.min(confidences), np.max(confidences), 1000)
            values = kernel(positions)
            results = {
                "keys": self.keys,
                "confidences": confidences,
                "x": positions,
                "y": values
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.confidences)

    @safe_return(None)
    def confidences_projection(self, n_bins=20):
        self.__log.debug("Confidences projection")
        if self._todo("confidences", inner=True):
            raise Exception("confidences must be done first")
        if self._todo("projection", inner=True):
            raise Exception("projection must be done first")
        fn = "confidences_projection"
        if self._todo(fn):
            results_proj = self._load_diagnosis_pickle("projection.pkl")
            results_inte = self._load_diagnosis_pickle("confidences.pkl")
            if np.any(results_proj["keys"] != results_inte["keys"]):
                raise Exception("keys do not coincide...")
            results = self._projection_binned(
                results_proj["P"], results_inte["confidences"], n_bins)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.confidences_projection)

    @safe_return(None)
    def intensities(self):
        self.__log.debug("Intensities")
        fn = "intensities"
        if self._todo(fn):
            V = self.V
            keys = self.keys
            m = np.mean(V, axis=0)
            s = np.std(V, axis=0)
            intensities = []
            for i in range(0, V.shape[0]):
                v = np.abs((V[i, :] - m) / s)
                intensities += [np.sum(v)]
            intensities = np.array(intensities)
            kernel = gaussian_kde(intensities)
            positions = np.linspace(
                np.min(intensities), np.max(intensities), 1000)
            values = kernel(positions)
            results = {
                "keys": self.keys,
                "intensities": intensities,
                "x": positions,
                "y": values
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.intensities)

    @safe_return(None)
    def intensities_projection(self, n_bins=20):
        self.__log.debug("Intensities projection")
        if self._todo("intensities", inner=True):
            raise Exception("intensities must be done first")
        if self._todo("projection", inner=True):
            raise Exception("projection must be done first")
        fn = "intensities_projection"
        if self._todo(fn):
            results_proj = self._load_diagnosis_pickle("projection.pkl")
            results_inte = self._load_diagnosis_pickle("intensities.pkl")
            if np.any(results_proj["keys"] != results_inte["keys"]):
                raise Exception("keys do not coincide...")
            results = self._projection_binned(
                results_proj["P"], results_inte["intensities"], n_bins)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.intensities_projection)

    def _latent(self, sign):
        fn = os.path.join(self.path, "latent-%d.pkl" % self.V.shape[0])
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                results = pickle.load(fn)
        else:
            pca = PCA(n_components=0.9)
            V = sign.subsample(self.V.shape[0])[0]
            pca.fit(V)
            results = {"explained_variance": pca.explained_variance_ratio_}
        return results

    @safe_return(None)
    def dimensions(self, datasets=None, exemplary=True, ref_cctype=None,
                   molset=None, **kwargs):
        """Get dimensions of the signature and compare to other signatures."""
        self.__log.debug("Dimensions")

        fn = "dimensions"
        if ref_cctype is None:
            ref_cctype = self.ref_cctype
        if molset is None:
            molset = self.sign.molset
        if self._todo(fn):
            datasets = self._select_datasets(datasets, exemplary)
            results = {}
            for ds in datasets:
                self.__log.debug("Latents for dataset %s" % ds)
                sign = self.ref_cc.get_signature(ref_cctype, molset, ds)
                results[ds] = {
                    "keys": sign.shape[0],
                    "features": sign.shape[1],
                    "expl": self._latent(sign)["explained_variance"]
                }
            results["MY"] = {
                "keys": self.sign.shape[0],
                "features": self.sign.shape[1],
                "expl": self._latent(self.sign)["explained_variance"]
            }
        else:
            results = None
        kw_plotter = {
            "exemplary": exemplary,
            "cctype": ref_cctype,
            "molset": molset}
        kw_plotter.update(kwargs)
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.dimensions,
            kw_plotter=kw_plotter)

    @safe_return(None)
    def across_coverage(self, datasets=None, exemplary=True, ref_cctype=None,
                        molset=None, **kwargs):
        """Check coverage against a collection of other CC signatures.

        Args:
            datasets (list): List of datasets. If None, all available are used.
                (default=None)
            exemplary (bool): Whether to use only exemplary datasets
                (recommended). (default=True)
            cctype (str): CC signature type. (default=None)
            molset (str): Molecule set to use. Full is recommended.
                (default=None)
            kwargs (dict): params of hte cross_coverage method.
        """
        self.__log.debug("Across coverage")
        fn = "across_coverage"
        if ref_cctype is None:
            ref_cctype = self.ref_cctype
        if molset is None:
            molset = self.sign.molset
        if self._todo(fn):
            datasets = self._select_datasets(datasets, exemplary)
            results = {}
            for ds in datasets:
                sign = self.ref_cc.get_signature(ref_cctype, molset, ds)
                results[ds] = self.cross_coverage(
                    sign, save=False, force_redo=True, **kwargs)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.across_coverage,
            kw_plotter={
                "exemplary": exemplary,
                "cctype": ref_cctype,
                "molset": molset})

    def _sample_accuracy_individual(self, sign, n_neighbors, min_shared,
                                    metric):
        # do nearest neighbors (start by the target signature)
        keys, V1 = sign.get_vectors(self.keys)
        if keys is None:
            return None
        if len(keys) < min_shared:
            return None
        n_neighbors = np.min([V1.shape[0], n_neighbors])
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn.fit(V1)
        neighs1_ = nn.kneighbors(V1)[1][:, 1:]
        # do nearest neighbors for self
        keys_set = set(keys)
        idxs = [i for i, k in enumerate(self.keys) if k in keys_set]
        V0 = self.V[idxs]
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn.fit(V0)
        neighs0_ = nn.kneighbors(V0)[1][:, 1:]
        # reindex
        keys_dict = dict((k, i) for i, k in enumerate(self.keys))
        neighs0 = np.zeros(neighs0_.shape).astype(np.int)
        neighs1 = np.zeros(neighs1_.shape).astype(np.int)
        rows = []
        for i in range(0, neighs0_.shape[0]):
            rows += [keys_dict[keys[i]]]
            for j in range(0, neighs0_.shape[1]):
                neighs0[i, j] = keys_dict[keys[neighs0_[i, j]]]
                neighs1[i, j] = keys_dict[keys[neighs1_[i, j]]]
        return {"neighs0": neighs0, "neighs1": neighs1, "rows": rows}

    def _rbo(self, neighs0, neighs1, p):
        from chemicalchecker.tool.rbo.rbo import rbo
        scores = []
        for i in range(0, neighs0.shape[0]):
            scores += [rbo(neighs0[i, :], neighs1[i, :], p=p).ext]
        return np.array(scores)

    @safe_return(None)
    def ranks_agreement(self, datasets=None, exemplary=True, ref_cctype="sign0",
                        molset="full", n_neighbors=100, min_shared=100,
                        metric="minkowski", p=0.9, **kwargs):
        """Sample-specific accuracy.

        Estimated as general agreement with the rest of the CC.
        """
        self.__log.debug("Sample-specific agreement to the rest of CC")
        fn = "ranks_agreement"
        if ref_cctype is None:
            ref_cctype = self.sign.cctype
        if molset is None:
            molset = self.sign.molset

        def q67(r):
            return np.percentile(r, 67)

        def stat(R, func):
            s = []
            for r in R:
                if len(r) == 0:
                    s += [np.nan]
                else:
                    s += [func(r)]
            return np.array(s)

        if self._todo(fn):
            datasets = self._select_datasets(datasets, exemplary)
            results_datasets = {}
            for ds in datasets:
                sign = self.ref_cc.get_signature(ref_cctype, molset, ds)
                dn = self._sample_accuracy_individual(
                    sign, n_neighbors, min_shared, metric)
                if dn is None:
                    continue
                sc = self._rbo(dn["neighs0"], dn["neighs1"], p=p)
                results_datasets[ds] = [r for r in zip(dn["rows"], sc)]
            d = {}
            for k in self.keys:
                d[k] = []
            for k, v in results_datasets.items():
                for r in v:
                    d[self.keys[r[0]]] += [(k, r[1])]
            R = []
            for k in self.keys:
                R += [d[k]]
            R_ = [[x[1] for x in r] for r in R]
            results = {
                "all": R,
                "max": stat(R_, np.max),
                "median": stat(R_, np.median),
                "mean": stat(R_, np.mean),
                "q67": stat(R_, q67),
                "size": np.array([len(r) for r in R_]),
                "keys": self.keys
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.ranks_agreement,
            kw_plotter={
                "exemplary": exemplary,
                "cctype": ref_cctype,
                "molset": molset})

    @safe_return(None)
    def ranks_agreement_projection(self, n_bins=20):
        self.__log.debug("Ranks agreement projection")
        if self._todo("ranks_agreement", inner=True):
            raise Exception("ranks_agreement must be done first")
        if self._todo("projection", inner=True):
            raise Exception("projection must be done first")
        fn = "ranks_agreement_projection"
        if self._todo(fn):
            results_proj = self._load_diagnosis_pickle("projection.pkl")
            results_rnks = self._load_diagnosis_pickle("ranks_agreement.pkl")
            if np.any(results_proj["keys"] != results_rnks["keys"]):
                raise Exception("keys do not coincide...")
            results = self._projection_binned(
                results_proj["P"], results_rnks["mean"], n_bins)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.ranks_agreement_projection)

    @safe_return(None)
    def global_ranks_agreement(self, n_neighbors=100, min_shared=100,
                               metric="minkowski", p=0.9, ref_cctype=None,
                               **kwargs):
        """Sample-specific global accuracy.

        Estimated as general agreement with the rest of the CC, based on a
        Z-global ranking.
        """
        self.__log.debug(
            "Sample-specific agreement to the rest of CC,"
            " based on a Z-global ranking")
        fn = "global_ranks_agreement"

        if ref_cctype is None:
            ref_cctype = self.ref_cctype
        molset = kwargs.get("molset", "full")
        exemplary_datasets = self._select_datasets(None, True)
        datasets = kwargs.get("datasets", exemplary_datasets)

        def q67(r):
            return np.percentile(r, 67)

        def stat(R, func):
            s = []
            for r in R:
                if len(r) == 0:
                    s += [np.nan]
                else:
                    s += [func(r)]
            return np.array(s)

        if self._todo(fn):
            results_datasets = {}
            for ds in datasets:
                sign = self.ref_cc.get_signature(ref_cctype, molset, ds)
                dn = self._sample_accuracy_individual(
                    sign, n_neighbors, min_shared, metric)
                if dn is None:
                    continue
                sc = self._rbo(dn["neighs0"], dn["neighs1"], p=p)
                results_datasets[ds] = [r for r in zip(dn["rows"], sc)]
            d = {}
            for k in self.keys:
                d[k] = []
            for k, v in results_datasets.items():
                for r in v:
                    d[self.keys[r[0]]] += [(k, r[1])]
            R = []
            for k in self.keys:
                R += [d[k]]
            R_ = [[x[1] for x in r] for r in R]
            results = {
                "all": R,
                "max": stat(R_, np.max),
                "median": stat(R_, np.median),
                "mean": stat(R_, np.mean),
                "q67": stat(R_, q67),
                "size": np.array([len(r) for r in R_]),
                "keys": self.keys
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.global_ranks_agreement)

    @safe_return(None)
    def global_ranks_agreement_projection(self, n_bins=20):
        self.__log.debug("Global ranks agreement projection")
        if self._todo("global_ranks_agreement", inner=True):
            raise Exception("global_ranks_agreement must be done first")
        if self._todo("projection", inner=True):
            raise Exception("projection must be done first")
        fn = "global_ranks_agreement_projection"
        if self._todo(fn):
            results_proj = self._load_diagnosis_pickle("projection.pkl")
            results_rnks = self._load_diagnosis_pickle(
                "global_ranks_agreement.pkl")
            if np.any(results_proj["keys"] != results_rnks["keys"]):
                raise Exception("keys do not coincide...")
            results = self._projection_binned(
                results_proj["P"], results_rnks["mean"], n_bins)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.global_ranks_agreement_projection)

    @safe_return(None)
    def across_roc(self, datasets=None, exemplary=True, ref_cctype=None,
                   molset="full", **kwargs):
        """Check coverage against a collection of other CC signatures.

        Args:
            datasets (list): List of datasets. If None, all available are used.
                (default=None).
            exemplary (bool): Whether to use only exemplary datasets
                (recommended). (default=True)
            cctype (str): CC signature type. (default='sign0')
            molset (str): Molecule set to use. Full is recommended.
                (default='full')
            kwargs (dict): Parameters of hte cross_coverage method.
        """
        self.__log.debug("Across ROC")
        fn = "across_roc"
        if ref_cctype is None:
            ref_cctype = self.ref_cctype
        if molset is None:
            molset = self.sign.molset
        if self._todo(fn):
            datasets = self._select_datasets(datasets, exemplary)
            results = {}
            for ds in datasets:
                sign = self.ref_cc.get_signature(ref_cctype, molset, ds)
                results[ds] = self.cross_roc(
                    sign, save=False, force_redo=True, **kwargs)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.across_roc,
            kw_plotter={
                "exemplary": exemplary,
                "cctype": ref_cctype,
                "molset": molset})

    @safe_return(None)
    def atc_roc(self, ref_cctype=None, **kwargs):
        self.__log.debug("ATC ROC")
        fn = "atc_roc"
        if ref_cctype is None:
            ref_cctype = self.ref_cctype
        molset = "full"
        ds = "E1.001"
        if self._todo(fn):
            sign = self.ref_cc.get_signature(ref_cctype, molset, ds)
            results = self.cross_roc(
                sign, save=False, force_redo=True, **kwargs)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.atc_roc)

    @safe_return(None)
    def moa_roc(self, ref_cctype=None, **kwargs):
        self.__log.debug("MoA ROC")
        fn = "moa_roc"
        if ref_cctype is None:
            ref_cctype = self.ref_cctype
        molset = "full"
        ds = "B1.001"
        if self._todo(fn):
            sign = self.ref_cc.get_signature(ref_cctype, molset, ds)
            results = self.cross_roc(
                sign, save=False, force_redo=True, **kwargs)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.moa_roc)

    @safe_return(None)
    def redundancy(self, cpu=4):
        from chemicalchecker.util.remove_near_duplicates import RNDuplicates
        self.__log.debug("Redundancy")
        fn = "redundancy"
        if self._todo(fn):
            self.__log.debug("Removing near duplicates")
            rnd = RNDuplicates(cpu=cpu)
            mappings = rnd.remove(self.sign.data_path,
                                  save_dest=None, just_mappings=True)
            mappings = np.array(sorted(mappings.items()))
            mps = []
            for i in range(0, mappings.shape[0]):
                mps += [(mappings[i, 0], mappings[i, 1])]
            mps = np.array(mps)
            n_full = mps.shape[0]
            n_ref = len(set(mps[:, 1]))
            red_counts = collections.defaultdict(int)
            for i in range(0, mps.shape[0]):
                red_counts[mps[i, 1]] += 1
            red_counts = [(k, v) for k, v in sorted(
                red_counts.items(), key=lambda item: -item[1])]
            results = {
                "n_full": n_full,
                "n_ref": n_ref,
                "counts": red_counts
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.redundancy)

    def _cluster(self, expected_coverage, n_neighbors, min_samples,
                 top_clusters):
        self.__log.debug("Clustering")
        P = self._load_diagnosis_pickle("projection.pkl")["P"]
        min_samples = int(
            np.max([np.min([P.shape[0] * 0.01, min_samples]), 5]))
        self.__log.debug("Estimating epsilon")
        if n_neighbors is None:
            n_neighbors = [1, 2, 3, 5, 10]
        else:
            n_neighbors = [n_neighbors]

        def do_clustering(n_neigh):
            nn = NearestNeighbors(n_neigh + 1)
            nn.fit(P)
            dists = nn.kneighbors(P)[0][:, n_neigh]
            h = np.histogram(dists, bins="auto")
            y = np.cumsum(h[0]) / np.sum(h[0])
            x = h[1][1:]
            eps = x[np.where(y > expected_coverage)[0][0]]
            self.__log.debug("Running DBSCAN")
            cl = DBSCAN(eps=eps, min_samples=min_samples).fit(P)
            labels = cl.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            lab_counts = collections.defaultdict(int)
            for lab in labels:
                lab_counts[lab] += 1
            lab_counts = [(k, v) for k, v in sorted(
                lab_counts.items(), key=lambda item: -item[1])]
            return (labels, lab_counts, eps, n_clusters_, n_noise_)
        best_score = 0
        best_n_neigh = None
        for n_neigh in n_neighbors:
            labels, lab_counts, eps, n_clusters_, n_noise_ = do_clustering(
                n_neigh)
            lc = list([x[1] for x in lab_counts[:top_clusters]])
            if len(lc) < top_clusters:
                lc += [0] * (top_clusters - len(lc))
            score = np.median(lc)
            if best_n_neigh is None:
                best_n_neigh = n_neigh
                best_score = score
            if best_score < score:
                best_n_neigh = n_neigh
                best_score = score
        labels, lab_counts, eps, n_clusters_, n_noise_ = do_clustering(
            best_n_neigh)
        results = {
            "P": P,
            "min_samples": min_samples,
            "labels": labels,
            "lab_counts": lab_counts,
            "epsilon": eps,
            "n_clusters": n_clusters_,
            "n_noise": n_noise_
        }
        return results

    @safe_return(None)
    def cluster_sizes(self, expected_coverage=0.95, n_neighbors=None,
                      min_samples=10, top_clusters=20):
        if self._todo("projection", inner=True):
            raise Exception("projection needs to be done first")
        self.__log.debug("Cluster sizes")
        fn = "clusters"
        if self._todo(fn):
            results = self._cluster(
                expected_coverage, n_neighbors, min_samples, top_clusters)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.cluster_sizes)

    @safe_return(None)
    def clusters_projection(self):
        self.__log.debug("Projection clusters")
        if self._todo("projection", inner=True):
            raise Exception("projection needs to be done first")
        fn = "clusters"
        if self._todo(fn, inner=True):
            raise Exception("clusters needs to be done first")
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.clusters_projection)

    @safe_return(None)
    def key_coverage(self, datasets=None, exemplary=True, cctype=None,
                     molset=None, **kwargs):
        self.__log.debug("Key coverages")
        fn = "key_coverage"
        if cctype is None:
            cctype = self.sign.cctype
        if molset is None:
            molset = self.sign.molset
        if self._todo(fn):
            datasets = self._select_datasets(datasets, exemplary)
            results_shared = {}
            for ds in datasets:
                sign = self.ref_cc.get_signature(cctype, molset, ds)
                results_shared[ds] = sorted(
                    set(self.keys).intersection(sign.keys))
            results_counts = {}
            for k in self.keys:
                results_counts[k] = 0
            for k, v in results_shared.items():
                for k in v:
                    results_counts[k] += 1
            results = {
                "shared": results_shared,
                "counts": results_counts,
                "key_counts": np.array([results_counts[k] for k in self.keys])
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.key_coverage,
            kw_plotter={
                "exemplary": exemplary
            })

    @safe_return(None)
    def key_coverage_projection(self, n_bins=20):
        if self._todo("key_coverage", inner=True):
            raise Exception("key_coverage needs to be run first")
        if self._todo("projection", inner=True):
            raise Exception("projection needs to be done first")
        fn = "key_coverage_projection"
        if self._todo(fn):
            results_proj = self._load_diagnosis_pickle("projection.pkl")
            results_cov = self._load_diagnosis_pickle("key_coverage.pkl")
            results = self._projection_binned(results_proj["P"], np.array(
                results_cov["key_counts"]), n_bins=n_bins)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.key_coverage_projection)

    @safe_return(None)
    def orthogonality(self, max_features=1000):
        fn = "orthogonality"
        if self._todo(fn):
            if max_features < self.V.shape[1]:
                idxs = np.random.choice(
                    self.V.shape[1], max_features, replace=False)
                V = self.V[:, idxs]
            else:
                V = self.V
            V = normalize(V, norm="l2", axis=0)
            dots = []
            for i in range(0, V.shape[1] - 1):
                for j in range(i + 1, V.shape[1]):
                    dots += [np.dot(V[:, i], V[:, j].T)]
            results = {
                "dots": np.array(dots)
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.orthogonality)

    @safe_return(None)
    def outliers(self, n_estimators=1000):
        fn = "outliers"
        if self._todo(fn):
            max_features = int(np.sqrt(self.V.shape[1]) + 1)
            max_samples = min(1000, int(self.V.shape[0] / 2 + 1))
            mod = IsolationForest(n_estimators=n_estimators, contamination=0.1,
                                  max_samples=max_samples,
                                  max_features=max_features)
            pred = mod.fit_predict(self.V)
            scores = mod.score_samples(self.V)
            results = {
                "scores": scores,
                "pred": pred
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            save=self.save,
            plot=self.plot,
            plotter_function=self.plotter.outliers)

    def available(self):
        return self.plotter.available()

    def canvas_medium(self, ref_cctype, title):
        self.__log.debug("Getting all needed data.")
        plot = self.plot
        self.plot = False
        save = self.save
        self.save = True
        self.cosine_distances()
        self.euclidean_distances()
        self.projection()
        self.image()
        self.features_bins()
        self.keys_bins()
        self.across_coverage(ref_cctype=ref_cctype)
        self.across_roc()
        self.dimensions(ref_cctype=ref_cctype)
        self.values()
        self.atc_roc()
        self.moa_roc()
        self.redundancy()
        self.intensities()
        self.intensities_projection()
        if self.sign.cctype == 'sign3':
            self.confidences()
            self.confidences_projection()
        self.cluster_sizes()
        self.clusters_projection()
        self.key_coverage(ref_cctype=ref_cctype)
        self.key_coverage_projection()
        self.global_ranks_agreement()
        self.global_ranks_agreement_projection()
        self.outliers()
        self.plot = plot
        self.save = save
        self.__log.debug("Plotting")
        fig = self.plotter.canvas_medium(title=title)
        return fig

    def canvas(self, ref_cctype="sign0", size="medium", title=None):
        if size == "small":
            return self.canvas_small(ref_cctype=ref_cctype, title=title)
        elif size == "medium":
            return self.canvas_medium(ref_cctype=ref_cctype, title=title)
        elif size == "large":
            return self.canvas_large(ref_cctype=ref_cctype, title=title)
        else:
            return None
