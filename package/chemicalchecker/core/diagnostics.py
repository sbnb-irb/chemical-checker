import numpy as np
import random
import pickle
import os
from chemicalchecker.util import logged
from chemicalchecker.util.plot import DiagnosisPlot


@logged
class Diagnosis(object):

    def __init__(self, cc, sign, save=True, plot=True, overwrite=False, n=10000):
        """Diagnosis for signatures.

        Args:
            cc(ChemicalChecker): A CC instance.
            sign(CC signature): The CC signature object to be diagnosed.
            save(bool): Whether to save results in the `stats` folder of the signature (default=True).
            plot(bool): Whether to save plots in the `stats` folder of the signature (default=True).
            overwrite(bool): Whether to overwrite the results of the diagnosis (default=False).
            n(int): Number of molecules to sample (default=10000)
        """
        self.cc = cc
        self.save = save
        self.plot = plot
        self.plotter = DiagnosisPlot(cc, sign)
        self.overwrite = overwrite
        if self.plot and not self.save:
            self.__log.warning("Saving is necessary to plot. Setting 'save' to True.")
            self.save=True
        self.sign = sign
        folds = self.sign.data_path.split("/")
        self.cctype = folds[-2]
        self.dataset = folds[-3]
        self.molset = folds[-6]
        V, keys = self.sign.subsample(n)
        self.V = V
        self.keys = keys
        if self.save:
            fn = os.path.join(self.sign.stats_path, "subsampled_data.pkl")
            pickle.dump({"V": V, "keys": keys}, open(fn, "wb"))
 
    def _todo(self, fn):
        if os.path.exists(os.path.join(self.sign.stats_path, fn+".pkl")) and not self.overwrite:
            return False
        else:
            return True

    def _returner(self, results, fn, save, plot, plotter_function, kw_plotter=None):
        if results is None:
            fn_ = os.path.join(self.sign.stats_path, fn+".pkl")
            with open(fn_, "rb") as f:
                results = pickle.load(f)
            if plot:
                if kw_plotter is None:
                    return plotter_function()
                else:
                    return plotter_function(**kw_plotter)
            else:
                return results
        else:
            if save:
                fn_ = os.path.join(self.sign.stats_path, fn+".pkl")
                with open(fn_, "wb") as f:
                    pickle.dump(results, f)
                if plot:
                    if kw_plotter is None:
                        return plotter_function()
                    else:
                        return plotter_function(**kw_plotter)
                else:
                    return fn_
            else:
                return results

    def _apply_mappings(self, sign):
        keys = sign.keys
        if 'mappings' not in self.sign.info_h5:
            self.__log.warning("Cannot apply mappings in validation.")
        else:
            inchikey_mappings = dict(self.sign.get_h5_dataset('mappings'))
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
            datasets = self.cc.datasets
        dss = []
        for ds in self.cc.datasets:
            if exemplary:
                if ds[0] in "ABCDE" and ds[-3:] == "001":
                    dss += [ds]
            else:
                dss += [ds]
        return dss

    def _distance_distribution(self, n_pairs, metric):
        """Distance distribution. Sampled with replacement.

        Args:
            n_pairs(int): Number of pairs to sample (default=10000).
            metric(str): 'cosine' or 'euclidean' (default='cosine').
        """
        if   metric == "cosine":
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

    def euclidean_distances(self, n_pairs=10000):
        """Euclidean distance distribution.

            Args:
                n_pairs(int): Number of pairs to sample (default=10000)
        """
        fn = "euclidean_distances"
        if self._todo(fn):
            results = self._distance_distribution(n_pairs=n_pairs, metric="euclidean")
        else:
            results = None
        return self._returner(
            results=results,
            fn = fn,
            save = self.save,
            plot = self.plot,
            plotter_function = self.plotter.euclidean_distances)

    def cosine_distances(self, n_pairs=10000):
        """Cosine distance distribution.

            Args:
                n_pairs(int): Number of pairs to sample (default=10000).
        """
        fn = "cosine_distances"
        if self._todo(fn):
            results = self._distance_distribution(n_pairs=n_pairs, metric="cosine")
        else:
            results = None
        return self._returner(
            results=results,
            fn = fn,
            save = self.save,
            plot = self.plot,
            plotter_function = self.plotter.cosine_distances)
        
    def cross_coverage(self, sign, apply_mappings=False, try_conn_layer=False, save=None, force_redo=False, **kwargs):
        """Intersection of coverages.

        Args:
            sign(signature object): A CC signature object to check against.
        """
        fn = os.path.join(self.sign.stats_path, "cross_coverage_%s" % self.cc.sign_name(sign))
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
                "inter"     : len(keys1),
                "my_overlap": len(keys1) / len(my_keys),
                "vs_overlap": len(keys1) / len(vs_keys)
            }
            if save is None:
                save = self.save
        else:
            results = None
        return self._returner(
            results=results,
            fn = fn,
            save = save,
            plot = self.plot,
            plotter_function = self.plotter.cross_coverage,
            kw_plotter = {"sign": sign})

    def cross_roc(self, sign, n_samples=1000, n_neighbors=5, neg_pos_ratio=1, apply_mappings=False, try_conn_layer=False, metric='cosine', save=None, force_redo=False, **kwargs):
        """Perform validations.

        Args:
            sign(signature object): A CC signature object to validate against
            n_samples(int): Number of samples
            apply_mappings(bool): Whether to use mappings to compute
                validation. Signature which have been redundancy-reduced
                (i.e. `reference`) have fewer molecules. The key are molecules
                from the `full` signature and values are molecules from the
                `reference` set.
            try_conn_layer(bool): Try with the inchikey connectivity layer (default=False).
            metric(str): 'cosine' or 'euclidean' (default='cosine').
            save(bool): Specific save parameter. If not specified, the global is set (default=None).
        """
        fn = os.path.join(self.sign.stats_path, "cross_roc_%s" % self.cc.sign_name(sign))
        if self._todo(fn) or force_redo:
            r = self.cross_coverage(sign, apply_mappings=apply_mappings, try_conn_layer=try_conn_layer, save=False, force_redo=force_redo)
            if r["inter"] < n_neighbors:
                self.__log.warning("Not enough shared molecules")
                return None
            from sklearn.neighbors import NearestNeighbors
            from sklearn.metrics import roc_curve, auc
            import random
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
                idxs = sorted(np.random.choice(len(keys1), n_samples, replace=False))
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
            for _ in range(0, int(len(pos_pairs)*10)):
                pair = np.random.choice(len(keys2), 2, replace=False)
                pair = sorted(pair)
                pair = (pair[0], pair[1])
                if pair in pos_pairs:
                    continue
                neg_pairs.update([pair])
                if len(neg_pairs) > len(pos_pairs)*neg_pos_ratio:
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
            fn = fn,
            save = save,
            plot = self.plot,
            plotter_function = self.plotter.cross_roc,
            kw_plotter = {"sign": sign})

    def projection(self, keys=None, focus_keys=None, max_keys=10000):
        """TSNE projection of CC signatures.

            Args:
                keys(list): Keys to be projected. If None specified, keys are randomly sampled (default=None).
                focus_keys(list): Keys to be highlighted in the projection (default=None).
                max_keys(int): Maximum number of keys to include in the projection (default=10000).
        """
        from sklearn.manifold import TSNE
        fn = "projection"
        if self._todo(fn):
            if focus_keys is not None:
                focus_keys = list(set(focus_keys).intersection(self.sign.keys))
                self.__log.debug("%d focus keys found" % len(focus_keys))
                focus_keys = sorted(random.sample(focus_keys, np.min([max_keys, len(focus_keys)])))
                self.__log.debug("Fetching focus signatures")
                X_focus = self.sign.get_vectors(focus_keys)[1]
            else:
                X_focus = None
            if keys is None:
                X = self.V
                keys = self.keys
                if focus_keys is not None:
                    fk = set(focus_keys)
                    idxs = [i for i,k in enumerate(keys) if k not in fk]
                    X = self.V[idxs]
                    keys = np.array(self.keys)[idxs]
            else:
                keys = set(keys).intersection(self.sign.keys)
                if focus_keys is not None:
                    keys = list(keys.difference(focus_keys))
                self.__log.debug("%d keys found" % len(keys))
                keys = sorted(random.sample(keys, np.min([max_keys, len(keys)])))
                if len(keys) == 0:
                    raise Exception("There are no non-focus keys")
                self.__log.debug("Fetching signatures")
                X = self.sign.get_vectors(keys)[1]
            if focus_keys is not None:
                X = np.vstack((X, X_focus))
            self.__log.debug("Fit-transforming t-SNE")
            tsne = TSNE()
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
            results = results,
            fn = fn,
            save = self.save,
            plot = self.plot,
            plotter_function = self.plotter.projection)

    def image(self, keys=None, max_keys=100):
        fn = "image"
        if self._todo(fn):
            if keys is None:
                X = self.V
                keys = self.keys
                idxs = np.random.choice(len(keys), np.min([max_keys, len(keys)]), replace=False)
                X = X[idxs]
                keys = np.array(keys)[idxs]
            else:
                keys = set(keys).intersection(self.sign.keys)
                self.__log.debug("%d keys found" % len(keys))
                keys = np.array(sorted(random.sample(keys, np.min([max_keys, len(keys)]))))
                X = self.sign.get_vectors(keys)[1]
            results = {
                "X": X,
                "keys": keys
            }
        else:
            results = None
        return self._returner(
            results = results,
            fn = fn,
            save = self.save,
            plot = self.plot,
            plotter_function = self.plotter.image)

    def _iqr(self, axis, keys, max_keys):
        if keys is None:
            X = self.V
            keys = self.keys
            idxs = np.random.choice(len(keys), np.min([max_keys, len(keys)]), replace=False)
            X = X[idxs]
            keys = np.array(keys)[idxs]
        else:
            keys = set(keys).intersection(self.sign.keys)
            self.__log.debug("%d keys found" % len(keys))
            keys = np.array(sorted(random.sample(keys, np.min([max_keys, len(keys)]))))
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

    def features_iqr(self, keys=None, max_keys=10000):
        fn = "features_iqr"
        if self._todo(fn):
            results = self._iqr(axis=0, keys=keys, max_keys=max_keys)
        else:
            results = None
        return self._returner(
            results = results,
            fn = fn,
            save = self.save,
            plot = self.plot,
            plotter_function = self.plotter.features_iqr)

    def keys_iqr(self, keys=None, max_keys=100000):
        fn = "keys_iqr"
        if self._todo(fn):
            results = self._iqr(axis=1, keys=keys, max_keys=max_keys)
        else:
            results = None
        return self._returner(
            results = results,
            fn = fn,
            save = self.save,
            plot = self.plot,
            plotter_function = self.plotter.keys_iqr)

    def values(self):
        from scipy.stats import gaussian_kde
        fn = "values"
        if self._todo(fn):
            V = self.V.ravel().ravel()
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
            results = results,
            fn = fn,
            save = self.save,
            plot = self.plot,
            plotter_function = self.plotter.values)

    def _latent(self, sign):
        from sklearn.decomposition import PCA
        fn = os.path.join(sign.stats_path, "latent-%d.pkl" % self.V.shape[0])
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                results = pickle.load(fn)
        else:
            pca = PCA(n_components=0.9)
            V = sign.subsample(self.V.shape[0])[0]
            pca.fit(V)
            results = {"explained_variance": pca.explained_variance_ratio_}
        return results

    def dimensions(self, datasets=None, exemplary=True, cctype=None, molset=None, **kwargs):
        """Get dimensions of the signature and compare to other signatures"""
        from sklearn.decomposition import PCA
        fn = "dimensions"
        if cctype is None:
            cctype = self.cctype
        if molset is None:
            molset = self.molset
        if self._todo(fn):
            datasets = self._select_datasets(datasets, exemplary)
            results = {}
            for ds in datasets:
                self.__log.debug("Latents for dataset %s" % ds)
                sign = self.cc.get_signature(cctype, molset, ds)
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
        return self._returner(
            results = results,
            fn = fn,
            save = self.save,
            plot = self.plot,
            plotter_function = self.plotter.dimensions,
            kw_plotter = {
                "exemplary": exemplary,
                "cctype": cctype,
                "molset": molset})

    def across_coverage(self, datasets=None, exemplary=True, cctype=None, molset=None, **kwargs):
        """Check coverage against a collection of other CC signatures.
        
        Args:
            datasets(list): List of datasets. If None, all available are used (default=None).
            exemplary(bool): Whether to use only exemplary datasets (recommended) (default=True).
            cctype(str): CC signature type (default=None).
            molset(str): Molecule set to use. Full is recommended (default=None).
            **kwargs of hte cross_coverage method.
        """
        fn = "across_coverage"
        if cctype is None:
            cctype = self.cctype
        if molset is None:
            molset = self.molset
        if self._todo(fn):
            datasets = self._select_datasets(datasets, exemplary)
            results = {}
            for ds in datasets:
                sign = self.cc.get_signature(cctype, molset, ds)
                results[ds] = self.cross_coverage(sign, save=False, force_redo=True, **kwargs)
        else:
            results = None
        return self._returner(
            results = results,
            fn = fn,
            save = self.save,
            plot = self.plot,
            plotter_function = self.plotter.across_coverage,
            kw_plotter = {
                "exemplary": exemplary,
                "cctype": cctype,
                "molset": molset})

    def across_roc(self, datasets=None, exemplary=True, cctype="sign1", molset="full", **kwargs):
        """Check coverage against a collection of other CC signatures.
        
        Args:
            datasets(list): List of datasets. If None, all available are used (default=None).
            exemplary(bool): Whether to use only exemplary datasets (recommended) (default=True).
            cctype(str): CC signature type (default='sign1').
            molset(str): Molecule set to use. Full is recommended (default='full').
            **kwargs of hte cross_coverage method.
        """
        fn = "across_roc"
        if cctype is None:
            cctype = self.cctype
        if molset is None:
            molset = self.molset
        if self._todo(fn):
            datasets = self._select_datasets(datasets, exemplary)
            results = {}
            for ds in datasets:
                sign = self.cc.get_signature(cctype, molset, ds)
                results[ds] = self.cross_roc(sign, save=False, force_redo=True, **kwargs)
        else:
            results = None
        return self._returner(
            results = results,
            fn = fn,
            save = self.save,
            plot = self.plot,
            plotter_function = self.plotter.across_roc,
            kw_plotter = {
                "exemplary": exemplary,
                "cctype": cctype,
                "molset": molset})

    def atc_roc(self, **kwargs):
        fn = "atc_roc"
        cctype = "sign1"
        molset = "full"
        ds = "E1.001"
        if self._todo(fn):
            sign = self.cc.get_signature(cctype, molset, ds)
            print(sign)
            results = self.cross_roc(sign, save=False, force_redo=True, **kwargs)
        else:
            results = None
        return self._returner(
            results = results,
            fn = fn,
            save = self.save,
            plot = self.plot,
            plotter_function = self.plotter.atc_roc)

    def moa_roc(self, **kwargs):
        fn = "moa_roc"
        cctype = "sign1"
        molset = "full"
        ds = "B1.001"
        if self._todo(fn):
            sign = self.cc.get_signature(cctype, molset, ds)
            results = self.cross_roc(sign, save=False, force_redo=True, **kwargs)
        else:
            results = None
        return self._returner(
            results = results,
            fn = fn,
            save = self.save,
            plot = self.plot,
            plotter_function = self.plotter.moa_roc)

    def available(self):
        return self.plotter.available()

    def canvas(self):
        self.__log.debug("Getting all needed data.")
        plot = self.plot
        self.plot = False
        save = self.save
        self.save = True
        self.cosine_distances()
        self.euclidean_distances()
        self.projection()
        self.image()
        self.features_iqr()
        self.keys_iqr()
        self.across_coverage()
        self.across_roc()
        self.dimensions()
        self.values()
        self.atc_roc()
        self.moa_roc()
        self.plot = plot
        self.save = save
        self.__log.debug("Plotting")
        fig = self.plotter.canvas()
        return fig