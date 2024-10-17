"""Cluster Signature.

Performs K-means clustering.
"""
import os
import csv
import glob
import h5py
import json
import bisect
import shelve
import joblib
import tempfile
import datetime
import numpy as np
from csvsort import csvsort
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import euclidean, pdist

from .signature_base import BaseSignature
from .signature_data import DataSignature

from chemicalchecker.util import logged
from chemicalchecker.util import Config
from chemicalchecker.util.plot import Plot


@logged
class clus(BaseSignature, DataSignature):
    """Cluster Signature class."""

    def __init__(self, signature_path, dataset, **params):
        """Initialize a Signature.

        Args:
            signature_path(str): the path to the signature directory.
            dataset(object): The dataset object with all info related
            type(str): The type of clustering between kmeans and hdbscan.
                (default:kmeans)
            metric(str): The metric used in the KNN algorithm: euclidean or
                cosine (default: cosine)
            k_neig(int): The number of k neighbours to search for
                (default:None)
            cpu(int): The number of cores to use (default:1)
            min_members(int): Minimum number of points per cluster (hdbscan)
                (default:5)
            num_subdim(int): Splitting of the PQ encoder (kmeans) (default:8)
            min_k(int): Minimum number of clusters (kmeans)(default:1)
            max_k(int): Maximum number of clusters (kmeans) (default: None)
            n_points(int): Number of points to calculate (kmeans) (default:100)
            balance(float): If 1, all clusters are of equal size. Greater
                values are increasingly more imbalanced (kmeans) (default:None)
            significance(float): Distance significance cutoff (kmeans)
                (default:0.05)
        """
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(self, signature_path, dataset, **params)
        self.__log.debug('signature path is: %s', signature_path)
        self.data_path = os.path.join(signature_path, "clus.h5")
        DataSignature.__init__(self, self.data_path)
        self.__log.debug('data_path: %s', self.data_path)
        self.clustencoder_file = "clustencoder.h5"
        self.clustcentroids_file = "clustcentroids.h5"
        self.clust_info_file = "clust_stats.json"
        self.clust_output = 'clust.h5'
        self.bg_pq_euclideans_file = "bg_pq_euclideans.h5"
        self.hdbscan_file = "hdbscan.pkl"

        self.type = "kmeans"
        self.cpu = 1
        self.k_neig = None
        self.min_members = 5
        self.num_subdim = 8
        self.min_k = 1
        self.max_k = None
        self.n_points = 100
        self.balance = None
        self.significance = 0.05
        self.metric = "euclidean"
        for param, value in params.items():
            self.__log.debug('parameter %s : %s', param, value)
            if "metric" in params:
                self.metric = params["metric"]
            if "cpu" in params:
                self.cpu = params["cpu"]
            if "k_neig" in params:
                self.k_neig = params["k_neig"]
            if "min_members" in params:
                self.min_members = params["min_members"]
            if "num_subdim" in params:
                self.num_subdim = params["num_subdim"]
            if "min_k" in params:
                self.min_k = params["min_k"]
            if "max_k" in params:
                self.max_k = params["max_k"]
            if "n_points" in params:
                self.n_points = params["n_points"]
            if "balance" in params:
                self.balance = params["balance"]
            if "significance" in params:
                self.significance = params["significance"]
            if "type" in params:
                self.type = params["type"]

    def fit(self, sign=None, validations=True):
        """Fit cluster model given a signature."""
        try:
            import faiss
        except ImportError:
            raise ImportError("requires faiss " +
                              "https://github.com/facebookresearch/faiss")
        try:
            import hdbscan
        except ImportError:
            raise ImportError("requires hdbscan " +
                              "https://hdbscan.readthedocs.io/en/latest/")
        BaseSignature.fit(self)

        plot = Plot(self.dataset, self.stats_path)

        mappings = None

        if sign is None:
            sign = self.get_sign(
                'sign' + self.cctype[-1]).get_molset("reference")

        if os.path.isfile(sign.data_path):
            self.data = sign.data.astype( 'float32' )
            self.data_type = self.data.dtype
            self.keys = sign.keys
            mappings = sign.mappings
        else:
            raise Exception("The file " + sign.data_path + " does not exist")

        tmp_dir = tempfile.mkdtemp(
            prefix='clus_' + self.dataset + "_", dir=Config().PATH.CC_TMP)

        self.__log.debug("Temporary files saved in " + tmp_dir)

        if self.type == "hdbscan":
            self.__log.info("Calculating HDBSCAN clusters")

            clusterer = hdbscan.HDBSCAN(min_cluster_size=int(
                np.max([2, self.min_members])), prediction_data=True).fit(self.data)

            self.__log.info("Saving the model")

            joblib.dump(clusterer, os.path.join(
                self.model_path, self.hdbscan_file))

            self.__log.info("Predicting...")

            labels, strengths = hdbscan.approximate_predict(
                clusterer, self.data)

            # Save

            self.__log.info("Saving matrix...")

            with h5py.File(self.data_path, "w") as hf:
                hf.create_dataset("labels", data=labels)
                hf.create_dataset("keys", data=np.array(
                    self.keys, DataSignature.string_dtype()))
                hf.create_dataset("strengths", data=strengths)

            if validations:

                self.__log.info("Doing validations")

                if mappings is not None:
                    inchikey_mappings = dict(mappings)
                else:
                    inchikey_mappings = None

                inchikey_clust = shelve.open(
                    os.path.join(tmp_dir, "clus.dict"), "n")
                for i in range(len(self.keys)):
                    lab = labels[i]
                    if lab == -1:
                        continue
                    inchikey_clust[str(self.keys[i])] = lab
                odds_moa, pval_moa = plot.label_validation(
                    inchikey_clust, "clus", prefix="moa", inchikey_mappings=inchikey_mappings)
                odds_atc, pval_atc = plot.label_validation(
                    inchikey_clust, "clus", prefix="atc", inchikey_mappings=inchikey_mappings)
                inchikey_clust.close()

            self.__log.info("Cleaning")
            for filename in glob.glob(os.path.join(tmp_dir, "clus.dict*")):
                os.remove(filename)

            os.rmdir(tmp_dir)

            self.metric = "hdbscan"

        if self.type == "kmeans":

            faiss.omp_set_num_threads(self.cpu)

            with h5py.File(sign.data_path, 'r') as dh5:
                if "elbow" not in dh5.keys():
                    Vn, Vm = self.data.shape[0], self.data.shape[1] / 2
                else:
                    Vn, Vm = self.data.shape[0], dh5["elbow"][0]

            if self.metric == "cosine":
                self.data = self._normalizer(self.data, False)

            if self.data.shape[1] < self.num_subdim:
                self.data = np.hstack(
                    (self.data,
                        np.zeros((self.data.shape[0],
                                  self.num_subdim - self.data.shape[1]))))
                self.data = self.data.astype( 'float32' )

            self.__log.info("Calculating k...")
            # Do reference distributions for the gap statistic

            if not self.max_k:
                self.max_k = int(np.sqrt(self.data.shape[0]))

            if self.k_neig is None:

                cluster_range = np.arange(self.min_k, self.max_k, step=np.max(
                    [int((self.max_k - self.min_k) / self.n_points), 1]))

                inertias = []
                disps = []
                bg_distances = sign.background_distances(self.metric)

                pvals = bg_distances["pvalue"]
                distance = bg_distances["distance"]

                sig_dist = distance[
                    bisect.bisect_left(pvals, self.significance)]

                for k in cluster_range:
                    niter = 20
                    d = self.data.shape[1]
                    kmeans = faiss.Kmeans(int(d), int(k), niter=niter)
                    kmeans.train(self.data)
                    D, labels = kmeans.index.search(self.data, 1)
                    inertias += [self._inertia(self.data,
                                               labels, kmeans.centroids)]
                    disps += [self._dispersion(kmeans.centroids,
                                               sig_dist, self.metric)]

                disps[0] = disps[1]

                # Smooting, monotonizing, and combining the scores

                Ncs = np.arange(self.min_k, self.max_k)
                D = self._minmaxscaler(np.interp(Ncs, cluster_range, self._smooth(
                    self._monotonize(np.array(disps), True), self.max_k)))
                I = self._minmaxscaler(np.interp(Ncs, cluster_range, self._smooth(
                    self._monotonize(np.array(inertias), False), self.max_k)))

                alpha = Vm / (Vm + np.sqrt(Vn / 2.))

                S = np.abs((I**(1 - alpha)) - (D**(alpha)))
                S = self._minmaxscaler(-self._smooth(S, self.max_k))

                k = plot.clustering_plot(Ncs, I, D, S)
            else:
                k = self.k_neig

            self.__log.info("Clustering with k = %d" % k)

            niter = 20
            d = self.data.shape[1]
            kmeans = faiss.Kmeans(int(d), int(k), niter=niter)
            kmeans.train(self.data)
            D, labels = kmeans.index.search(self.data, 1)

            centroids = kmeans.centroids

            with h5py.File(os.path.join(self.model_path, self.clustcentroids_file), "w") as hf:
                hf.create_dataset("centroids", data=centroids)

            self.__log.info("Balancing...")
            labels = self._get_balance(self.data, centroids,
                                       labels, self.balance, k, tmp_dir)

            self.__log.info("Saving matrix...")

            with h5py.File(self.data_path, "w") as hf:
                hf.create_dataset("labels", data=labels)
                hf.create_dataset("keys", data=np.array(
                    self.keys, DataSignature.string_dtype()))
                hf.create_dataset("V", data=self.data)

            if validations:
                # MOA validation

                self.__log.info("Doing validations")

                if mappings is not None:
                    inchikey_mappings = dict(mappings)
                else:
                    inchikey_mappings = None

                inchikey_clust = shelve.open(
                    os.path.join(tmp_dir, "clus.dict"), "n")
                for i in range(len(self.keys)):
                    inchikey_clust[str(self.keys[i])] = labels[i]
                odds_moa, pval_moa = plot.label_validation(
                    inchikey_clust, "clus", prefix="moa", inchikey_mappings=inchikey_mappings)
                odds_atc, pval_atc = plot.label_validation(
                    inchikey_clust, "clus", prefix="atc", inchikey_mappings=inchikey_mappings)
                inchikey_clust.close()

            self.__log.info("Cleaning")
            for filename in glob.glob(os.path.join(tmp_dir, "clus.dict*")):
                os.remove(filename)

            os.rmdir(tmp_dir)

            faiss.write_index(kmeans.index, os.path.join(
                self.model_path, "kmeans.index"))

            if self.k_neig is None and validations:
                self.__log.info("Saving info")
                INFO = {
                    "k": int(k),
                    "odds_moa": odds_moa,
                    "pval_moa": pval_moa,
                    "odds_atc": odds_atc,
                    "pval_atc": pval_atc
                }
                with open(os.path.join(self.model_path, self.clust_info_file), 'w') as fp:
                    json.dump(INFO, fp)

        with h5py.File(self.data_path, "a") as hf:
            name = str(self.dataset) + "_clus"
            hf.create_dataset(
                "name", data=[name.encode(encoding='UTF-8', errors='strict')])
            hf.create_dataset(
                "date", data=[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode(encoding='UTF-8', errors='strict')])
            hf.create_dataset("metric", data=[self.metric.encode(
                encoding='UTF-8', errors='strict')])
            hf.create_dataset("normed", data=[False])
            hf.create_dataset("integerized", data=[False])
            hf.create_dataset("principal_components", data=[False])
            if mappings is not None:
                hf.create_dataset("mappings", data=np.array(
                    mappings, DataSignature.string_dtype()))
        # also predict for full if available
        sign_full = self.get_sign('sign' + self.cctype[-1]).get_molset("full")
        if os.path.isfile(sign_full.data_path):
            self.predict(sign_full, self.get_molset("full").data_path)
        self.mark_ready()

    def predict(self, sign, destination=None, validations=False):
        """Use the fitted models to go from input to output."""
        try:
            import faiss
        except ImportError:
            raise ImportError("requires faiss " +
                              "https://github.com/facebookresearch/faiss")
        try:
            import hdbscan
        except ImportError:
            raise ImportError("requires hdbscan " +
                              "https://hdbscan.readthedocs.io/en/latest/")

        plot = Plot(self.dataset, self.stats_path)

        mappings = None

        if os.path.isfile(sign.data_path):
            self.data = sign.data.astype('float32')
            self.data_type = self.data.dtype
            self.keys = sign.keys
            mappings = sign.mappings
        else:
            raise Exception("The file " + sign.data_path + " does not exist")

        if destination is None:
            raise Exception(
                "Predict method requires a destination file to output results")

        tmp_dir = tempfile.mkdtemp(
            prefix='sign_' + self.dataset + "_", dir=Config().PATH.CC_TMP)

        self.__log.debug("Temporary files saved in " + tmp_dir)

        if self.type == "hdbscan":
            self.__log.info("Reading HDBSCAN clusters")

            clusterer = joblib.load(os.path.join(
                self.model_path, self.hdbscan_file))

            self.__log.info("Predicting...")

            labels, strengths = hdbscan.approximate_predict(
                clusterer, self.data)

            # Save

            self.__log.info("Saving matrix...")

            with h5py.File(destination, "w") as hf:
                hf.create_dataset("labels", data=labels)
                hf.create_dataset("keys", data=np.array(
                    self.keys, DataSignature.string_dtype()))
                hf.create_dataset("strengths", data=strengths)

            if validations:

                self.__log.info("Doing validations")

                if mappings is not None:
                    inchikey_mappings = dict(mappings)
                else:
                    inchikey_mappings = None

                inchikey_clust = shelve.open(
                    os.path.join(tmp_dir, "clus.dict"), "n")
                for i in range(len(self.keys)):
                    lab = labels[i]
                    if lab == -1:
                        continue
                    inchikey_clust[str(self.keys[i])] = lab
                odds_moa, pval_moa = plot.label_validation(
                    inchikey_clust, "clus", prefix="moa", inchikey_mappings=inchikey_mappings)
                odds_atc, pval_atc = plot.label_validation(
                    inchikey_clust, "clus", prefix="atc", inchikey_mappings=inchikey_mappings)
                inchikey_clust.close()

            self.__log.info("Cleaning")
            for filename in glob.glob(os.path.join(tmp_dir, "clus.dict*")):
                os.remove(filename)

            os.rmdir(tmp_dir)

            with h5py.File(destination, "a") as hf:
                name = str(self.dataset) + "_clus"
                hf.create_dataset(
                    "name", data=[name.encode(encoding='UTF-8', errors='strict')])
                hf.create_dataset(
                    "date", data=[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode(encoding='UTF-8', errors='strict')])
                hf.create_dataset("metric", data=["hdbscan"])
                hf.create_dataset("normed", data=[False])
                hf.create_dataset("integerized", data=[False])
                hf.create_dataset("principal_components", data=[False])
                if mappings is not None:
                    hf.create_dataset("mappings", data=np.array(
                        mappings, DataSignature.string_dtype()))

        if self.type == "kmeans":

            faiss.omp_set_num_threads(self.cpu)

            if not os.path.isfile(os.path.join(self.model_path, "kmeans.index")):
                raise Exception(
                    "There is not cluster info. Please run fit method.")

            if self.metric == "cosine":
                self.data = self._normalizer(self.data, True)

            if self.data.shape[1] < self.num_subdim:
                self.data = np.hstack(
                    (self.data,
                        np.zeros((self.data.shape[0],
                                  self.num_subdim - self.data.shape[1]))))
                self.data = self.data.astype('float32')

            index = faiss.read_index(os.path.join(
                self.model_path, "kmeans.index"))

            D, labels = index.search(self.data, 1)

            self.__log.info("Saving matrix...")

            with h5py.File(destination, "w") as hf:
                hf.create_dataset("labels", data=labels)
                hf.create_dataset("keys", data=np.array(
                    self.keys, DataSignature.string_dtype()))
                hf.create_dataset("V", data=self.data)

            if validations:
                # MOA validation

                self.__log.info("Doing validations")

                if mappings is not None:
                    inchikey_mappings = dict(mappings)
                else:
                    inchikey_mappings = None

                inchikey_clust = shelve.open(
                    os.path.join(tmp_dir, "clus.dict"), "n")
                for i in range(len(self.keys)):
                    inchikey_clust[str(self.keys[i])] = labels[i]
                odds_moa, pval_moa = plot.label_validation(
                    inchikey_clust, "clus", prefix="moa", inchikey_mappings=inchikey_mappings)
                odds_atc, pval_atc = plot.label_validation(
                    inchikey_clust, "clus", prefix="atc", inchikey_mappings=inchikey_mappings)
                inchikey_clust.close()

            self.__log.info("Cleaning")
            for filename in glob.glob(os.path.join(tmp_dir, "clus.dict*")):
                os.remove(filename)

            os.rmdir(tmp_dir)

            with h5py.File(destination, "a") as hf:
                name = str(self.dataset) + "_clus"
                hf.create_dataset(
                    "name", data=[name.encode(encoding='UTF-8', errors='strict')])
                hf.create_dataset(
                    "date", data=[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode(encoding='UTF-8', errors='strict')])
                hf.create_dataset("metric", data=[self.metric.encode(
                    encoding='UTF-8', errors='strict')])
                hf.create_dataset("normed", data=[False])
                hf.create_dataset("integerized", data=[False])
                hf.create_dataset("principal_components", data=[False])
                if mappings is not None:
                    hf.create_dataset("mappings", data=np.array(
                        mappings, DataSignature.string_dtype()))

    def _smooth(self, x, max_k, window_len=None, window='hanning'):
        if window_len is None:
            window_len = int(max_k / 10) + 1
        if window_len % 2 == 0:
            window_len += 1
        if x.size <= window_len:
            self.__log.warning(
                "Input vector was smaller or equal than window size.")
            window_len = x.size - 1
            if window_len % 2 == 0:
                window_len += 1

        if window_len < 3:
            return x
        if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError(
                "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')
        n = int((window_len - 1) / 2)
        y = np.convolve(w / w.sum(), s, mode='valid')
        return y[n:-n]

    def _inertia(self, V_pqcode, labels, centroids):
        ines = 0
        for i in range(V_pqcode.shape[0]):
            ines += euclidean(V_pqcode[i], centroids[labels[i]][0] )
        return ines

    def _dispersion(self, centroids, sig_dist, metric):
        if len(centroids) == 1:
            return None
        return np.sum(pdist(centroids, metric=metric) < sig_dist)

    def _monotonize(self, v, up=True):
        if up:
            return np.mean(np.array([np.maximum.accumulate(v), np.minimum.accumulate(v[::-1])[::-1]]), axis=0)
        else:
            return np.mean(np.array([np.minimum.accumulate(v), np.maximum.accumulate(v[::-1])[::-1]]), axis=0)

    def _minmaxscaler(self, v):
        v = np.array(v)
        Min = np.min(v)
        Max = np.max(v)
        return (v - Min) / (Max - Min)

    def _get_balance(self, V_pqcode, centroids, labels, balance, k, tmp):

        if balance is None:
            return labels

        if balance < 1:
            self.__log.info(
                "Balance is smaller than 1. I don't understand. Anyway, I just don't balance.")
            return labels

        S = np.ceil((V_pqcode.shape[0] / k) * balance)

        clusts = [None] * V_pqcode.shape[0]
        counts = [0] * k

        tmpfile = os.path.join(tmp, "clus_dists.csv")

        with open(tmpfile, "w") as f:

            for i, v in enumerate(V_pqcode):
                for j, c in enumerate(centroids):
                    d = euclidean(c, v)
                    f.write("%d,%d,%010d\n" % (i, j, d))

        csvsort(tmpfile, [2], has_header=False)

        with open(tmpfile, "r") as f:
            for r in csv.reader(f):
                item_id = int(r[0])
                cluster_id = int(r[1])
                if counts[cluster_id] >= S:
                    continue
                if clusts[item_id] is None:
                    clusts[item_id] = cluster_id
                    counts[cluster_id] += 1

        os.remove(tmpfile)

        return clusts

    def _normalizer(self, V, recycle):

        FILE = self.model_path + "/normalizer.pkl"

        if not recycle or not os.path.exists(FILE):
            nlz = Normalizer(copy=True, norm="l2")
            V = nlz.fit_transform(V)
            joblib.dump(nlz, FILE)
        else:
            nlz = joblib.load(FILE)
            V = nlz.transform(V)

        return V.astype('float32')
