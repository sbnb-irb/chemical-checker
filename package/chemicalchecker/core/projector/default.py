import os
import h5py
import json
import random
import datetime
import numpy as np
from numpy import linalg as LA
from sklearn.manifold import MDS

from chemicalchecker.core.signature_base import BaseSignature
from chemicalchecker.core.signature_data import DataSignature

from chemicalchecker.util import logged
from chemicalchecker.util.plot import Plot


@logged
class Default(BaseSignature, DataSignature):
    """A Signature bla bla."""

    def __init__(self, signature_path, dataset, **params):
        """Initialize the projection class.

        Args:
            signature_path(str): the path to the signature directory.
            dataset(object): The dataset object with all info related
            metric(str): The metric used in the KNN algorithm: euclidean or cosine (default: cosine)
            type(int): The type of plot technology used between tsne and mds (default:tsne)
            cpu(int): The number of cores to use (default:1)
            perplexity(int): Perplexity for the NN-grapn (default:30)
            angle(float): Angle in BH-TSNE, from 0 to 0.5 (default:0.5)
        """
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(
            self, signature_path, dataset, **params)
        self.__log.debug('signature path is: %s', signature_path)

        proj_name = self.__class__.__name__
        self.data_path = os.path.join(signature_path, "proj_%s.h5" % proj_name)
        self.model_path = os.path.join(self.model_path, proj_name)
        if not os.path.isdir(self.model_path):
            original_umask = os.umask(0)
            os.makedirs(self.model_path, 0o775)
            os.umask(original_umask)
        self.stats_path = os.path.join(self.stats_path, proj_name)
        if not os.path.isdir(self.stats_path):
            original_umask = os.umask(0)
            os.makedirs(self.stats_path, 0o775)
            os.umask(original_umask)
        DataSignature.__init__(self, self.data_path)
        self.__log.debug('data_path: %s', self.data_path)

        self.index_file = "faiss_proj.index"
        self.projections_file = "centroids.h5"
        self.start_k = 1000

        self.type = "tsne"
        self.cpu = 1
        self.k_neig = None
        self.min_members = 5
        self.num_subdim = 8
        self.min_k = 1
        self.max_k = None
        self.n_points = 100
        self.balance = None
        self.significance = 0.05
        self.metric = "cosine"
        self.perplexity = 30
        self.angle = 0.5
        for param, value in params.items():
            self.__log.debug('parameter %s : %s', param, value)
            if "type" in params:
                self.type = params["type"]
            if "cpu" in params:
                self.cpu = params["cpu"]
            if "metric" in params:
                self.metric = params["metric"]
            if "perplexity" in params:
                self.perplexity = params["perplexity"]
            if "angle" in params:
                self.angle = params["angle"]

    def fit(self, signature, validations=True):
        """Take an input and learns to produce an output."""
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
        try:
            from MulticoreTSNE import MulticoreTSNE as TSNE
        except ImportError:
            raise ImportError("requires MulticoreTSNE " +
                              "http://github.com/DmitryUlyanov/Multicore-TSNE")
        BaseSignature.fit(self)

        plot = Plot(self.dataset, self.stats_path)

        mappings = None

        faiss.omp_set_num_threads(self.cpu)

        if os.path.isfile(signature.data_path):
            self.data = signature.data.astype('float32')
            self.data_type = signature.data_type
            self.keys = signature.keys
            mappings = signature.mappings
        else:
            raise Exception(
                "The file " + signature.data_path + " does not exist")

        self.__log.info("Applying kmeans")

        size = self.data.shape[0]

        self.k = int(max(self.start_k, min(15000, size / 2)))

        if self.k > size:
            self.k = size

        niter = 20
        d = self.data.shape[1]

        kmeans = faiss.Kmeans(d, self.k, niter=niter)

        if self.k != self.data.shape[0]:
            kmeans.train(self.data)
        else:
            kmeans.k = self.k

        if kmeans.k != self.data.shape[0]:

            self.__log.info(
                "Applying first projection and another clustering to get final points")

            if self.type == "tsne":

                tsne = TSNE(n_jobs=self.cpu, perplexity=self.perplexity, angle=self.angle,
                            n_iter=1000, metric=self.metric)
                Proj = tsne.fit_transform(kmeans.centroids.astype('float64'))

            if self.type == "mds":

                mds = MDS(n_jobs=self.cpu)
                Proj = mds.fit_transform(kmeans.centroids.astype('np.float64'))

            X = Proj[:, 0]
            Y = Proj[:, 1]

            min_size = 0
            if len(X) <= self.start_k:
                min_size = 5

            if len(X) > self.start_k:
                min_size = int(
                    np.interp(len(X), [self.start_k, 10000], [5, 15]))

            if len(X) >= 10000:
                min_size = 15

            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)

            combined = np.vstack((X, Y)).T

            clusterer.fit(combined)

            mask_clust = clusterer.labels_ != -1

            final_indexes = np.where(mask_clust)
            final_Proj = Proj[mask_clust]

            if self.metric == "euclidean":
                index = faiss.IndexFlatL2(
                    kmeans.centroids[final_indexes].shape[1])
            else:
                index = faiss.IndexFlatIP(
                    kmeans.centroids[final_indexes].shape[1])

            if self.metric == "cosine":

                norms = LA.norm(kmeans.centroids[final_indexes], axis=1)

                index.add( np.array( kmeans.centroids[final_indexes] / norms[:, None], dtype='float32') )
            else:
                index.add( np.array(kmeans.centroids[final_indexes], dtype='float32') )

            points_proj = {}

            points_proj["weights"], points_proj["ids"] = self._get_weights(
                index, self.data, 3)

            xlim, ylim = plot.projection_plot(final_Proj, bw=0.1, levels=10)

            projections = self._project_points(final_Proj, points_proj)

        else:

            if self.perplexity is None:
                neigh = np.max(
                    [30, np.min([150, int(np.sqrt(self.data.shape[0]) / 2)])])
                self.perplexity = int(neigh / 3)

            if self.type == "tsne":
                tsne = TSNE(n_jobs=self.cpu, perplexity=self.perplexity, angle=self.angle,
                            n_iter=1000, metric=self.metric)
                final_Proj = tsne.fit_transform(self.data.astype('float64'))

            if self.type == "mds":

                mds = MDS(n_jobs=self.cpu)
                final_Proj = mds.fit_transform(self.data.astype('float64'))

            if self.metric == "euclidean":
                index = faiss.IndexFlatL2(self.data.shape[1])
                index.add( np.array( self.data, dtype='float32') )
            else:
                index = faiss.IndexFlatIP(self.data.shape[1])

                norms = LA.norm(self.data, axis=1)

                index.add( np.array(self.data / norms[:, None], dtype='float32') )

            xlim, ylim = plot.projection_plot(final_Proj, bw=0.1, levels=10)

            projections = self._project_points(final_Proj)

        with h5py.File(self.data_path, "w") as hf:
            inchikey_proj = {}
            for i in range(len(self.keys)):
                k = self.keys[i]
                inchikey_proj[k] = projections[i]
            hf.create_dataset("V", data=projections)
            hf.create_dataset("keys", data=np.array(self.keys, DataSignature.string_dtype()))
            name = str(self.dataset) + "_proj"
            hf.create_dataset(
                "name", data=[name.encode(encoding='UTF-8', errors='strict')])
            hf.create_dataset(
                "date", data=[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode(encoding='UTF-8', errors='strict')])
            hf.create_dataset("metric", data=[self.metric.encode(
                encoding='UTF-8', errors='strict')])
            if mappings is not None:
                hf.create_dataset("mappings", data=np.array(mappings, DataSignature.string_dtype()))

        faiss.write_index(index, os.path.join(
            self.model_path, self.index_file))

        with h5py.File(os.path.join(
                self.model_path, self.projections_file), "w") as hf:
            hf.create_dataset("Proj", data=final_Proj)

        if validations:
            self.__log.info("Doing MoA & ATC validations")

            if mappings is not None:
                inchikey_mappings = dict(mappings)
            else:
                inchikey_mappings = None

            ks_moa, auc_moa, frac_moa = plot.vector_validation(
                self, "proj", prefix="moa", mappings=inchikey_mappings)
            ks_atc, auc_atc, frac_atc = plot.vector_validation(
                self, "proj", prefix="atc", mappings=inchikey_mappings)

            # Saving results

            INFO = {
                "molecules": final_Proj.shape[0],
                "moa_ks_d": ks_moa[0],
                "moa_ks_p": ks_moa[1],
                "moa_auc": auc_moa,
                "atc_ks_d": ks_atc[0],
                "atc_ks_p": ks_atc[1],
                "atc_auc": auc_atc,
                "xlim": xlim,
                "ylim": ylim
            }

            with open(os.path.join(self.stats_path, 'proj_stats.json'), 'w') as fp:
                json.dump(INFO, fp)

        self.mark_ready()

    def predict(self, signature, destination):
        """Use the fitted models to go from input to output."""
        try:
            import faiss
        except ImportError:
            raise ImportError("requires faiss " +
                              "https://github.com/facebookresearch/faiss")
        BaseSignature.predict(self)
        mappings = None

        faiss.omp_set_num_threads(self.cpu)

        input_data_file = ''

        if isinstance(signature, str):
            input_data_file = signature
        else:
            input_data_file = signature.data_path

        if os.path.isfile(input_data_file):
            dh5 = h5py.File(input_data_file, 'r')
            if "keys" not in dh5.keys() or "V" not in dh5.keys():
                raise Exception(
                    "H5 file %s does not contain datasets 'keys' and 'V'"
                    % signature.data_path)
            self.data = np.array(dh5["V"][:], dtype='float32')
            self.data_type = dh5["V"].dtype
            self.keys = dh5["keys"][:]
            if "mappings" in dh5.keys():
                mappings = dh5["mappings"][:]
            dh5.close()

        else:
            raise Exception("The file " + input_data_file + " does not exist")

        if destination is None:
            raise Exception(
                "Predict method requires a destination file to output results")

        if not os.path.isfile(os.path.join(self.model_path, self.index_file)) or not os.path.isfile(os.path.join(self.model_path, self.projections_file)):
            raise Exception(
                "This projection does not have prediction information. Please, call fit method first to use the predict method.")

        index = faiss.read_index(os.path.join(
            self.model_path, self.index_file))

        dh5 = h5py.File(os.path.join(
            self.model_path, self.projections_file), 'r')
        Proj = dh5["Proj"][:]
        dh5.close()

        # base_points = faiss.vector_float_to_array(index.xb).reshape(-1, index.d)

        points_proj = {}

        points_proj["weights"], points_proj["ids"] = self._get_weights(
            index, self.data, 3)

        projections = self._project_points(Proj, points_proj)

        with h5py.File(destination, "w") as hf:
            inchikey_proj = {}
            for i in range(len(self.keys)):
                k = self.keys[i]
                inchikey_proj[k] = projections[i]
            hf.create_dataset("V", data=projections)
            hf.create_dataset("keys", data=np.array(self.keys, DataSignature.string_dtype()))
            name = str(self.dataset) + "_proj"
            hf.create_dataset(
                "name", data=[name.encode(encoding='UTF-8', errors='strict')])
            hf.create_dataset(
                "date", data=[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode(encoding='UTF-8', errors='strict')])
            hf.create_dataset("metric", data=[self.metric.encode(
                encoding='UTF-8', errors='strict')])
            if mappings is not None:
                hf.create_dataset("mappings", data=np.array(mappings, DataSignature.string_dtype()))

    def _get_weights(self, index, data, neigh):
        """Get weights for the first 'neigh' neighbours ."""

        if self.metric == "cosine":

            norms = LA.norm(data, axis=1)

            D, I = index.search( np.array(data / norms[:, None], dtype='float32'), neigh)

            # Convert to [0,1]
            D = np.maximum(0.0, (1.0 + D) / 2.0)

        else:

            D, I = index.search( np.array( data[np.array(random.sample(
                range(0, data.shape[0]), 10))], dtype='float32'), index.ntotal)
            max_val = np.max(D)

            D, I = index.search( np.array(data, dtype='float32'), neigh)

            D[:, :] = np.maximum((max_val - D[:, :]) / max_val, 0)

        return D, I

    def _project_points(self, Proj, extra_data=None):

        X = Proj[:, 0]
        Y = Proj[:, 1]

        if extra_data is not None:

            weights = extra_data["weights"]
            ids = extra_data["ids"]

            size = weights.shape[0]
            proj_data = np.zeros((size, 2))
            for i in range(0, size):

                if any(np.isclose(weights[i], 1.0)):
                    pos = ids[i][0]
                    if i in ids[i] or len(weights) == len(X):
                        pos = i
                    proj_data[i][0] = X[pos]
                    proj_data[i][1] = Y[pos]
                else:
                    proj_data[i][0] = np.average(
                        X[np.array(ids[i])], weights=weights[i])
                    proj_data[i][1] = np.average(
                        Y[np.array(ids[i])], weights=weights[i])
        else:
            size = X.shape[0]
            proj_data = np.zeros((size, 2))
            for i in range(0, size):
                proj_data[i][0] = X[i]
                proj_data[i][1] = Y[i]

        return proj_data
