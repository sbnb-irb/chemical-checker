"""Signature type 2.

Network embedding of the similarity matrix derived from signatures type 1.
They have fixed length, which is convenient for machine learning, and capture
both explicit and implicit similarity relationships in the data.

Signatures type 2 are the result of a two-step process:

1. Transform nearest-neighbor of signature type 1 to graph.

2. Perform network embedding with Node2Vec.

It is not possible to produce network embeddings for out-of-sample
(out-of-vocabulary) nodes, so a multi-output regression needs to be performed
a posteriori (from signatures type 1 to signatures type 2) in order to endow
predict() capabilities. This is done using the AdaNet framework for for
automatically learning high-quality models with minimal expert intervention.
"""
import os
import h5py
import shutil
import numpy as np
from time import time
from scipy.stats.mstats import rankdata
from numpy import linalg as LA
from tqdm import tqdm

from .signature_base import BaseSignature
from .signature_data import DataSignature

from chemicalchecker.util.plot import Plot
from chemicalchecker.util import logged
from chemicalchecker.util import Config
from chemicalchecker.util.splitter import Traintest


@logged
class sign2(BaseSignature, DataSignature):
    """Signature type 2 class."""

    def __init__(self, signature_path, dataset, **params):
        """Initialize the signature.

        Args:
            signature_path(str): The signature root directory.
            dataset(`Dataset`): `chemicalchecker.database.Dataset` object.
            params(): Parameters, expected keys are 'graph', 'node2vec', and
                'adanet'.
        """
        # Calling init on the base class to trigger file existence checks
        BaseSignature.__init__(self, signature_path,
                               dataset, **params)
        # generate needed paths
        self.data_path = os.path.join(self.signature_path, 'sign2.h5')
        DataSignature.__init__(self, self.data_path)
        self.model_path = os.path.join(self.signature_path, 'models')
        self.faiss_path = os.path.join(self.model_path, 'faiss')
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        self.stats_path = os.path.join(self.signature_path, 'stats')
        if not os.path.isdir(self.model_path):
            os.makedirs(self.stats_path)
        # assign dataset
        self.dataset = dataset
        # logging
        self.__log.debug('dataset: %s', dataset)
        self.__log.debug('data_path: %s', self.data_path)
        self.__log.debug('model_path: %s', self.model_path)
        self.__log.debug('stats_path: %s', self.stats_path)
        # get parameters or default values
        self.params = dict()
        self.params['graph'] = params.get('graph', None)
        self.params['node2vec'] = params.get('node2vec', None)
        self.params['adanet'] = params.get('adanet', None)
        self.clustering = 1000
        if self.params['adanet'] is not None:
            self.cpu = self.params['adanet'].get('cpu', 1)

    def fit(self, sign1, neig1, reuse=True, validations=True, compare_nn=False):
        """Learn a model.

        Node2vec embeddings are computed using the graph derived from sign1.
        The predictive model is learned with AdaNet.

        Args:
            sign1(sign1): Signature type 1.
            neig1(neig1): Nearest neighbor of type 1.
            reuse(bool): Reuse already generated intermediate files. Set to
                False to re-train from scratch.
        """
        #########
        # step 1: Node2Vec (learn graph embedding) input is neig1
        #########
        try:
            from chemicalchecker.util.network import SNAPNetwork
            from chemicalchecker.util.performance import LinkPrediction
            from chemicalchecker.tool.adanet import AdaNet
            from chemicalchecker.tool.node2vec import Node2Vec
        except ImportError as err:
            raise err

        self.__log.debug('Node2Vec on %s' % sign1)
        n2v = Node2Vec(executable=Config().TOOLS.node2vec_exec)
        # use neig1 to generate the Node2Vec input graph (as edgelist)
        graph_params = self.params['graph']
        node2vec_path = os.path.join(self.model_path, 'node2vec')
        if not os.path.isdir(node2vec_path):
            os.makedirs(node2vec_path)
        graph_file = os.path.join(node2vec_path, 'graph.edgelist')
        if not reuse or not os.path.isfile(graph_file):
            if graph_params:
                n2v.to_edgelist(sign1, neig1, graph_file, **graph_params)
            else:
                n2v.to_edgelist(sign1, neig1, graph_file)
        # check that all molecules are considered in the graph
        with open(graph_file, 'r') as fh:
            lines = fh.readlines()
        graph_mol = set(l.split()[0] for l in lines)
        # we can just compare the total nr
        if not len(graph_mol) == len(sign1.unique_keys):
            raise Exception("Graph %s is missing nodes." % graph_file)
        # save graph stats
        graph_stat_file = os.path.join(self.stats_path, 'graph_stats.json')
        graph = None
        if not reuse or not os.path.isfile(graph_stat_file):
            graph = SNAPNetwork.from_file(graph_file)
            graph.stats_toJSON(graph_stat_file)
        # run Node2Vec to generate embeddings
        node2vec_params = self.params['node2vec']
        emb_file = os.path.join(node2vec_path, 'n2v.emb')
        if not reuse or not os.path.isfile(emb_file):
            if node2vec_params:
                n2v.run(graph_file, emb_file, **node2vec_params)
            else:
                n2v.run(graph_file, emb_file)
        # convert to signature h5 format
        if not reuse or not os.path.isfile(self.data_path):
            n2v.emb_to_h5(sign1.keys, emb_file, self.data_path)
        # save link prediction stats
        linkpred_file = os.path.join(self.stats_path, 'linkpred.json')
        if not reuse or not os.path.isfile(linkpred_file):
            if not graph:
                graph = SNAPNetwork.from_file(graph_file)
            linkpred = LinkPrediction(self, graph)
            linkpred.performance.toJSON(linkpred_file)
        # copy reduced-full mappingsfrom sign1
        if "mappings" not in self.info_h5 and "mappings" in sign1.info_h5:
            self.copy_from(sign1, "mappings")
        else:
            self.__log.warn("Cannot copy 'mappings' from sign1.")
        sign2_plot = Plot(self.dataset, self.stats_path)
        sign2_plot.sign_feature_distribution_plot(self)
        #########
        # step 2: AdaNet (learn to predict sign2 from sign1 without Node2Vec)
        #########
        self.__log.debug('AdaNet fit %s with Node2Vec output' % sign1)
        # get params and set folder
        adanet_params = self.params['adanet']
        adanet_path = os.path.join(self.model_path, 'adanet')
        if adanet_params:
            if 'model_dir' in adanet_params:
                adanet_path = adanet_params.pop('model_dir')
        if not reuse or not os.path.isdir(adanet_path):
            os.makedirs(adanet_path)
        # prepare train-test file
        traintest_file = os.path.join(adanet_path, 'traintest.h5')
        if adanet_params:
            traintest_file = adanet_params.pop(
                'traintest_file', traintest_file)
        if not reuse or not os.path.isfile(traintest_file):
            Traintest.create_signature_file(
                sign1.data_path, self.data_path, traintest_file)

        if adanet_params:
            ada = AdaNet(model_dir=adanet_path,
                         traintest_file=traintest_file, **adanet_params)
        else:
            ada = AdaNet(model_dir=adanet_path, traintest_file=traintest_file)
        # learn NN with AdaNet
        self.__log.debug('AdaNet training on %s' % traintest_file)
        ada.train_and_evaluate()
        # save AdaNet performances and plots
        sign2_plot = Plot(self.dataset, adanet_path)
        extra_preditors = dict()
        if compare_nn:
            nearest_neighbor_pred = sign2.predict_nearest_neighbor(
                self.model_path, traintest_file)
            extra_preditors['NearestNeighbor'] = nearest_neighbor_pred
        ada.save_performances(adanet_path, sign2_plot, extra_preditors)
        self.__log.debug('model saved to %s' % adanet_path)
        # save background distances, validate and mark ready
        self.background_distances("cosine")
        if validations:
            self.validate()
        self.mark_ready()

    def fit_merge(self, signs, reuse=True, validations=True, compare_nn=False):
        """Learn a model.

        Node2vec embeddings are computed using the graph derived from several signatures.
        The predictive model is learned with AdaNet.

        Args:
            sign1(sign1): Signature type 1.
            neig1(neig1): Nearest neighbor of type 1.
            reuse(bool): Reuse already generated intermediate files. Set to
                False to re-train from scratch.
        """
        #########
        # step 1: Node2Vec (learn graph embedding) input is neig1
        #########
        try:
            from chemicalchecker.util.network import SNAPNetwork
            from chemicalchecker.util.performance import LinkPrediction
            from chemicalchecker.tool.adanet import AdaNet
            from chemicalchecker.tool.node2vec import Node2Vec
            import faiss
        except ImportError as err:
            raise err

        faiss.omp_set_num_threads(self.cpu)

        self.__log.debug('Node2Vec on several signatures')
        n2v = Node2Vec(executable=Config().TOOLS.node2vec_exec)
        # use neig1 to generate the Node2Vec input graph (as edgelist)
        graph_params = self.params['graph']
        node2vec_path = os.path.join(self.model_path, 'node2vec')
        if not os.path.isdir(node2vec_path):
            os.makedirs(node2vec_path)
        graph_file = os.path.join(node2vec_path, 'graph.edgelist')
        if not reuse or not os.path.isfile(graph_file):
            graph_files_in = []
            for sign in signs:
                sign2_model_path = sign.model_path.replace("sign1", "sign2")
                graph_files_in.append(os.path.join(
                    sign2_model_path, 'node2vec', 'graph.edgelist'))

            if graph_params:
                n2v.merge_edgelists(signs, graph_files_in,
                                    graph_file, **graph_params)
            else:
                n2v.merge_edgelists(
                    signs, graph_files_in, graph_file)

        with open(graph_file, 'r') as fh:
            lines = fh.readlines()
        keys = set(l.split()[0] for l in lines)

        unique_keys = set()
        for sign in signs:
            unique_keys.update(sign.unique_keys)

        if not len(keys) == len(unique_keys):
            raise Exception("Graph %s is missing nodes. %d/%d" %
                            graph_file, len(keys), len(unique_keys))
        else:
            list_keys = list(unique_keys)
            list_keys.sort()

        # save graph stats
        graph_stat_file = os.path.join(self.stats_path, 'graph_stats.json')
        graph = None
        if not reuse or not os.path.isfile(graph_stat_file):
            graph = SNAPNetwork.from_file(graph_file)
            graph.stats_toJSON(graph_stat_file)
        # run Node2Vec to generate embeddings
        node2vec_params = self.params['node2vec']
        emb_file = os.path.join(node2vec_path, 'n2v.emb')
        if not reuse or not os.path.isfile(emb_file):
            if node2vec_params:
                n2v.run(graph_file, emb_file, **node2vec_params)
            else:
                n2v.run(graph_file, emb_file)
        # convert to signature h5 format
        if not reuse or not os.path.isfile(self.data_path):
            n2v.emb_to_h5(list_keys, emb_file, self.data_path)
        # save link prediction stats
        linkpred_file = os.path.join(self.stats_path, 'linkpred.json')
        if not reuse or not os.path.isfile(linkpred_file):
            if not graph:
                graph = SNAPNetwork.from_file(graph_file)
            linkpred = LinkPrediction(self, graph)
            linkpred.performance.toJSON(linkpred_file)

        mappings = {}
        for sign in signs:
            # copy reduced-full mappingsfrom sign1
            if "mappings" not in self.info_h5 and "mappings" in sign.info_h5:
                mappings.update(dict(sign.get_h5_dataset('mappings').tolist()))
            else:
                self.__log.warn(
                    "Cannot copy 'mappings' from sign: %s" % sign.data_path)
        if len(mappings) > 0:
            # Fixing merged mappings
            for key, value in mappings.items():
                if value not in keys:
                    mappings[key] = mappings[value]
            list_maps = sorted(mappings.items())
            with h5py.File(self.data_path, 'a') as hf:
                # delete if already there
                if 'mappings' in hf:
                    del hf['mappings']
                hf.create_dataset("mappings",
                                  data=np.array(list_maps,
                                                DataSignature.string_dtype()))
        sign2_plot = Plot(self.dataset, self.stats_path)
        sign2_plot.sign_feature_distribution_plot(self)

        if "centroids" not in self.info_h5:

            # Find 1000 molecules that are representative
            # Centroids of a k=1000 clustering should work
            niter = 20
            d = self.shape[1]
            kmeans = faiss.Kmeans(int(d), int(self.clustering), niter=niter)
            with h5py.File(self.data_path, 'r') as dh5:
                data = np.float32(np.array(dh5["V"][:], dtype=np.float32))
                kmeans.train(data)

            centroids = kmeans.centroids

            # Let's find the closest molecule to each one of the centroids
            # We look for the nearest neighbous of the centroids

            index = faiss.IndexFlatL2(d)
            index.add(data)
            D, I = index.search(centroids, 3)

            # Sometimes the closest is already in the list so we take the next
            # one
            final_centroids = set()
            for i in I:
                for j in i:
                    if j not in final_centroids:
                        final_centroids.add(j)
                        break

            centroids = list(final_centroids)
            centroids.sort()
            with h5py.File(self.data_path, 'a') as hf:
                if 'centroids' in hf:
                    del hf['centroids']
                hf.create_dataset("centroids", data=np.array(centroids))
        else:
            centroids = self.get_h5_dataset("centroids")

        adanet_params = self.params['adanet']
        adanet_path = os.path.join(self.model_path, 'adanet')
        if adanet_params:
            if 'model_dir' in adanet_params:
                adanet_path = adanet_params.pop('model_dir')
        if not reuse or not os.path.isdir(adanet_path):
            os.makedirs(adanet_path)

        if not reuse or not os.path.isdir(self.faiss_path):
            os.makedirs(self.faiss_path)

        data_centroids = []
        for centroid in centroids:
            data_centroids.append(self[centroid])

        # For each signature 1 to merge
        for sign in signs:

            faiss_path_sign = os.path.join(self.faiss_path, sign.dataset)
            if not os.path.isdir(faiss_path_sign):
                os.makedirs(faiss_path_sign)

            final_index_file = os.path.join(
                faiss_path_sign, 'final.index')

            if os.path.isfile(final_index_file) and reuse:
                continue

            # create an index of sign2 merged for all molecules present in this
            # sign1
            dataset_keys = sign.keys

            index = faiss.IndexFlatIP(self.shape[1])

            for chunk in sign.chunker():

                _, data = self.get_vectors(dataset_keys[chunk])
                data_temp = np.array(data, dtype=np.float32)
                normst = LA.norm(data_temp, axis=1)
                index.add(data_temp / normst[:, None])

            # Find the closest molecule to each one of the 1K representative
            # molecules
            data_temp = np.array(data_centroids, dtype=np.float32)
            normst = LA.norm(data_temp, axis=1)
            Dt, It = index.search(data_temp / normst[:, None], 10)

            # Again if one of the closest was already taken, get the next one
            representative = set()
            for i in It:
                for j in i:
                    if j not in representative:
                        representative.add(j)
                        break

            # check actually there are 1K representative molecules
            if not len(representative) == self.clustering:
                raise Exception("Faiss do not have enough values.")

            index = faiss.IndexFlatIP(sign.shape[1])

            data = []
            for reprs in sorted(list(representative)):
                data.append(sign[reprs])

            # Index the signature1 of the 1K representative molecules and save
            # index to be used later
            data_temp = np.array(data, dtype=np.float32)
            normst = LA.norm(data_temp, axis=1)
            index.add(data_temp / normst[:, None])

            faiss.write_index(index, final_index_file)

        distances_file = os.path.join(self.faiss_path, 'distances.h5')

        with h5py.File(distances_file, "w") as results:
            # initialize V and keys datasets
            results.create_dataset(
                'V', (self.shape[0], self.clustering), dtype=np.float32)
            results.create_dataset('keys',
                                   data=np.array(self.keys,
                                                 DataSignature.string_dtype()))
            results.create_dataset("shape", data=(
                self.shape[0], self.clustering))

            # For all molecules in sign2 merged
            for chunk in tqdm(self.chunker()):

                distances = []
                keys = self.keys[chunk]
                for sign in signs:

                    faiss_path_sign = os.path.join(
                        self.faiss_path, sign.dataset)
                    final_index_file = os.path.join(
                        faiss_path_sign, 'final.index')

                    index = faiss.read_index(final_index_file)

                    # Find nearest neighbours to the 1K representatives of each
                    # subspace
                    _, data = sign.get_vectors(keys, include_nan=True)
                    data_temp = np.array(data, dtype=np.float32)
                    normst = LA.norm(data_temp, axis=1)
                    Dt, It = index.search(
                        data_temp / normst[:, None], self.clustering)
                    Dt = np.maximum(0.0, 1.0 - Dt)
                    # Order the distances acording to indices since we want the
                    # distance to representative0 to the first position, etc
                    distance = np.take_along_axis(
                        Dt, np.argsort(It, axis=1), axis=1)
                    # Rank the distances
                    ranked = rankdata(distance, axis=1)
                    # Put default value to vectors where there was no signature
                    ranked[It < 0] = float(self.clustering)
                    # Reverse ranks. Larger ranks for shorter distances
                    ranked = float(self.clustering) - ranked
                    distances += [ranked]

                # To aggregate the different distances, Let's take the maximum
                # of them
                final_distance = np.amax(distances, axis=0)
                results['V'][chunk] = final_distance / float(self.clustering)

        # prepare train-test file
        traintest_file = os.path.join(adanet_path, 'traintest.h5')
        if adanet_params:
            traintest_file = adanet_params.pop(
                'traintest_file', traintest_file)
        if not reuse or not os.path.isfile(traintest_file):
            Traintest.create_signature_file(
                distances_file, self.data_path, traintest_file)

        if adanet_params:
            ada = AdaNet(model_dir=adanet_path,
                         traintest_file=traintest_file, **adanet_params)
        else:
            ada = AdaNet(model_dir=adanet_path, traintest_file=traintest_file)
        # learn NN with AdaNet
        self.__log.debug('AdaNet training on %s' % traintest_file)
        ada.train_and_evaluate()

        # save background distances, validate and mark ready
        self.background_distances("cosine")
        if validations:
            self.validate()
        self.mark_ready()

    def predict_merge(self, signs, destination, validations=False):
        """Use the learned model to predict the signature."""
        try:
            from chemicalchecker.tool.adanet import AdaNet
            import faiss
        except ImportError as err:
            raise err
        # load AdaNet model
        adanet_path = os.path.join(self.model_path, 'adanet', 'savedmodel')
        self.__log.debug('loading model from %s' % adanet_path)
        predict_fn = AdaNet.predict_fn(adanet_path)

        join_keys = set()
        for sign in signs:
            join_keys.update(sign.keys)

        self.keys = list(join_keys)

        self.keys.sort()

        with h5py.File(destination, "w") as results:
            # initialize V and keys datasets
            results.create_dataset(
                'V', (len(self.keys), 128), dtype=np.float32)
            results.create_dataset('keys',
                                   data=np.array(self.keys,
                                                 DataSignature.string_dtype()))
            results.create_dataset("shape", data=(
                len(self.keys), 128))

            map_index = {}
            for sign in signs:

                faiss_path_sign = os.path.join(
                    self.faiss_path, sign.dataset)
                final_index_file = os.path.join(
                    faiss_path_sign, 'final.index')

                index = faiss.read_index(final_index_file)

                map_index[sign] = index

            size = 1000
            # For all molecules in sign2 merged
            for i in tqdm(range(0, len(self.keys), size)):
                chunk = slice(i, i + size)

                distances = []
                keys = self.keys[chunk]

                for sign in signs:

                    index = map_index[sign]

                    # Find nearest neighbours to the 1K representatives of each
                    # subspace
                    _, data = sign.get_vectors(keys, include_nan=True)
                    data_temp = np.array(data, dtype=np.float32)
                    normst = LA.norm(data_temp, axis=1)
                    Dt, It = index.search(
                        data_temp / normst[:, None], self.clustering)
                    Dt = np.maximum(0.0, 1.0 - Dt)
                    # Order the distances acording to indices since we want the
                    # distance to representative0 to the first position, etc
                    distance = np.take_along_axis(
                        Dt, np.argsort(It, axis=1), axis=1)
                    # Rank the distances
                    ranked = rankdata(distance, axis=1)
                    # Put default value to vectors where there was no signature
                    ranked[It < 0] = float(self.clustering)
                    # Reverse ranks. Larger ranks for shorter distances
                    ranked = float(self.clustering) - ranked
                    distances += [ranked]

                # To aggregate the different distances, Let's take the maximum
                # of them
                final_distance = np.amax(distances, axis=0)

                results['V'][chunk] = AdaNet.predict(
                    final_distance / float(self.clustering), predict_fn)

        if validations:
            self.validate()

    def predict(self, sign1, destination, validations=False):
        """Use the learned model to predict the signature."""
        try:
            from chemicalchecker.tool.adanet import AdaNet
        except ImportError as err:
            raise err
        # load AdaNet model
        adanet_path = os.path.join(self.model_path, 'adanet', 'savedmodel')
        self.__log.debug('loading model from %s' % adanet_path)
        predict_fn = AdaNet.predict_fn(adanet_path)
        tot_inks = len(sign1.keys)
        with h5py.File(destination, "w") as results:
            # initialize V and keys datasets
            results.create_dataset('V', (tot_inks, 128), dtype=np.float32)
            results.create_dataset('keys',
                                   data=np.array(sign1.keys,
                                                 DataSignature.string_dtype()))
            results.create_dataset("shape", data=(tot_inks, 128))
            # predict signature 2
            for chunk in sign1.chunker():
                results['V'][chunk] = AdaNet.predict(sign1[chunk], predict_fn)

        if validations:
            self.validate()

    @staticmethod
    def predict_nearest_neighbor(destination_path, traintest_file):
        """Prediction with nearest neighbor.

        Find nearest neighbor in sign 1 and mapping it to known sign 2.
        """
        from .data import DataFactory
        from .neig import neig
        sign2.__log.info('Performing Nearest Neighbor prediction.')
        # create directory to save neig and sign (delete if exists)
        nn_path = os.path.join(destination_path, "nearest_neighbor")
        if os.path.isdir(nn_path):
            shutil.rmtree(nn_path)
        # evaluate all data splits
        datasets = ['train', 'test', 'validation']
        nn_pred = dict()
        nn_pred_start = time()
        for ds in datasets:
            # get dataset split
            traintest = Traintest(traintest_file, ds)
            traintest.open()
            x_data = traintest.get_all_x()
            y_data = traintest.get_all_y()
            traintest.close()
            sign2.__log.info('Nearest Neighbor %s  X:%s  Y:%s.',
                             ds, x_data.shape, y_data.shape)
            # check that there are samples left
            if x_data.shape[0] == 0:
                sign2.__log.warning("No samples available, skipping.")
                return None
            # fit on train set
            if ds == "train":
                # signaturize dataset
                sign1_dest = os.path.join(nn_path, "sign1")
                os.makedirs(sign1_dest)
                nn_sign1 = DataFactory.signaturize(
                    "sign1", sign1_dest, x_data)
                # sign2 is needed just to get the default keys
                # as neig1.get_kth_nearest is returning keys of sign1
                sign2_dest = os.path.join(nn_path, "sign2")
                os.makedirs(sign2_dest)
                nn_sign2 = DataFactory.signaturize(
                    "sign2", sign2_dest, y_data)
                # create temporary neig1 and call fit
                neig1_dest = os.path.join(nn_path, "neig1")
                os.makedirs(neig1_dest)
                nn_neig1 = neig(neig1_dest, "NN.001")
                nn_neig1.fit(nn_sign1)
            # save nearest neighbor signatures as predictions
            nn_pred[ds] = dict()
            # get nearest neighbor indices and keys
            nn_neig1_idxs, _ = nn_neig1.get_kth_nearest(x_data, 1)
            nn_idxs = nn_neig1_idxs[:, 0]
            nn_pred[ds]['true'] = y_data
            nn_pred[ds]['pred'] = list()
            for idx in nn_idxs:
                nn_pred[ds]['pred'].append(nn_sign2[idx])
            nn_pred[ds]['pred'] = np.vstack(nn_pred[ds]['pred'])
        nn_pred_end = time()
        nn_pred['time'] = nn_pred_end - nn_pred_start
        nn_pred['name'] = "NearestNeighbor"
        return nn_pred

    @staticmethod
    def predict_adanet(destination_path, traintest_file, params):
        """Prediction with adanet."""
        from chemicalchecker.tool.adanet import AdaNet
        sign2.__log.info('Performing AdaNet prediction.')
        # create directory to save neig and sign (delete if exists)
        ada_path = os.path.join(destination_path, "adanet")
        if os.path.isdir(ada_path):
            shutil.rmtree(ada_path)
        # evaluate all data splits
        datasets = ['train', 'test', 'validation']
        ada_pred = dict()
        ada_pred_start = time()
        ada = AdaNet(model_dir=ada_path, traintest_file=traintest_file,
                     **params)
        ada.train_and_evaluate()
        ada_pred_end = time()
        for ds in datasets:
            # get dataset split
            traintest = Traintest(traintest_file, ds)
            traintest.open()
            x_data = traintest.get_all_x()
            y_data = traintest.get_all_y()
            traintest.close()
            sign2.__log.info('AdaNet %s  X:%s  Y:%s.',
                             ds, x_data.shape, y_data.shape)
            # check that there are samples left
            if x_data.shape[0] == 0:
                sign2.__log.warning("No samples available, skipping.")
                return None
            ada_pred[ds] = dict()
            # get nearest neighbor indices and keys
            ada_pred[ds]['true'] = y_data
            ada_pred[ds]['pred'] = AdaNet.predict(ada.save_dir, x_data)
        ada_pred['time'] = ada_pred_end - ada_pred_start
        ada_pred['name'] = "AdaNet"
        return ada_pred

    @staticmethod
    def predict_linear_regression(destination_path, traintest_file):
        """Prediction with adanet."""
        from sklearn.linear_model import LinearRegression
        sign2.__log.info('Performing LinearRegression prediction.')
        # create directory to save neig and sign (delete if exists)
        lr_path = os.path.join(destination_path, "linear")
        if os.path.isdir(lr_path):
            shutil.rmtree(lr_path)
        # evaluate all data splits
        datasets = ['train', 'test', 'validation']
        lr_pred = dict()
        for ds in datasets:
            # get dataset split
            traintest = Traintest(traintest_file, ds)
            traintest.open()
            x_data = traintest.get_all_x()
            y_data = traintest.get_all_y()
            traintest.close()
            sign2.__log.info('LinearRegression %s  X:%s  Y:%s.',
                             ds, x_data.shape, y_data.shape)
            # check that there are samples left
            if x_data.shape[0] == 0:
                sign2.__log.warning("No samples available, skipping.")
                return None
            lr_pred[ds] = dict()
            if ds == 'train':
                lr_pred_start = time()
                linreg = LinearRegression().fit(x_data, y_data)
                lr_pred_end = time()
            # get nearest neighbor indices and keys
            lr_pred[ds]['true'] = y_data
            lr_pred[ds]['pred'] = linreg.predict(x_data)
        lr_pred['time'] = lr_pred_end - lr_pred_start
        lr_pred['name'] = "LinearRegression"
        return lr_pred

    def eval_node2vec(self, sign1, neig1, reuse=True):
        """Evaluate node2vec performances.

        Node2vec embeddings are computed using the graph derived from sign1.
        We split edges in a train and test set so we can compute the ROC
        of link prediction.

        Args:
            sign1(sign1): Signature type 1.
            neig1(neig1): Nearest neighbor of type 1.
            reuse(bool): Reuse already generated intermediate files. Set to
                False to re-train from scratch.
        """
        #########
        # step 1: Node2Vec (learn graph embedding) input is neig1
        #########
        try:
            from chemicalchecker.util.network import SNAPNetwork
            from chemicalchecker.util.performance import LinkPrediction
            from chemicalchecker.tool.node2vec import Node2Vec
        except ImportError as err:
            raise err

        self.__log.debug('Node2Vec on %s' % sign1)
        n2v = Node2Vec(executable=Config().TOOLS.node2vec_exec)
        # define the n2v model path
        node2vec_params = self.params['node2vec']
        node2vec_path = os.path.join(self.model_path, 'node2vec_eval')
        if node2vec_params:
            if 'model_dir' in node2vec_params:
                node2vec_path = node2vec_params.pop('model_dir')
        if not reuse or not os.path.isdir(node2vec_path):
            os.makedirs(node2vec_path)
        # use neig1 to generate the Node2Vec input graph (as edgelist)
        graph_params = self.params['graph']
        graph_file = os.path.join(
            self.model_path, 'node2vec', 'graph.edgelist')
        if not reuse or not os.path.isfile(graph_file):
            if graph_params:
                n2v.to_edgelist(sign1, neig1, graph_file, **graph_params)
            else:
                n2v.to_edgelist(sign1, neig1, graph_file)
        # split graph in train and test
        graph_train = graph_file + ".train"
        graph_test = graph_file + ".test"
        if not reuse or not os.path.isfile(graph_train) \
                or not os.path.isfile(graph_test):
            graph = SNAPNetwork.from_file(graph_file)
            n2v.split_edgelist(graph, graph_train, graph_test)
        # run Node2Vec to generate embeddings based on train
        emb_file = os.path.join(node2vec_path, 'n2v.emb')
        if not reuse or not os.path.isfile(emb_file):
            if node2vec_params:
                n2v.run(graph_train, emb_file, **node2vec_params)
            else:
                n2v.run(graph_train, emb_file)
        # create evaluation sign2
        eval_s2 = sign2(node2vec_path, self.dataset)
        # convert to signature h5 format
        if not reuse or not os.path.isfile(eval_s2.data_path):
            n2v.emb_to_h5(sign1.keys, emb_file, eval_s2.data_path)
        # save link prediction stats
        linkpred = LinkPrediction(eval_s2, SNAPNetwork.from_file(graph_train))
        perf_train_filen = os.path.join(node2vec_path, 'linkpred.train.json')
        linkpred.performance.toJSON(perf_train_filen)
        linkpred = LinkPrediction(eval_s2, SNAPNetwork.from_file(graph_test))
        perf_test_file = os.path.join(node2vec_path, 'linkpred.test.json')
        linkpred.performance.toJSON(perf_test_file)

    def grid_search_adanet(self, sign1, cc_root, job_path, parameters, dir_suffix="", traintest_file=None):
        """Perform a grid search.

        parameters = {
            'boosting_iterations': [10, 25, 50],
            'adanet_lambda': [1e-3, 5 * 1e-3, 1e-2],
            'layer_size': [8, 128, 512, 1024]
        }
        """
        import chemicalchecker
        from chemicalchecker.util.hpc import HPC
        from sklearn.model_selection import ParameterGrid

        gridsearch_path = os.path.join(
            self.model_path, 'grid_search_%s' % dir_suffix)
        if not os.path.isdir(gridsearch_path):
            os.makedirs(gridsearch_path)
        # prepare train-test file
        if traintest_file is None:
            traintest_file = os.path.join(gridsearch_path, 'traintest.h5')
            if not os.path.isfile(traintest_file):
                Traintest.create_signature_file(
                    sign1.data_path, self.data_path, traintest_file)
        elements = list()
        for params in ParameterGrid(parameters):
            model_dir = '-'.join("%s_%s" % kv for kv in params.items())
            params.update(
                {'model_dir': os.path.join(gridsearch_path, model_dir)})
            params.update({'traintest_file': traintest_file})
            elements.append({'adanet': params})

        # create job directory if not available
        if not os.path.isdir(job_path):
            os.mkdir(job_path)
        # create script file
        cc_config = os.environ['CC_CONFIG']
        cc_package = os.path.join(chemicalchecker.__path__[0], '../')
        script_lines = [
            "import sys, os",
            "import pickle",
            "os.environ['CC_CONFIG'] = '%s'" % cc_config,  # cc_config location
            "sys.path.append('%s')" % cc_package,  # allow package import
            "from chemicalchecker.core import ChemicalChecker",
            "cc = ChemicalChecker('%s')" % cc_root,
            "task_id = sys.argv[1]",  # <TASK_ID>
            "filename = sys.argv[2]",  # <FILE>
            "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
            "data = inputs[task_id]",  # elements for current job
            "for params in data:",  # elements are indexes
            "    ds = '%s'" % self.dataset,
            "    s1 = cc.get_signature('sign1', 'reference', ds)",
            "    n1 = cc.get_signature('neig1', 'reference', ds)",
            "    s2 = cc.get_signature('sign2', 'reference', ds, **params)",
            "    s2.fit(s1, n1)",
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'sign2_grid_search_adanet.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        params = {}
        params["num_jobs"] = len(elements)
        params["jobdir"] = job_path
        params["job_name"] = "CC_SIGN2_GRID_SEARCH_ADANET"
        params["elements"] = elements
        params["wait"] = False
        params["memory"] = 32
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC.from_config(Config())
        cluster.submitMultiJob(command, **params)
        return cluster

    def grid_search_node2vec(self, cc_root, job_path, parameters, dir_suffix=""):
        """Perform a grid search.

        parameters = {
            'd': [2**i for i in range(1,11)]
        }
        """
        import chemicalchecker
        from chemicalchecker.util.hpc import HPC
        from sklearn.model_selection import ParameterGrid
        from chemicalchecker.util.network import SNAPNetwork
        from chemicalchecker.tool.node2vec import Node2Vec

        n2v = Node2Vec(executable=Config().TOOLS.node2vec_exec)
        gridsearch_path = os.path.join(
            self.model_path, 'grid_search_%s' % dir_suffix)
        if not os.path.isdir(gridsearch_path):
            os.makedirs(gridsearch_path)
        elements = list()
        graph_file = os.path.join(
            self.model_path, 'node2vec', 'graph.edgelist')
        graph_train = graph_file + ".train"
        graph_test = graph_file + ".test"
        if not os.path.isfile(graph_train) \
                or not os.path.isfile(graph_test):
            graph = SNAPNetwork.from_file(graph_file)
            n2v.split_edgelist(graph, graph_train, graph_test)
        for params in ParameterGrid(parameters):
            model_dir = '-'.join("%s_%s" % kv for kv in params.items())
            params.update(
                {'model_dir': os.path.join(gridsearch_path, model_dir)})
            elements.append({'node2vec': params})

        # create job directory if not available
        if not os.path.isdir(job_path):
            os.mkdir(job_path)
        # create script file
        cc_config = os.environ['CC_CONFIG']
        cc_package = os.path.join(chemicalchecker.__path__[0], '../')
        script_lines = [
            "import sys, os",
            "import pickle",
            "os.environ['CC_CONFIG'] = '%s'" % cc_config,  # cc_config location
            "sys.path.append('%s')" % cc_package,  # allow package import
            "from chemicalchecker.core import ChemicalChecker",
            "cc = ChemicalChecker('%s')" % cc_root,
            "task_id = sys.argv[1]",  # <TASK_ID>
            "filename = sys.argv[2]",  # <FILE>
            "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
            "data = inputs[task_id]",  # elements for current job
            "for params in data:",  # elements are indexes
            "    ds = '%s'" % self.dataset,
            "    s1 = cc.get_signature('sign1', 'reference', ds)",
            "    n1 = cc.get_signature('neig1', 'reference', ds)",
            "    s2 = cc.get_signature('sign2', 'reference', ds, **params)",
            "    s2.eval_node2vec(s1, n1)",
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'sign2_grid_search_adanet.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        params = {}
        params["num_jobs"] = len(elements)
        params["jobdir"] = job_path
        params["job_name"] = "CC_SIGN2_GRID_SEARCH_NODE2VEC"
        params["elements"] = elements
        params["wait"] = False
        params["memory"] = 6
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC.from_config(Config())
        cluster.submitMultiJob(command, **params)
        return cluster
