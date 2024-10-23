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
predict() capabilities.
"""
import os
import h5py
import shutil
import numpy as np
from time import time

from .signature_base import BaseSignature
from .signature_data import DataSignature

from chemicalchecker.util.plot import Plot
from chemicalchecker.util import logged, Config
from chemicalchecker.util.splitter import Traintest


@logged
class sign2(BaseSignature, DataSignature):
    """Signature type 2 class."""

    def __init__(self, signature_path, dataset, **params):
        """Initialize a Signature.

        Args:
            signature_path(str): The signature root directory.
            dataset(`Dataset`): `chemicalchecker.database.Dataset` object.
            params(): Parameters, expected keys are 'graph', 'node2vec', and
                'adanet'.
        """
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(self, signature_path, dataset, **params)
        self.data_path = os.path.join(self.signature_path, 'sign2.h5')
        DataSignature.__init__(self, self.data_path)
        # assign dataset
        self.dataset = dataset

    def fit(self, sign1=None, neig1=None, reuse=True, compare_nn=False,
            oos_predictor=True, graph_kwargs={}, node2vec_kwargs={},
            adanet_kwargs={'cpu': 1}, **kwargs):
        """Fit signature 2 given signature 1 and its nearest neighbors.

        Node2vec embeddings are computed using the graph derived from sign1.
        The predictive model is learned with AdaNet.

        Args:
            sign1(sign1): Signature type 1.
            neig1(neig1): Nearest neighbor of type 1.
            reuse(bool): Reuse already generated intermediate files. Set to
                False to re-train from scratch.
        """
        try:
            from chemicalchecker.util.network import SNAPNetwork
            from chemicalchecker.util.performance import LinkPrediction
            from chemicalchecker.tool.adanet import AdaNet
            from chemicalchecker.tool.node2vec import Node2Vec
        except ImportError as err:
            raise err
        BaseSignature.fit(self, **kwargs)
        # signature specific checks
        if self.molset != "reference":
            self.__log.debug("Fit will be done with the reference sign2")
            self = self.get_molset("reference")
        if sign1 is None:
            sign1 = self.get_sign('sign1').get_molset("reference")
        if neig1 is None:
            neig1 = sign1.get_neig().get_molset("reference")
        if sign1.cctype != "sign1":
            raise Exception("A signature type 1 is expected!")
        if sign1.molset != "reference":
            self.__log.debug("Fit will be done with the reference sign1")
            sign1 = self.get_sign('sign1').get_molset("reference")
        if neig1.cctype != "neig1":
            raise Exception("A neighbors of signature type 1 is expected!")
        if neig1.molset != "reference":
            self.__log.debug("Fit will be done with the reference neig1")
            neig1 = sign1.get_neig().get_molset("reference")

        #########
        # step 1: Node2Vec (learn graph embedding) input is neig1
        #########
        self.update_status("Node2Vec")
        self.__log.debug('Node2Vec on %s' % sign1)
        n2v = Node2Vec(executable=Config().TOOLS.node2vec_exec)
        # use neig1 to generate the Node2Vec input graph (as edgelist)
        node2vec_path = os.path.join(self.model_path, 'node2vec')
        if not os.path.isdir(node2vec_path):
            os.makedirs(node2vec_path)
        graph_file = os.path.join(node2vec_path, 'graph.edgelist')
        if not reuse or not os.path.isfile(graph_file):
            n2v.to_edgelist(sign1, neig1, graph_file, **graph_kwargs)
        # check that all molecules are considered in the graph
        with open(graph_file, 'r') as fh:
            lines = fh.readlines()
        graph_mol = set(l.split()[0] for l in lines)
        # we can just compare the total nr
        if not len(graph_mol) == len(neig1.unique_keys):
            raise Exception("Graph %s is missing nodes." % graph_file)
        # save graph stats
        graph_stat_file = os.path.join(self.stats_path, 'graph_stats.json')
        graph = None
        if not reuse or not os.path.isfile(graph_stat_file):
            graph = SNAPNetwork.from_file(graph_file)
            graph.stats_toJSON(graph_stat_file)
        # run Node2Vec to generate embeddings
        emb_file = os.path.join(node2vec_path, 'n2v.emb')
        if not reuse or not os.path.isfile(emb_file):
            n2v.run(graph_file, emb_file, **node2vec_kwargs)
        # convert to signature h5 format
        if not reuse or not os.path.isfile(self.data_path):
            n2v.emb_to_h5(sign1.keys, emb_file, self.data_path)
        # save link prediction stats
        linkpred_file = os.path.join(self.stats_path, 'linkpred.json')
        if not reuse or not os.path.isfile(linkpred_file):
            if not graph:
                graph = SNAPNetwork.from_file(graph_file)
            try:
                linkpred = LinkPrediction(self, graph)
                linkpred.performance.toJSON(linkpred_file)
            except Exception as ex:
                self.__log.error('Problem with LinkPrediction: %s' % str(ex))
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
        if oos_predictor:
            self.update_status("Training out-of-sample predictor")
            self.__log.debug('AdaNet fit %s with Node2Vec output' % sign1)
            # get params and set folder
            adanet_path = os.path.join(self.model_path, 'adanet')
            adanet_path = adanet_kwargs.get('model_dir', adanet_path)
            if not reuse or not os.path.isdir(adanet_path):
                os.makedirs(adanet_path, exist_ok=True)
            # prepare train-test file
            traintest_file = os.path.join(adanet_path, 'traintest.h5')
            traintest_file = adanet_kwargs.get(
                'traintest_file', traintest_file)
            if not reuse or not os.path.isfile(traintest_file):
                Traintest.create_signature_file(
                    sign1.data_path, self.data_path, traintest_file)
            ada = AdaNet(model_dir=adanet_path,
                         traintest_file=traintest_file, **adanet_kwargs)
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
            ada.save_performances(adanet_path, sign2_plot,
                                  extra_predictors=extra_preditors)
            self.__log.debug('model saved to %s' % adanet_path)

        self.update_status("Generating `full` molset")
        cc_tmp = self.get_cc()
        sign1_full = cc_tmp.get_signature('sign1', 'full', self.dataset)
        sign2_full = cc_tmp.get_signature('sign2', 'full', self.dataset)
        # we want agreement between reference and full
        # so we overwrite the original embeddings using the predictor
        sign1_ref = cc_tmp.get_signature('sign1', 'reference', self.dataset)
        sign2_ref = cc_tmp.get_signature('sign2', 'reference', self.dataset)
        if oos_predictor:
            self.predict(sign1_full, sign2_full.data_path)
            self.predict(sign1_ref, sign2_ref.data_path)
        else:
            self.save_full(sign2_full.data_path)
        # finalize signature
        BaseSignature.fit_end(self,  **kwargs)

    def predict(self, sign1, destination=None):
        """Use the learned model to predict the signature.

        Args:
            sign1(signature): A valid Signature type 1
            destination(None|path|signature): If None the prediction results are
                returned as dictionary, if str then is used as path for H5 data,
                if empty Signature type 2 its data_path is used as destination.
        """
        try:
            from chemicalchecker.tool.adanet import AdaNet
        except ImportError as err:
            raise err
        if isinstance(destination, BaseSignature):
            destination = destination.data_path
        # load AdaNet model
        adanet_path = os.path.join(self.model_path, 'adanet', 'savedmodel')
        self.__log.debug('loading model from %s' % adanet_path)
        predict_fn = AdaNet.predict_fn(adanet_path)
        tot_inks = len(sign1.keys)
        if destination is None:
            results = {
                "V": AdaNet.predict(sign1[:], predict_fn),
                "keys": sign1.keys
            }
            return results
        else:
            if isinstance(destination, BaseSignature):
                destination = destination.data_path
            with h5py.File(destination, "w") as results:
                # initialize V and keys datasets
                results.create_dataset('V', (tot_inks, 128), dtype='float32')
                results.create_dataset(
                    'keys', data=np.array(sign1.keys,
                                          DataSignature.string_dtype()))
                results.create_dataset("shape", data=(tot_inks, 128))
                # predict signature 2
                for chunk in sign1.chunker():
                    results['V'][chunk] = AdaNet.predict(
                        sign1[chunk], predict_fn)

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

    def eval_node2vec(self, sign1, neig1, reuse=True, graph_kwargs={},
                      node2vec_kwargs={}):
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
        node2vec_path = os.path.join(self.model_path, 'node2vec_eval')
        node2vec_path = node2vec_kwargs.get('model_dir', node2vec_path)
        if not reuse or not os.path.isdir(node2vec_path):
            os.makedirs(node2vec_path)
        # use neig1 to generate the Node2Vec input graph (as edgelist)
        graph_file = os.path.join(
            self.model_path, 'node2vec', 'graph.edgelist')
        if not reuse or not os.path.isfile(graph_file):
            n2v.to_edgelist(sign1, neig1, graph_file, **graph_kwargs)
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
            n2v.run(graph_train, emb_file, **node2vec_kwargs)
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

    def grid_search_adanet(self, sign1, cc_root, job_path, parameters,
                           dir_suffix="", traintest_file=None):
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
