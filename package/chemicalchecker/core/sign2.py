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
import shutil
import numpy as np
from time import time

from .signature_base import BaseSignature

from chemicalchecker.util.plot import Plot
from chemicalchecker.util import logged
from chemicalchecker.util import Config


@logged
class sign2(BaseSignature):
    """Signature type 2 class."""

    def __init__(self, signature_path, validation_path, dataset, **params):
        """Initialize the signature.

        Args:
            signature_path(str): The signature root directory.
            dataset(`Dataset`): `chemicalchecker.database.Dataset` object.
            params(): Parameters, expected keys are 'graph', 'node2vec', and
                'adanet'.
        """
        # Calling init on the base class to trigger file existence checks
        BaseSignature.__init__(self, signature_path,
                               validation_path, dataset, **params)
        self.validation_path = validation_path
        # generate needed paths
        self.data_path = os.path.join(signature_path, 'sign2.h5')
        self.model_path = os.path.join(signature_path, 'models')
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        self.stats_path = os.path.join(signature_path, 'stats')
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

    def fit(self, sign1, neig1, reuse=True):
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
            from chemicalchecker.tool.adanet import AdaNet, Traintest
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
            n2v.emb_to_h5(sign1, emb_file, self.data_path)
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
        sign2_plot = Plot(self.dataset, self.stats_path, self.validation_path)
        sign2_plot.sign2_feature_distribution_plot(self)
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
            traintest_file = adanet_params.get(
                'traintest_file', traintest_file)
            adanet_params.pop('traintest_file')
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
        sign2_plot = Plot(self.dataset, adanet_path, self.validation_path)
        nearest_neighbor_pred = self.predict_nearest_neighbor(traintest_file)
        extra_preditors = dict()
        extra_preditors['NearestNeighbor'] = nearest_neighbor_pred
        ada.save_performances(adanet_path, sign2_plot, extra_preditors)
        self.__log.debug('model saved to %s' % adanet_path)

        self.mark_ready()

    def predict(self, sign1):
        """Use the learned model to predict the signature."""
        try:
            from chemicalchecker.tool.adanet import AdaNet
        except ImportError as err:
            raise err
        # load AdaNet model
        adanet_path = os.path.join(self.model_path, 'adanet', 'savedmodel')
        self.__log.debug('loading model from %s' % adanet_path)
        return AdaNet.predict_signature(adanet_path, sign1)

    def predict_nearest_neighbor(self, traintest_file):
        """Prediction with nearest neighbor.

        Find nearest neighbor in sign 1 and mapping it to known sign 2.
        """
        from .data import DataFactory
        from .neig1 import neig1
        from chemicalchecker.tool.adanet import Traintest
        self.__log.info('Performing Nearest Neighbor prediction.')
        # create directory to save neig and sign (delete if exists)
        nn_path = os.path.join(self.model_path, "nearest_neighbor")
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
            sign1_data = traintest.get_all_x()
            sign2_data = traintest.get_all_y()
            traintest.close()
            # fit on train set
            if ds == "train":
                # signaturize dataset
                sign1_dest = os.path.join(nn_path, "sign1")
                os.makedirs(sign1_dest)
                nn_sign1 = DataFactory.signaturize(
                    "sign1", sign1_dest, sign1_data, dataset=self.dataset)
                sign2_dest = os.path.join(nn_path, "sign2")
                os.makedirs(sign2_dest)
                nn_sign2 = DataFactory.signaturize(
                    "sign2", sign2_dest, sign2_data, dataset=self.dataset)
                # create temporary neig1 and call fit
                neig1_dest = os.path.join(nn_path, "neig1")
                os.makedirs(neig1_dest)
                nn_neig1 = neig1(neig1_dest, neig1_dest, self.dataset)
                nn_neig1.fit(nn_sign1)
            # get nearest neighbor indices and keys
            nn_idxs = nn_neig1.get_kth_nearest(sign1_data, 1)
            # save nearest neighbor signatures as predictions
            nn_pred[ds] = list()
            for idx in nn_idxs:
                nn_pred[ds].append(nn_sign2[idx])
            nn_pred[ds] = np.stack(nn_pred[ds])
        nn_pred_end = time()
        nn_pred['time'] = nn_pred_end - nn_pred_start
        nn_pred['name'] = "NearestNeighbor"
        return nn_pred

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
        node2vec_path = os.path.join(self.model_path, 'node2vec')
        if node2vec_params:
            if 'model_dir' in node2vec_params:
                node2vec_path = node2vec_params.pop('model_dir')
        if not reuse or not os.path.isdir(node2vec_path):
            os.makedirs(node2vec_path)
        # use neig1 to generate the Node2Vec input graph (as edgelist)
        graph_params = self.params['graph']
        graph_file = os.path.join(self.model_path, 'graph.edgelist')
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
        # check that all molecules are considered in all the graph
        for g_file in [graph_file, graph_train, graph_test]:
            with open(g_file, 'r') as fh:
                lines = fh.readlines()
            graph_mol = set(l.split()[0] for l in lines)
            # we can just compare the total nr
            if not len(graph_mol) == len(sign1.unique_keys):
                raise Exception("Graph %s is missing nodes." % g_file)
            # save graph stats
            graph_stat_file = g_file + ".stats"
            graph = None
            if not reuse or not os.path.isfile(graph_stat_file):
                graph = SNAPNetwork.from_file(g_file)
                graph.stats_toJSON(graph_stat_file)
        # run Node2Vec to generate embeddings based on train
        emb_file = os.path.join(node2vec_path, 'n2v.emb')
        if not reuse or not os.path.isfile(emb_file):
            if node2vec_params:
                n2v.run(graph_train, emb_file, **node2vec_params)
            else:
                n2v.run(graph_train, emb_file)
        # convert to signature h5 format
        if not reuse or not os.path.isfile(self.data_path):
            n2v.emb_to_h5(sign1, emb_file, self.data_path)
        # save link prediction stats
        linkpred_file_train = os.path.join(
            self.model_path, 'linkpred.train.json')
        if not reuse or not os.path.isfile(linkpred_file_train):
            graph = SNAPNetwork.from_file(graph_train)
            linkpred = LinkPrediction(self, graph)
            linkpred.performance.toJSON(linkpred_file_train)
        linkpred_file_test = os.path.join(
            self.model_path, 'linkpred.test.json')
        if not reuse or not os.path.isfile(linkpred_file_test):
            graph = SNAPNetwork.from_file(graph_test)
            linkpred = LinkPrediction(self, graph)
            linkpred.performance.toJSON(linkpred_file_test)

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
        from chemicalchecker.tool.adanet import Traintest
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
        cluster = HPC(Config())
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

        gridsearch_path = os.path.join(
            self.model_path, 'grid_search_%s' % dir_suffix)
        if not os.path.isdir(gridsearch_path):
            os.makedirs(gridsearch_path)
        elements = list()
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
        params["memory"] = 16
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC(Config())
        cluster.submitMultiJob(command, **params)
        return cluster
