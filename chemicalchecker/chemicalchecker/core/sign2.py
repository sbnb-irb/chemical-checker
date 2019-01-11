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
from .signature_base import BaseSignature
from chemicalchecker.util import logged
from chemicalchecker.util import Config
from chemicalchecker.util import SNAPNetwork
from chemicalchecker.util import LinkPrediction
from chemicalchecker.tool import Node2Vec  # , AdaNet, Traintest


@logged
class sign2(BaseSignature):
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
        BaseSignature.__init__(self, signature_path, dataset, **params)
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
        """
        # step 1: Node2Vec (learn graph embedding) input is neig1
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
        # save graph stats
        graph_stat_file = os.path.join(self.stats_path, 'graph_stats.json')
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
            graph = SNAPNetwork.from_file(graph_file)
            linkpred = LinkPrediction(self, graph)
            linkpred.performance.toJSON(linkpred_file)
        """
        # step 2: AdaNet (learn to predict sign2 from sign1 without Node2Vec)
        self.__log.debug('AdaNet fit %s with Node2Vec output' % sign1)
        adanet_params = self.params['adanet']
        adanet_path = os.path.join(self.model_path, 'adanet')
        if not os.path.isdir(adanet_path):
            os.makedirs(adanet_path)
        if adanet_params:
            ada = AdaNet(model_path=adanet_path, **adanet_params)
        else:
            ada = AdaNet(model_path=adanet_path)
        # prepare train-test file
        traintest_file = os.path.join(adanet_path, 'traintest.h5')
        Traintest.create(sign1.data_path, self.data_path, traintest_file)
        # learn NN with AdaNet
        ada.train_and_evaluate(traintest_file)
        self.__log.debug('model saved to %s' % self.model_path)
        """

    def predict(self, sign1):
        """Use the learned model to predict the signature."""
        """
        # load AdaNet model
        adanet_path = os.path.join(self.model_path, 'adanet')
        self.__log.debug('loading model from %s' % adanet_path)
        ada = AdaNet(model_path=adanet_path)
        # prepare input file
        self.__log.debug('AdaNet predict %s' % sign1)
        ada.prepare_predict(sign1.data_path)
        return ada.predict()
        """

    def statistics(self):
        """Perform a statistics """
        BaseSignature.validate(self)
