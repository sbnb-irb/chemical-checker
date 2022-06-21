"""Link prediction.

The idea is to check wether the embedding (signature 2) distances are
predictive of a link being or not present.
"""
import numpy as np
from tqdm import tqdm
from numpy.random import randint
from scipy.spatial.distance import cosine

from .performance import PerformanceBinary

from chemicalchecker.util import logged


@logged
class LinkPrediction():
    """LinkPrediction class."""

    def __init__(self, sign2, network, metric=cosine, limit_nodes=None):
        """Initialize a LinkPrediction instanca.
        Args:
            sign2 (sign): Signature or embedding to validate.
            network (network): The network that we want to reconstruct.
            metric (set): The function used to compute vector distance.
            limit_nodes (set): Limit sampling to nodes in this set.
        """
        self.sign2 = sign2
        self.network = network
        y_true, y_pred = self.get_sample_actual_pred(3, metric, limit_nodes)
        self.performance = PerformanceBinary(y_true, y_pred)

    def get_sample_actual_pred(self, edges_to_sample, metric, limit_nodes):
        """Sample positive (present) and negative (absent).

        Args:
            edges_to_sample (int): Number of edges to sample per node.
            metric (func): The function used to compute vector distance.
            limit_nodes (set): Limit sampling to nodes in this set.
        """
        y_positive = list()
        y_pred_pos = list()
        y_negative = list()
        y_pred_neg = list()
        # iterate on nodes
        all_nodes_net = set(list(self.network.nodes()))
        all_nodes_emb = set(range(self.sign2.shape[0]))
        if not all_nodes_net == all_nodes_emb:
            self.__log.warn("Network and embedding do not have same nodes!")
        all_nodes_set = all_nodes_net & all_nodes_emb
        if limit_nodes:
            all_nodes_set = all_nodes_set & limit_nodes
        all_nodes = list(all_nodes_set)
        matrix = self.sign2.get_h5_dataset('V')
        if len(all_nodes) < 100:
            raise Exception(
                "Insufficient nodes for validation: %s" % len(all_nodes))
        for node in tqdm(all_nodes):
            node_sign2 = matrix[node]
            # get edges_to_sample true and distances
            neig = list(self.network.neighbors(node))
            neig = list(set(neig) & all_nodes_set)
            if len(neig) < edges_to_sample:
                sample_size = len(neig)
            else:
                sample_size = edges_to_sample
            samples = neig[:sample_size]
            for sample in samples:
                y_positive.append(1)
                y_pred_pos.append(1 - metric(node_sign2, matrix[sample]))
            # sample until edges_to_sample false and distances
            samples = list()
            while len(samples) < sample_size:
                sample = all_nodes[randint(len(all_nodes))]
                if sample == node or sample in neig or sample in samples:
                    continue
                samples.append(sample)
                y_negative.append(0)
                y_pred_neg.append(1 - metric(node_sign2, matrix[sample]))

        dists = "min %.2f 1st %.2f 2nd %.2f, 3rd %.2f max %.2f" % (
            min(y_pred_pos), np.percentile(
                y_pred_pos, 25), np.median(y_pred_pos),
            np.percentile(y_pred_pos, 75), max(y_pred_pos))
        sampling_pos = "y_positive cases %s : %s" % (len(y_positive), dists)
        self.__log.info(sampling_pos)
        dists = "min %.2f 1st %.2f 2nd %.2f, 3rd %.2f max %.2f" % (
            min(y_pred_neg), np.percentile(
                y_pred_neg, 25), np.median(y_pred_neg),
            np.percentile(y_pred_neg, 75), max(y_pred_neg))
        sampling_neg = "y_negative cases %s : %s" % (len(y_negative), dists)
        self.__log.info(sampling_neg)
        return (y_positive + y_negative, y_pred_pos + y_pred_neg)
