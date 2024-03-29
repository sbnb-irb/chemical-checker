"""Wrapper for SNAP C++ implementation of node2vec.

An algorithmic framework for representational learning on graphs. [Aug 27 2018]
usage: node2vec
   -i:Input graph path (default:'graph/karate.edgelist')
   -o:Output graph path (default:'emb/karate.emb')
   -d:Number of dimensions. Default is 128 (default:128)
   -l:Length of walk per source. Default is 80 (default:80)
   -r:Number of walks per source. Default is 10 (default:10)
   -k:Context size for optimization. Default is 10 (default:10)
   -e:Number of epochs in SGD. Default is 1 (default:1)
   -p:Return hyperparameter. Default is 1 (default:1)
   -q:Inout hyperparameter. Default is 1 (default:1)
   -v Verbose output.
   -dr Graph is directed.
   -w Graph is weighted.
   -ow Output random walks instead of embeddings.


"""
import os
import h5py
import subprocess
import operator
import pickle
import numpy as np
import gc
from tqdm import tqdm
import collections
import distutils.spawn
from datetime import datetime
from bisect import bisect_left
from chemicalchecker.util import logged

from chemicalchecker.core.signature_data import DataSignature


@logged
class Node2Vec():
    """Wrapper to run SNAP's node2vec."""

    def __init__(self, executable="node2vec", **kwargs):
        """Check if executable is found."""
        exec_file = distutils.spawn.find_executable(executable)
        if not exec_file:
            raise Exception("node2vec executable not found.")
        self.executable = exec_file

    @staticmethod
    def heuristic_max_degree(nr_nodes, max_mbytes=6000.):
        """Return the maximum degree.

        The heuristic is based on the limiting factor that is node2vec's
        memory footprint. To precompute transition probabilities, node2vec
        needs 12 bytes for each triplet node-neighbor-neighbor.

        Args:
            nr_nodes(int): Number of nodes the network will contain.
            max_mbytes(float): Maximum RAM consumption to allow (in MBs).
        """
        max_degree = 10
        for max_deg in reversed(range(10, 1000, 5)):
            mbytes = (nr_nodes * max_deg * max_deg * 12) / 1000000.
            if mbytes < max_mbytes:
                max_degree = max_deg
                break
        return max_degree

    def _pvalue_to_weigth(self, pvalues):
        """Scale p-values linearly to weights.

        shift minimum to zero
        w = p - min_pvalue
        scale range to one
        w *= 1/range_pvalue
        scale range to weight range
        w *= range_weight
        shift minimum to weight minimum
        w += min_weight
        weight = (pval_log - min_pvalue) * range_weight / range_pvalue + min_w
        """
        min_w = self.min_weight
        max_w = self.max_weight
        min_p = -np.log10(self.min_pvalue)
        max_p = -np.log10(self.max_pvalue)
        range_p = max_p - min_p
        range_w = max_w - min_w
        # compute weights
        return (-np.log10(pvalues) - min_p) * range_w / range_p + min_w

    def emb_to_h5(self, keys, in_file, out_file):
        """Convert from native node2vec format to familiar h5.

        We need to map back from sign1 ids to inchikeys and sort.
        """
        self.__log.info("Converting %s to %s" % (in_file, out_file))
        inchikeys = keys
        with open(in_file, 'r') as fh:
            words = list()
            vectors = list()
            header = fh.readline()  # skip first row
            nr_words = int(header.split()[0])
            nr_components = int(header.split()[1])
            self.__log.info("words: %s features: %s", nr_words, nr_components)
            try:
                for idx, line in enumerate(fh):
                    fields = line.split()
                    # first colum is id
                    word = inchikeys[int(fields[0])]
                    # rest is embedding
                    vector = np.fromiter((float(x) for x in fields[1:]),
                                         dtype=np.float32)
                    words.append(word)
                    vectors.append(vector)
            except Exception as ex:
                self.__log.info("Error at line %s: %s", idx, str(ex))
        # consistency check
        assert(len(words) == len(inchikeys))
        assert(len(words) == len(vectors))
        # to numpy arrays
        words = np.array(words)
        matrix = np.array(vectors)
        # get them sorted
        sorted_idx = np.argsort(words)
        with h5py.File(out_file, "w") as hf:
            hf.create_dataset('keys',
                              data=np.array(words[sorted_idx],
                                            DataSignature.string_dtype()))
            hf.create_dataset('V', data=matrix[sorted_idx], dtype=np.float32)
            hf.create_dataset("shape", data=matrix.shape)

    def to_edgelist(self, sign1, neig1, out_file, **params):
        """Convert Nearest-neighbor to an edgelist.

        Args:
            sign1(str): Signature 1, h5 file path.
            neig1(str): Nearest neighbors 1, h5 file path.
            out_file(str): Destination file.
            params(dict): Parameters defining similarity network.
        """
        self.min_weight = params.get("min_weight", 1e-2)
        self.max_weight = params.get("max_weight", 1.0)
        self.min_pvalue = params.get("min_pvalue", 1e-2)
        self.max_pvalue = params.get("max_pvalue", 1e-6)
        self.min_degree = params.get("min_degree", 3)
        self.range_pvalue = self.max_pvalue - self.min_pvalue
        self.range_weight = self.max_weight - self.min_weight
        # get sign1 background distances thresholds
        thresholds = sign1.background_distances('cosine')
        thr_pvals = thresholds['pvalue']
        thr_dists = thresholds['distance']
        # derive max_degree
        mem_max_degree = Node2Vec.heuristic_max_degree(sign1.shape[0])
        self.max_degree = params.get("max_degree", mem_max_degree)
        if self.max_degree > mem_max_degree:
            self.__log.warn('user max_degree too large: %s', self.max_degree)
            self.__log.warn('using memory safe degree: %s', mem_max_degree)
            self.max_degree = mem_max_degree
        # write edgelist
        with open(out_file, 'w') as fh:
            for src_idx, neighbors in enumerate(neig1):
                # self.__log.debug('molecule: %s', src_idx)
                neig_idxs, neig_dists = neighbors
                # exclude self
                self_idx = np.argwhere(neig_idxs == src_idx)
                neig_idxs = np.delete(neig_idxs, self_idx)
                neig_dists = np.delete(neig_dists, self_idx)
                edges = self._get_edges(
                    neig_idxs, neig_dists, thr_pvals, thr_dists)
                for dst_idx, weight in edges:
                    fh.write("%i %i %.4f\n" % (src_idx, dst_idx, weight))

    def merge_edgelists(self, signs, edgefiles, out_file, **params):
        """Convert Nearest-neighbor to an edgelist.

        Args:
            signs(list): List of signature objects.
            edgefiles(list): List of edge files.
            out_file(str): Destination file.
            params(dict): Parameters defining similarity network.
        """

        edges_weights = collections.defaultdict(dict)
        all_keys = set()
        for i, edgefile in enumerate(edgefiles):
            keys = signs[i].keys
            all_keys.update(keys)
            with open(edgefile, 'r') as fh:
                for line in fh:
                    elems = line.rstrip().split(' ')
                    key1 = keys[int(elems[0])]
                    key2 = keys[int(elems[1])]
                    if key2 not in edges_weights[key1]:
                        edges_weights[key1][key2] = 0.0
                    edges_weights[key1][key2] = max(
                        edges_weights[key1][key2], float(elems[2]))

        mem_max_degree = Node2Vec.heuristic_max_degree(len(edges_weights))
        self.max_degree = params.get("max_degree", mem_max_degree)
        if self.max_degree > mem_max_degree:
            self.__log.warn('user max_degree too large: %s', self.max_degree)
            self.__log.warn('using memory safe degree: %s', mem_max_degree)
            self.max_degree = mem_max_degree

        all_keys_list = list(all_keys)
        all_keys_list.sort()

        dictOfkeys = {all_keys_list[i]: i for i in range(0, len(all_keys_list))}

        # write edgelist
        with open(out_file, 'w') as fh:
            for node, map_nodes in edges_weights.items():
                # self.__log.debug('molecule: %s', node)

                mols = list(map_nodes.items())
                mols.sort(key=lambda tup: tup[1])
                mols.reverse()
                for dst_idx, weight in mols[:self.max_degree]:
                    fh.write("%s %s %.4f\n" %
                             (dictOfkeys[node], dictOfkeys[dst_idx], min(weight, 1.0)))

    def split_edgelist(self, full_graph, train, test, train_fraction=0.8):
        """Split a graph in train and test.

        Give a Network object split it in two sets, train and test, so that
        train has train_fraction of edges for each node.
        """
        train_out = open(train, 'w')
        test_out = open(test, 'w')
        # we want each node to keep edges so the split is on each node
        for node in tqdm(full_graph.nodes()):
            neighbors = list(full_graph.neighbors(node, data=True))
            np.random.shuffle(neighbors)
            split_id = int(len(neighbors) * train_fraction)
            for neig, weight in neighbors[:split_id]:
                train_out.write("%i %i %.4f\n" % (node, neig, weight))
            for neig, weight in neighbors[split_id:]:
                test_out.write("%i %i %.4f\n" % (node, neig, weight))
        train_out.close()
        test_out.close()

    def _get_edges(self, neig_idxs, neig_dists, thr_pvals, thr_dists):
        """Get a molecule neighbors and edge weight.

        We have a list of all possible pvalues and the pvalue id for each
        similar molecule ('thresholds'). We want as many as possible
        significant pvalues (< MIN_PVALUE, >= MAX_PVALUE). We want at least
        MIN_DEGREE similar molecules (even if less significant than MIN_PVALUE)
        and less than MAX_DEGREE (even if kicking out significant ones).

        Args:
            neig_idxs(array): The indexes of the neighbour molecule.
            neig_dists(array): Corresponding distances.
            thr_pvals(array): Array with threshold of pvalues.
            thr_dists(array): Corresponding threshold distances.
        Returns:
            edges(list(tuple)): list with with destination and weight.
        """
        curr_pval_idx = bisect_left(thr_pvals, self.min_pvalue) + 1
        curr_pval = thr_pvals[curr_pval_idx]
        neig_mask = neig_dists <= thr_dists[curr_pval_idx]
        degree = np.count_nonzero(neig_mask)
        # self.__log.debug("curr_pval[%i] %s - degree %s",
        #                  curr_pval_idx, curr_pval, degree)
        # we want more than MIN_DEGREE and less than MAX_DEGREE
        if self.min_degree <= degree <= self.max_degree:
            considered_mol_ids = neig_idxs[neig_mask]
        # if we have too few
        elif degree < self.min_degree:
            # iterate increasing threshold pvalue
            for incr_pval_idx in range(curr_pval_idx + 1, len(thr_dists)):
                # check rank degree
                incr_pval = thr_pvals[incr_pval_idx]
                neig_mask = neig_dists <= thr_dists[incr_pval_idx]
                degree = np.count_nonzero(neig_mask)
                # self.__log.debug("incr_pval[%i] %s - degree %s",
                #                  incr_pval_idx, incr_pval, degree)
                # increasing threshold we have too many
                if degree >= self.min_degree:
                    # sample neighbors from previous threshold
                    prev_neig_mask = np.logical_and(
                        neig_dists <= thr_dists[incr_pval_idx],
                        neig_dists > thr_dists[incr_pval_idx - 1])
                    already_in = neig_dists <= thr_dists[
                        incr_pval_idx - 1]
                    to_add = self.min_degree - np.count_nonzero(already_in)
                    # self.__log.debug(
                    #     "sampling %s from pval %s", to_add, incr_pval)
                    sampled_idxs = np.random.choice(
                        np.argwhere(prev_neig_mask).flatten(),
                        to_add, replace=False)
                    sampled_neig_mask = np.full(neig_mask.shape, False)
                    sampled_neig_mask[sampled_idxs] = True
                    # add neighbors within current threshold
                    neig_mask = np.logical_or(already_in, sampled_neig_mask)
                    break
                # corner case where the highest threshold has to be sampled
                elif incr_pval_idx == len(thr_dists):
                    # sample from highest threshold
                    already_in = neig_dists <= thr_dists[
                        incr_pval_idx - 1]
                    to_add = self.min_degree - np.count_nonzero(already_in)
                    # self.__log.debug("corner sampling %s", to_add)
                    curr_neig_mask = neig_dists == thr_dists[
                        incr_pval_idx]
                    sampled_idxs = np.random.choice(
                        np.argwhere(curr_neig_mask).flatten(),
                        to_add, replace=False)
                    sampled_neig_mask = np.full(neig_mask.shape, False)
                    sampled_neig_mask[sampled_idxs] = True
                    # add neighbors within previous threshold
                    prev_neig_mask = neig_dists <= thr_dists[
                        incr_pval_idx - 1]
                    neig_mask = np.logical_or(
                        prev_neig_mask, sampled_neig_mask)
                    break
            # at this point the mask contain the right amount of neighbors
            considered_mol_ids = neig_idxs[neig_mask]
            assert(len(considered_mol_ids) == self.min_degree)
        # else too many
        else:
            # iterate decreasing threshold pvalue
            for decr_pval_idx in range(curr_pval_idx - 1, -1, -1):
                # check degree
                decr_pval = thr_pvals[decr_pval_idx]
                neig_mask = neig_dists <= thr_dists[decr_pval_idx]
                degree = np.count_nonzero(neig_mask)
                # self.__log.debug("decr_pval[%i] %s - degree %s",
                #                  decr_pval_idx, decr_pval, degree)
                # reducing threshold we have too few
                if degree <= self.max_degree:
                    # sample neighbors from previous threshold
                    to_add = self.max_degree - degree
                    # self.__log.debug(
                    #     "sampling %s from pval %s", to_add, decr_pval)
                    prev_neig_mask = np.logical_and(
                        neig_dists <= thr_dists[decr_pval_idx + 1],
                        neig_dists > thr_dists[decr_pval_idx])
                    sampled_idxs = np.random.choice(
                        np.argwhere(prev_neig_mask).flatten(),
                        to_add, replace=False)
                    sampled_neig_mask = np.full(neig_mask.shape, False)
                    sampled_neig_mask[sampled_idxs] = True
                    # add neighbors within current threshold
                    neig_mask = np.logical_or(neig_mask, sampled_neig_mask)
                    break
                # corner case where the lowest threshold has to be sampled
                elif decr_pval_idx == 0:
                    to_add = self.max_degree
                    # self.__log.debug("corner sampling %s", to_add)
                    sampled_idxs = np.random.choice(
                        np.argwhere(neig_mask).flatten(),
                        to_add, replace=False)
                    neig_mask = np.full(neig_mask.shape, False)
                    neig_mask[sampled_idxs] = True
                    break
            # at this point the mask contain the right amount of neighbors
            considered_mol_ids = neig_idxs[neig_mask]
            assert(len(considered_mol_ids) == self.max_degree)
        # convert distances to p-values index
        pvalues_idx = np.zeros(neig_dists.shape, dtype=int)
        for dist in thr_dists[:-1]:  # skip last distance for corner cases
            pvalues_idx += neig_dists > dist
        # convert p-value indexes to p-values
        pvalues = thr_pvals[pvalues_idx]
        # replace p-value 0.0 with 1e-6 to avoid -inf
        pvalues[np.argwhere(pvalues == 0.0).flatten()] = self.max_pvalue
        # scale p-values linearly to weights
        weights = self._pvalue_to_weigth(pvalues[neig_mask])
        # we force a minimal degree so we might have weights below MIN_WEIGHT
        # we cap to a default values one order of magnitude below MIN_WEIGHT
        weights[weights < self.min_weight] = self.min_weight / 10.
        # self.__log.debug("similar molecules considered: %s",
        #                  len(considered_mol_ids))
        return zip(considered_mol_ids, weights)

    def run(self, i, o, **kwargs):
        """Call external exe with given parameters.

        Args:
            i:Input graph path (default:'graph/karate.edgelist')
            o:Output graph path (default:'emb/karate.emb')
            d:Number of dimensions. Default is 128 (default:128)
            l:Length of walk per source. Default is 80 (default:80)
            r:Number of walks per source. Default is 10 (default:10)
            k:Context size for optimization. Default is 10 (default:10)
            e:Number of epochs in SGD. Default is 1 (default:1)
            p:Return hyperparameter. Default is 1 (default:1)
            q:Inout hyperparameter. Default is 1 (default:1)
            v Verbose output.
            dr Graph is directed.
            w Graph is weighted.
            ow Output random walks instead of embeddings.
        """
        # check input
        if not os.path.isfile(i):
            raise Exception("Input file not found.")

        # get arguments or default values
        d = kwargs.get("d", 128)
        l = kwargs.get("l", 80)
        r = kwargs.get("r", 10)
        k = kwargs.get("k", 10)
        e = kwargs.get("e", 1)
        p = kwargs.get("p", 1)
        q = kwargs.get("q", 1)
        v = kwargs.get("v", True)
        ow = kwargs.get("ow", False)
        dr = kwargs.get("dr", False)
        w = kwargs.get("w", True)
        cpu = kwargs.get("cpu", 1)

        # prepare arguments
        args = [
            "-i:%s" % i,
            "-o:%s" % o,
            "-d:%s" % d,
            "-l:%s" % l,
            "-r:%s" % r,
            "-k:%s" % k,
            "-e:%s" % e,
            "-p:%s" % p,
            "-q:%s" % q,
        ]
        if v:
            args.append("-v")
        if dr:
            args.append("-dr")
        if w:
            args.append("-w")
        if ow:
            args.append("-ow")

        # this enables using as many CPU as required
        os.environ['OMP_NUM_THREADS'] = str(cpu)

        # log command
        self.__log.info(' '.join(args))
        self.__log.info("cpu: %s" % cpu)

        # run process
        process = subprocess.Popen([self.executable] + args,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)

        # stream output as it gets generated
        while True:
            line = process.stdout.readline().decode("utf-8").strip()
            self.__log.debug(line)
            if line == '' and process.poll() is not None:
                break
