"""Wrapper for SNAP C++ implementation of node2vec.

An algorithmic framework for representational learning on graphs. [Aug 27 2018]
================================================================================
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
import numpy as np
import distutils.spawn
from bisect import bisect_left
from chemicalchecker.util import logged


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
        # compute weights
        return (-np.log10(pvalues) - self.min_pvalue) * self.range_weight / \
            self.range_pvalue + self.min_weight

    def emb_to_h5(self, sign1, in_file, out_file):
        """Convert from native node2vec format to familiar h5.

        We need to map back from sign1 ids to inchikeys and sort.
        """
        self.__log.info("Converting %s to %s" % (in_file, out_file))
        with open(in_file, 'r') as fh:
            words = list()
            vectors = list()
            fh.next()  # skip first row
            skipped = 0
            for line in fh:
                fields = line.split()
                # first colum is id
                word = int(fields[0])
                # then embedding
                vector = np.fromiter((float(x) for x in fields[1:]),
                                     dtype=np.float)
                words.append(word)
                vectors.append(vector)
        # to numpy arrays
        words = np.array(words)
        matrix = np.array(vectors)
        # get them sorted
        sorted_idx = np.argsort(words)
        self.__log.info('words: %s' % str(words.shape))
        self.__log.info('matrix: %s' % str(matrix.shape))
        self.__log.info('skipped: %s' % skipped)
        sign1 = h5py.File(sign1, 'r')
        names = sign1['inchikeys'][:]
        with h5py.File(out_file, "w") as fh:
            fh.create_dataset('inchikeys', data=names[words[sorted_idx]])
            fh.create_dataset('V', data=matrix[sorted_idx])

    def to_edgelist(self, neig1, sign1, out_file, **params):
        """Convert Nearest-neighbor to an edgelist.

        Args:
            neig1(str): Nearest neighbors 1, h5 file path.
            sign1(str): Signature 1, h5 file path.
            out_file(str): Destination file.
            params(dict): Parameters defining similarity network.

        N.B.
        neig1 must have:
            'indeces': for each molecule indeces of 1000 closest molecules.
            'distances': for each molecule distances of 1000 closest molecules.
            'keys': inchikeys position in array is the index.
        """
        self.min_weight = params.get("min_weight", 1e-2)
        self.max_weight = params.get("max_weight", 1.0)
        self.min_pvalue = params.get("min_pvalue", 1e-2)
        self.max_pvalue = params.get("max_pvalue", 1e-6)
        self.min_degree = params.get("min_degree", 3)
        self.range_pvalue = self.max_pvalue - self.min_pvalue
        self.range_weight = self.max_weight - self.min_weight
        # open h5s
        neig1 = h5py.File(neig1, 'r')
        sign1 = h5py.File(sign1, 'r')
        thresholds = sign1.background_distances("cosine")
        # derive max_degree
        nr_nodes = neig1['keys'].shape[0]
        self.max_degree = Node2Vec.heuristic_max_degree(nr_nodes)
        with open(out_file, 'w') as fh:
            for src_idx in range(nr_nodes):
                neig_indices = neig1['indices'][src_idx]
                # FIXME nearest neighbor return doubled distances, why?
                neig_distances = neig1['distances'][src_idx] / 2.
                edges = self._get_edges(
                    neig_distances, neig_indices, thresholds)
                for dst_idx, weight in edges:
                    # exclude self loops
                    if dst_idx == src_idx:
                        continue
                    fh.write("%i %i %.4f\n" % (src_idx, dst_idx, weight))
        # close h5s
        neig1.close()
        sign1.close()

    def _get_edges(self, neig_distances, neig_indices, thresholds):
        """Get a molecule neighbors and edge weight.

        We have a list of all possible pvalues and the pvalue id for each
        similar molecule. We want as much as possible significant pvalues
        (below MIN_PVALUE, equal or higher than MAX_PVALUE). We want at least
        MIN_DEGREE similar molecules (even if less significant than MIN_PVALUE)
        and less than MAX_DEGREE (even if kicking out significant ones).
        Returns the edges (tuple with destination, source, and weight)

        Args:
            mol_ink(str): The name of the molecule.
            coordinate(str): The coordinate of the square.
            sim_mol(array): Array of similar molecules (pvalues ids)
        Returns:
            edges(list(tuple)): list with with destination and weight.
                indexes are converted from local square to global ones.
        """
        # first check if the MIN_PVALUE is a sufficient condition (base case)
        thr_pvalues = thresholds['pvalues']
        thr_distances = thresholds['distances']
        curr_pval_idx = bisect_left(thr_pvalues, self.min_pvalue) + 1
        curr_pval = thr_pvalues[curr_pval_idx]
        neig_mask = neig_distances <= thr_distances[curr_pval_idx]
        degree = np.count_nonzero(neig_mask)
        print("curr_pval[%i]" %
              curr_pval_idx, curr_pval, "degree", degree)
        # we want more than MIN_DEGREE and less than MAX_DEGREE
        if self.min_degree <= degree <= self.max_degree:
            considered_mol_ids = neig_indices[neig_mask]
        # if we have too few
        elif degree < self.min_degree:
            # iterate increasing threshold pvalue
            for incr_pval_idx in range(curr_pval_idx + 1, len(thr_distances)):
                # check rank degree
                incr_pval = thr_pvalues[incr_pval_idx]
                neig_mask = neig_distances <= thr_distances[incr_pval_idx]
                degree = np.count_nonzero(neig_mask)
                print("incr_pval[%i]" %
                      incr_pval_idx, incr_pval, "degree", degree)
                # increasing threshold we have too many
                if degree > self.min_degree:
                    # sample neighbors from previous threshold
                    to_add = self.min_degree - degree
                    print("sampling", to_add)
                    prev_neig_mask = np.logical_and(
                        neig_distances <= thr_distances[incr_pval_idx],
                        neig_distances > thr_distances[incr_pval_idx - 1])
                    sampled_idxs = np.random.choice(
                        np.argwhere(prev_neig_mask).flatten(),
                        to_add, replace=False)
                    sampled_neig_mask = np.full(neig_mask.shape, False)
                    sampled_neig_mask[sampled_idxs] = True
                    # add neighbors within current threshold
                    neig_mask = np.logical_or(neig_mask, sampled_neig_mask)
                    considered_mol_ids = neig_indices[neig_mask]
                    break
                # corner case where the highest threshold has to be sampled
                elif incr_pval_idx == len(thr_distances):
                    # sample from highest threshold
                    to_add = self.min_degree - degree
                    print("sampling", to_add)
                    curr_neig_mask = neig_distances == thr_distances[
                        incr_pval_idx]
                    sampled_idxs = np.random.choice(
                        np.argwhere(curr_neig_mask).flatten(),
                        to_add, replace=False)
                    sampled_neig_mask = np.full(neig_mask.shape, False)
                    sampled_neig_mask[sampled_idxs] = True
                    # add neighbors within previous threshold
                    prev_neig_mask = neig_distances <= thr_distances[
                        incr_pval_idx - 1]
                    neig_mask = np.logical_or(
                        prev_neig_mask, sampled_neig_mask)
                    considered_mol_ids = neig_indices[neig_mask]
                    break
        # else too many
        else:
            # iterate decreasing threshold pvalue
            for decr_pval_idx in range(curr_pval_idx - 1, -1, -1):
                # check degree
                decr_pval = thr_pvalues[decr_pval_idx]
                neig_mask = neig_distances <= thr_distances[decr_pval_idx]
                degree = np.count_nonzero(neig_mask)
                print("decr_pval[%i]" %
                      decr_pval_idx, decr_pval, "degree", degree)
                # reducing threshold we have too few
                if degree < self.max_degree:
                    # sample neighbors from previous threshold
                    to_add = self.max_degree - degree
                    print("sampling", to_add)
                    prev_neig_mask = np.logical_and(
                        neig_distances <= thr_distances[decr_pval_idx + 1],
                        neig_distances > thr_distances[decr_pval_idx])
                    sampled_idxs = np.random.choice(
                        np.argwhere(prev_neig_mask).flatten(),
                        to_add, replace=False)
                    sampled_neig_mask = np.full(neig_mask.shape, False)
                    sampled_neig_mask[sampled_idxs] = True
                    # add neighbors within current threshold
                    neig_mask = np.logical_or(neig_mask, sampled_neig_mask)
                    considered_mol_ids = neig_indices[neig_mask]
                    break
                # corner case where the lowest threshold has to be sampled
                elif decr_pval_idx == 0:
                    to_add = self.max_degree
                    print("sampling", to_add)
                    sampled_idxs = np.random.choice(
                        np.argwhere(neig_mask).flatten(),
                        to_add, replace=False)
                    sampled_neig_mask = np.full(neig_mask.shape, False)
                    sampled_neig_mask[sampled_idxs] = True
                    considered_mol_ids = neig_indices[sampled_neig_mask]
                    break
        # convert distances to p-values index
        pvalues = np.zeros(neig_distances.shape, dtype=int)
        for dist in thr_distances:
            pvalues += neig_distances > dist
        # convert p-value index to p-value
        # scale p-values linearly to weights
        weights = self._pvalue_to_weigth(pvalues[considered_mol_ids])
        # we force a minimal degree so we might have weights below MIN_WEIGHT
        # we cap to a default values one order of magnitude below MIN_WEIGHT
        weights[weights < self.min_weight] = self.min_weight / 10.
        self.__log.info("similar molecules considered:",
                        len(considered_mol_ids))
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
        dr = kwargs.get("dr", True)
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
                                   stderr=subprocess.PIPE)

        # stream output as get generated
        for line in iter(process.stdout.readline, ''):
            self.__log.info(line.strip())

        for line in iter(process.stderr.readline, ''):
            self.__log.error(line.strip())
