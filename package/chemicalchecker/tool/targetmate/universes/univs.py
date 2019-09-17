"""Universe of molecules to be used for the sampling of negative data.

For now, only the CC universe is accepted. This is convenient, since for the CC universe we have all signatures
pre-computed.
"""
import os
import uuid
import pickle
import random
import numpy as np
import collections

from sklearn.svm import OneClassSVM
from sklearn.cluster import MiniBatchKMeans

from ..utils.chemistry import maccs_matrix, morgan_arena, load_morgan_arena
from ..utils.chemistry import similarity_matrix, read_smiles
from ..datasets.chembl import ChemblDb

from chemicalchecker.util import logged


class UniverseLoader(pickle.Unpickler):

    def find_class(self, module, name):
        if name == "Universe":
            from targetmate import Universe
            return Universe
        return super().find_class(module, name)


def load_universe(model_path):
    file_name = os.path.join(os.path.abspath(model_path), "universe.pkl")
    with open(file_name, "rb") as f:
        univ = UniverseLoader(f).load()
    return univ


@logged
class Universe:

    def __init__(self, molrepo='chembl', standardize=True, k=None,
                 model_path='/tmp/tm/universe',
                 tmp_path='/tmp/tm/tmp_universe',
                 min_actives_oneclass=10, max_actives_oneclass=1000,
                 representative_mols_per_cluster=10,
                 max_universe_size=None,
                 trials = 1000000):
        """Initialize the Universe class.

        Args:
            molrepo(str): A molrepo name.
            k(int): Number of partitions for the k-Means clustering
                (default=sqrt(N/2)).
            model_path(str): Folder where the universe should be stored
                (default = .)
            tmp_path(str): Temporary directory (default=/tmp/tm/tmp_universe).
            inactives_per_active(int): Number of inactives to sample for each
                active (default=100).
            min_actives_oneclass(int): Minimum number of actives to use in the
                OneClassSVM (default=10).
            max_actives_oneclass(int): Maximum number of actives to use in the
                OneClassSVM (default=1000).
            representative_mols_per_cluster(int): Number of molecules to
                samples for each cluster (default=10).
            max_universe_size(int): Maximum number of molecules in the
                universe. If not specified, all are considered (default=None).
            trials(int): Number of sampling trials before stop trying (default=1000000)
       """
        self.smiles_argument = smiles
        self.standardize = standardize
        self.k = k
        self.tmp_path = os.path.abspath(tmp_path)
        if not os.path.isdir(self.tmp_path):
            os.makedirs(self.tmp_path)
        self.model_path = os.path.abspath(model_path)
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        self.min_actives_oneclass = min_actives_oneclass
        self.max_actives_oneclass = max_actives_oneclass
        self.representative_mols_per_cluster = representative_mols_per_cluster
        self.max_universe_size = max_universe_size
        self.trials = trials

    def save(self):
        pkl_file = os.path.join(self.model_path, "universe.pkl")
        with open(pkl_file, "wb") as f:
            pickle.dump(self, f)

    def cluster(self):
        maccs = maccs_matrix(self.smiles)
        if not self.k:
            self.k = int(np.sqrt(maccs.shape[0] / 2)) + 1
        kmeans = MiniBatchKMeans(
            n_clusters=self.k, init_size=np.max([3 * self.k, 300]))
        kmeans.fit(maccs)
        clusters = kmeans.predict(maccs)
        self.clusters_dict = collections.defaultdict(list)
        for i, c in enumerate(clusters):
            self.clusters_dict[c] += [i]
        self.representative_smiles = []
        for c, idxs in self.clusters_dict.items():
            idxs_ = random.choices(
                idxs, k=self.representative_mols_per_cluster)
            for i in idxs_:
                self.representative_smiles += [(c, self.smiles[i])]

    def calculate_arena(self):
        fps_file = os.path.join(self.model_path, "arena.h5")
        morgan_arena([smi[1][0]
                      for smi in self.representative_smiles], fps_file)
        self.arena_file = fps_file

    def fit_oneclass_svm(self, actives):
        tag = str(uuid.uuid4())
        actives_list = list(actives)
        if len(actives_list) < self.min_actives_oneclass:
            actives_list = random.choices(
                actives_list, k=self.min_actives_oneclass)
        else:
            actives_list = actives_list[:self.max_actives_oneclass]
        actives_list_smiles = [smi[0] for smi in actives_list]
        fps_file = os.path.join(self.tmp_path, tag + "_actives_arena.h5")
        actives_arena = morgan_arena(actives_list_smiles, fps_file)
        sim_mat = similarity_matrix(
            actives_list_smiles, actives_arena, len(actives_list_smiles))
        clf = OneClassSVM(kernel="precomputed")
        clf.fit(sim_mat)
        os.remove(fps_file)
        return clf, actives_list_smiles

    def fit(self):
        self.__log.info("Processing molecules")
        self.process_smiles()
        self.__log.info("Clustering")
        self.cluster()
        self.__log.info("Calculating arena")
        self.calculate_arena()

    def predict(self, actives, inactives, inactives_per_active=100,
                min_actives=10):
        """
        Args:
            actives(list or set): Should include (smiles, id, inchikey).
            inactives(list or set): Should include (smiles, id, inchkey).
            inactives_per_active(int): Number of inactives to sample from the
                universe. Can be None (default=100).
            min_actives(int): Minimum number of actives (default=10).
        """
        self.__log.info("Sampling candidate inactives")
        common_iks = set([smi[-1] for smi in actives]
                         ).intersection([smi[-1] for smi in inactives])
        actives = set([smi for smi in actives if smi[-1] not in common_iks])
        if len(actives) < min_actives:
            return None
        inactives = set(
            [smi for smi in inactives if smi[-1] not in common_iks])
        if not inactives_per_active:
            return actives, inactives, set()
        # Inchikeys
        actives_iks = set([smi[-1] for smi in actives])
        inactives_iks = set([smi[-1] for smi in inactives])
        # Inactives sampling procedure
        N = int(len(actives) * inactives_per_active) + 1
        if len(inactives) >= N:
            return actives, random.sample(inactives, N), set()
        N = N - len(inactives)
        # Fitting one-class SVM
        clf, actives_list_smiles = self.fit_oneclass_svm(actives)
        # Loading universe fingerprint arena
        arena = load_morgan_arena(self.arena_file)
        # Calculating similarity matrix
        sim_mat = similarity_matrix(
            actives_list_smiles, arena, len(self.representative_smiles))
        sim_mat = sim_mat.T
        # Predicting using one-class SVM
        prd = clf.predict(sim_mat)
        # Assigning weights to clusters
        weight_dict = collections.defaultdict(int)
        for i, p in enumerate(prd):
            if p == -1:
                weight_dict[self.representative_smiles[i][0]] += 1
        candidate_clusters = sorted(weight_dict.keys())
        candidate_weights = [weight_dict[k] for k in candidate_clusters]
        # Sample from candidates
        trials = self.trials
        t = 0
        candidates = set()
        while len(candidates) < N and t < trials:
            c = random.choices(candidate_clusters, k=1,
                               weights=candidate_weights)[0]
            i = random.choice(self.clusters_dict[c])
            cand = self.smiles[i]
            if cand[-1] not in actives_iks and cand[-1] not in inactives_iks:
                candidates.update([cand])
            t += 1
        if len(candidates) < N:
            all_iks = actives_iks.union(inactives_iks).union(
                [cc[-1] for cc in candidates])
            remaining_universe = [
                smi for smi in self.smiles if smi[-1] not in all_iks]
            N = N - len(candidates)
            if N >= len(remaining_universe):
                candidates.update(remaining_universe)
            else:
                candidates.update(random.sample(remaining_universe, k=N))
        return actives, inactives, candidates


if __name__ == "__main__":
    universe = Universe(max_universe_size=100)
    universe.fit()
    universe.save()
