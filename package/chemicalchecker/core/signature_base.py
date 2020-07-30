"""Implementation of the abstract signature class.

Each signature class derived from this base class will have to implement the
`fit`, `predict` and `validate` methods. As the underlying data format for
every signature is the same, this class implements the iterator and attribute
getter.
Also implements the signature status, and persistence of parameters.
"""
import os
import six
import sys
import h5py
import json
import pickle
import tempfile
import numpy as np
from tqdm import tqdm
from datetime import datetime
from bisect import bisect_left
from abc import ABCMeta, abstractmethod

from chemicalchecker.util.hpc import HPC
from chemicalchecker.util import Config
from chemicalchecker.util import logged
from chemicalchecker.util.plot import Plot


class cached_property(object):
    """
    Decorator for properties calculated/stored on-demand on first use.
    NS: With this, the call to a class function becomes an attribute ie: myobject.function and NOT myobject.function()
    NS: It is calculated when the function is defined and stored as a property
    """

    def __init__(self, func):
        self._attr_name = func.__name__           # grabs the name of the decorated func
        self._func = func

    def __get__(self, instance, owner):           
        attr = self._func(instance)                #execute the method (bizarre to have instance inside parenthesis)
        setattr(instance, self._attr_name, attr)   #setattr(object, fctname-->f(object))
        return attr                              # returns f(object)


@logged
@six.add_metaclass(ABCMeta)
class BaseSignature(object):
    """A Signature base class.

    Implements methods and checks common to all signatures.
    """

    @abstractmethod
    def __init__(self, signature_path, dataset, **params):
        """Initialize or load the signature at the given path."""
        self.dataset = dataset
        self.cctype = signature_path.split("/")[-1]
        self.molset = signature_path.split("/")[-5]
        self.signature_path = os.path.abspath(signature_path)
        if sys.version_info[0] == 2:
            if isinstance(self.signature_path, unicode):
                self.signature_path = self.signature_path.encode('ascii',
                                                                 'ignore')
        self.readyfile = "fit.ready"

        if not os.path.isdir(self.signature_path):
            BaseSignature.__log.info(
                "Initializing new signature in: %s" % self.signature_path)
            original_umask = os.umask(0)
            os.makedirs(self.signature_path, 0o775)
            os.umask(original_umask)

        self.model_path = os.path.join(self.signature_path, "models")
        if not os.path.isdir(self.model_path):
            BaseSignature.__log.info(
                "Creating model_path in: %s" % self.model_path)
            original_umask = os.umask(0)
            os.makedirs(self.model_path, 0o775)
            os.umask(original_umask)
        self.stats_path = os.path.join(self.signature_path, "stats")
        if not os.path.isdir(self.stats_path):
            BaseSignature.__log.info(
                "Creating stats_path in: %s" % self.stats_path)
            original_umask = os.umask(0)
            os.makedirs(self.stats_path, 0o775)
            os.umask(original_umask)
        self.diags_path = os.path.join(self.signature_path, "diags")
        if not os.path.isdir(self.diags_path):
            BaseSignature.__log.info(
                "Creating diags_path in: %s" % self.diags_path)
            original_umask = os.umask(0)
            os.makedirs(self.diags_path, 0o775)
            os.umask(original_umask)

    @abstractmethod
    def fit(self):
        """Take an input and learns to produce an output."""
        BaseSignature.__log.debug('fit')
        if os.path.isdir(self.model_path):
            BaseSignature.__log.warning("Model already available.")
        if os.path.exists(os.path.join(self.model_path, self.readyfile)):
            os.remove(os.path.join(self.model_path, self.readyfile))

    def func_hpc(self, func_name, *args, **kwargs):
        """Execute the *any* method on the configured HPC.

        Args:
            args(tuple): the arguments for of the fit method
            kwargs(dict): arguments for the HPC method.
        """
        # read config file
        cc_config = kwargs.get("cc_config", os.environ['CC_CONFIG'])
        cfg = Config(cc_config)
        # create job directory if not available
        job_base_path = cfg.PATH.CC_TMP
        tmp_dir = tempfile.mktemp(prefix='tmp_', dir=job_base_path)
        job_path = kwargs.get("job_path", tmp_dir)
        if not os.path.isdir(job_path):
            os.mkdir(job_path)
        # check cpus
        cpu = kwargs.get("cpu", 1)
        # create script file
        script_lines = [
            "import os, sys",
            "os.environ['OMP_NUM_THREADS'] = str(%s)" % cpu,
            "import pickle",
            "sign, args = pickle.load(open(sys.argv[1], 'rb'))",
            "sign.%s(*args)" % func_name,
            "print('JOB DONE')"
        ]
        if kwargs.get("delete_job_path", False):
            script_lines.append("print('DELETING JOB PATH: %s')" % job_path)
            script_lines.append("os.system('rm -rf %s')" % job_path)
        script_name = '%s_%s_hpc.py' % (self.__class__.__name__, func_name)
        script_path = os.path.join(job_path, script_name)
        with open(script_path, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # pickle self and fit args
        pickle_file = '%s_%s_hpc.pkl' % (self.__class__.__name__, func_name)
        pickle_path = os.path.join(job_path, pickle_file)
        pickle.dump((self, args), open(pickle_path, 'wb'))
        # hpc parameters
        params = kwargs
        params["num_jobs"] = 1
        params["jobdir"] = job_path
        params["job_name"] = script_name
        params["wait"] = False
        # job command
        singularity_image = cfg.PATH.SINGULARITY_IMAGE
        command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={}" +\
            " singularity exec {} python {} {}"
        command = command.format(
            os.path.join(cfg.PATH.CC_REPO, 'package'), cc_config,
            singularity_image, script_name, pickle_file)
        # submit jobs
        cluster = HPC.from_config(Config())
        cluster.submitMultiJob(command, **params)
        return cluster

    def fit_hpc(self, *args, **kwargs):
        """Execute the fit method on the configured HPC.

        Args:
            args(tuple): the arguments for of the fit method
            kwargs(dict): arguments for the HPC method.
        """
        return self.func_hpc("fit", *args, **kwargs)

    @abstractmethod
    def predict(self):
        """Use the fitted models to go from input to output."""
        BaseSignature.__log.debug('predict')
        if not os.path.isdir(self.model_path):
            raise Exception("Model file not available.")

        if not self.is_fit():
            raise Exception(
                "Before calling predict method, fit method needs to be called.")

    def subsample(self, n):
        """Subsample from a signature without replacement.

            Args:
               n(int): Maximum number of samples (default=10000).

            Returns:
               V(matrix): A (samples, features) matrix.
               keys(array): The list of keys.
        """
        self.__log.debug("Subsampling dataset (n=%d)" % n)
        if n > len(self.keys):
            V = self[:]
            keys = self.keys
        else:
            idxs = np.array(sorted(np.random.choice(
                len(self.keys), n, replace=False)))
            with h5py.File(self.data_path, "r") as hf:
                V = hf["V"][idxs]
            keys = np.array(self.keys)[idxs]
        return V, keys

    def validate_versus_signature(self, sign, n_samples=1000, n_neighbors=5, apply_mappings=True, metric='cosine'):
        """Perform validations.


        Args:
            sign(signature object): A CC signature object to validate against
            apply_mappings(bool): Whether to use mappings to compute
                validation. Signature which have been redundancy-reduced
                (i.e. `reference`) have fewer molecules. The key are moleules
                from the `full` signature and values are moleules from the
                `reference` set.
        """
        from sklearn.neighbors import NearestNeighbors
        # from sklearn.metrics import roc_curve, auc
        import random
        # check if we apply mapping (i.e. the signature is a 'reference')
        if apply_mappings:
            if 'mappings' not in self.info_h5:
                self.__log.warning("Cannot apply mappings in validation.")
                inchikey_mappings = None
            else:
                inchikey_mappings = dict(self.mappings)
        else:
            inchikey_mappings = None
        plot = Plot(self.dataset, self.stats_path, None)
        cctype = self.__class__.__name__
        # select pairs
        # first look for the intersection of inchikeys
        vs_keys = sign.keys
        if inchikey_mappings is None:
            my_keys = self.keys
        else:
            my_keys = [inchikey_mappings[k] for k in self.keys]
        # consider connectivity layer to increase converage
        vs_conn_set = set([ik.split("-")[0] for ik in vs_keys])
        my_conn_set = set([ik.split("-")[0] for ik in my_keys])
        common_conn = my_conn_set.intersection(vs_conn_set)

        frac_shared = 100 * len(common_conn) / float(len(my_conn_set))

        if n_samples < len(common_conn):
            common_conn = set(random.sample(list(common_conn), n_samples))
        # extract matrices
        vs_keys_conn = dict(
            (ik.split("-")[0], ik) for ik in vs_keys if ik.split("-")[0] in common_conn)
        my_keys_conn = dict(
            (ik.split("-")[0], ik) for ik in my_keys if ik.split("-")[0] in common_conn)
        common_conn = sorted(common_conn)
        vs_vectors = sign.get_vectors(
            [vs_keys_conn[c] for c in common_conn])[1]
        my_vectors = self.get_vectors(
            [my_keys_conn[c] for c in common_conn])[1]
        # do nearest neighbors
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn.fit(vs_vectors)
        neighs = nn.kneighbors(vs_vectors)[1][:, 1:]
        # sample positive and negative pairs
        pos_pairs = set()
        neg_pairs = set()
        for i in range(0, len(neighs)):
            for j in neighs[i]:
                pair = [common_conn[i], common_conn[j]]
                pair = sorted(pair)
                pos_pairs.update([(pair[0], pair[1])])
        for _ in range(0, len(pos_pairs) * 10):
            pair = random.sample(common_conn, 2)
            pair = sorted(pair)
            pair = (pair[0], pair[1])
            if pair in pos_pairs:
                continue
            neg_pairs.update([pair])
            if len(neg_pairs) > len(pos_pairs):
                break
        # do distances
        if metric == "cosine":
            from scipy.spatial.distance import cosine as metric
        if metric == "euclidean":
            from scipy.spatial.distance import euclidean as metric
        y_t = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))
        pairs = list(pos_pairs) + list(neg_pairs)
        y_p = []
        for pair in pairs:
            idx1 = bisect_left(common_conn, pair[0])
            idx2 = bisect_left(common_conn, pair[1])
            y_p += [metric(my_vectors[idx1], my_vectors[idx2])]
        # convert to similarity-respected order
        y_p = -np.abs(np.array(y_p))

        plot.roc_curve_plot(y_t, y_p, cctype, sign.dataset,
                            len(common_conn), frac_shared)

    def validate(self, apply_mappings=True, metric='cosine'):
        """Perform validations.

        A validation file is an external resource basically presenting pairs of
        molecules and whether they share or not a given property (i.e the file
        format is inchikey inchikey 0/1).
        Current test are performed on MOA (Mode Of Action) and ATC (Anatomical
        Therapeutic Chemical) corresponding to B1.001 and E1.001 dataset.

        Args:
            apply_mappings(bool): Whether to use mappings to compute
                validation. Signature which have been redundancy-reduced
                (i.e. `reference`) have fewer molecules. The key are moleules
                from the `full` signature and values are moleules from the
                `reference` set.
        """
        # check if we apply mapping (i.e. the signature is a 'reference')
        if apply_mappings:
            if 'mappings' not in self.info_h5:
                self.__log.warning("Cannot apply mappings in validation.")
                inchikey_mappings = None
            else:
                inchikey_mappings = dict(self.mappings)
        else:
            inchikey_mappings = None

        stats = {"molecules": len(self.keys)}
        results = dict()
        validation_path = self.signature_path + \
            '/../../../../../tests/validation_sets/'
        if not os.path.exists(validation_path):
            self.__log.warn("Standard validation path does not exist, taking validations from examples")
            validation_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "examples/validation_sets/")
        validation_files = os.listdir(validation_path)
        self.__log.info(validation_path)
        plot = Plot(self.dataset, self.stats_path, validation_path)
        if len(validation_files) == 0:
            raise Exception("Validation dir %s is empty." % validation_path)
        for validation_file in validation_files:
            vset = validation_file.split('_')[0]
            cctype = self.__class__.__name__
            res = plot.vector_validation(self, cctype, prefix=vset,
                                         mappings=inchikey_mappings, distance=metric)
            results[vset] = res
            stats.update({
                "%s_ks_d" % vset: res[0][0],
                "%s_ks_p" % vset: res[0][1],
                "%s_auc" % vset: res[1],
                "%s_cov" % vset: res[2],
            })

        validation_stat_file = os.path.join(
            self.stats_path, 'validation_stats.json')
        with open(validation_stat_file, 'w') as fp:
            json.dump(stats, fp)
        plot.matrix_plot(self.data_path)
        return results

    def mark_ready(self):
        filename = os.path.join(self.model_path, self.readyfile)
        with open(filename, 'w') as fh:
            pass

    def is_fit(self):
        """The fit method was already called for this signature."""
        if os.path.exists(os.path.join(self.model_path, self.readyfile)):
            return True
        else:
            return False

    def available(self):
        """This signature data is available."""
        if os.path.isfile(self.data_path):
            return True
        else:
            return False

    def consistency_check(self):
        """Check that signature is valid."""
        if os.path.isfile(self.data_path):
            # check that keys are unique
            if len(self.keys) != len(self.unique_keys):
                raise Exception("Inconsistent: keys are not unique.")
            # check that amout of keys is same as amount of signatures
            with h5py.File(self.data_path, 'r') as hf:
                nr_signatures = hf['V'].shape[0]
            if len(self.keys) > nr_signatures:
                raise Exception("Inconsistent: more Keys than signatures.")
            if len(self.keys) < nr_signatures:
                raise Exception("Inconsistent: more signatures than Keys.")
            # check that keys are sorted
            if not np.all(self.keys[:-1] <= self.keys[1:]):
                raise Exception("Inconsistent: Keys are not sorted.")

    def map(self, out_file):
        """Map signature throught mappings."""
        if "mappings" not in self.info_h5:
            raise Exception("Data file has no mappings.")
        with h5py.File(self.data_path, 'r') as hf:
            mappings = dict(hf['mappings'][:])
        # avoid trivial mappings (where key==value)
        to_map = set(mappings.keys()) - set(mappings.values())
        if len(to_map) == 0:
            # corner case where there's nothing to map
            with h5py.File(self.data_path, 'r') as hf:
                src_keys = hf['keys'][:]
                src_vectors = hf['V'][:]
            with h5py.File(out_file, "w") as hf:
                hf.create_dataset('keys', data=src_keys)
                hf.create_dataset('V', data=src_vectors, dtype=np.float32)
                hf.create_dataset("shape", data=src_vectors.shape)
                hf.create_dataset(
                    "date", data=[datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            return
        # prepare key-vector arrays
        dst_keys = list()
        dst_vectors = list()
        for dst_key in sorted(to_map):
            dst_keys.append(dst_key)
            dst_vectors.append(self[mappings[dst_key]])
        # to numpy arrays
        dst_keys = np.array(dst_keys)
        matrix = np.vstack(dst_vectors)
        # join with current key-signatures
        with h5py.File(self.data_path, 'r') as hf:
            src_vectors = hf['V'][:]
        dst_keys = np.concatenate((dst_keys, self.keys))
        matrix = np.concatenate((matrix, src_vectors))
        # get them sorted
        sorted_idx = np.argsort(dst_keys)
        with h5py.File(out_file, "w") as hf:
            hf.create_dataset('keys', data=dst_keys[sorted_idx])
            hf.create_dataset('V', data=matrix[sorted_idx], dtype=np.float32)
            hf.create_dataset("shape", data=matrix.shape)
            hf.create_dataset(
                "date", data=[datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    def __repr__(self):
        """String representig the signature."""
        return self.data_path

    def generator_fn(self, batch_size=None):
        """Return the generator function that we can query for batches."""
        hf = h5py.File(self.data_path, 'r')
        dset = hf['V']
        total = dset.shape[0]
        if not batch_size:
            batch_size = total

        def _generator_fn():
            beg_idx, end_idx = 0, batch_size
            while True:
                if beg_idx >= total:
                    self.__log.debug("EPOCH completed")
                    beg_idx = 0
                    return
                yield dset[beg_idx: end_idx]
                beg_idx, end_idx = beg_idx + batch_size, end_idx + batch_size

        return _generator_fn

    def get_non_redundant_intersection(self, sign):
        """Return the non redundant intersection between two signatures.

        (i.e. keys and vectors that are common to both signatures.)
        N.B: to maximize overlap it's better to use signatures of type 'full'.
        N.B: Near duplicates are found in the first signature.
        """
        from chemicalchecker.util.remove_near_duplicates import RNDuplicates
        shared_keys = self.unique_keys.intersection(sign.unique_keys)
        self.__log.debug("%s shared keys.", len(shared_keys))
        _, self_matrix = self.get_vectors(shared_keys)
        rnd = RNDuplicates()
        nr_keys, nr_matrix, mappings = rnd.remove(
            self_matrix, keys=list(shared_keys))
        a, self_matrix = self.get_vectors(nr_keys)
        b, sign_matrix = sign.get_vectors(nr_keys)
        assert(all(a == b))
        return a, self_matrix, sign_matrix

    def get_intersection(self, sign):
        """Return the intersection between two signatures."""
        shared_keys = self.unique_keys.intersection(sign.unique_keys)
        a, self_matrix = self.get_vectors(shared_keys)
        b, sign_matrix = sign.get_vectors(shared_keys)
        return a, self_matrix, sign_matrix

    def to_csv(self, filename, smiles=None):
        """Write smiles to h5.

        At the moment this is done quering the `Structure` table for inchikey
        inchi mapping and then converting via `Converter`.
        """
        from chemicalchecker.database import Molecule
        from chemicalchecker.util.parser import Converter
        if not smiles:
            # fetch inchi
            ink_inchi = Molecule.get_inchikey_inchi_mapping(self.keys)
            if len(ink_inchi) != len(self.keys):
                raise Exception(
                    "Not same number of inchi found for given keys!")
            # convert inchi to smiles (sorted)
            converter = Converter()
            smiles = list()
            for ink in tqdm(self.keys):
                smiles.append(converter.inchi_to_smiles(ink_inchi[ink]))
            if len(smiles) != len(self.keys):
                raise Exception(
                    "Not same number of smiles converted for given keys!")
        # write to disk
        with open(filename, 'w') as fh:
            header = ['c%s' % c for c in range(128)] + ['smiles']
            fh.write(','.join(header) + '\n')
            for chunk in tqdm(self.chunker()):
                V_chunk = self[chunk]
                smiles_chunk = smiles[chunk]
                for comps, sml in zip(V_chunk, smiles_chunk):
                    fh.write(','.join(comps.astype(str)))
                    fh.write(',%s\n' % sml)

    def get_molset(self, molset):
        '''Return a signature from a different molset'''
        folds = self.signature_path.split('/')
        folds[-5] = molset
        new_path = '/'.join(folds)
        newsign = self.__class__(new_path, self.dataset)
        return newsign

    def get_neig(self):
        '''Return the neighbors signature, given a signature'''
        from .neig import neig
        folds = self.signature_path.split('/')
        folds[-1] = "neig%s" % folds[-1][-1]
        new_path = "/".join(folds)
        return neig(new_path, self.dataset)

    def get_cc(self):
        '''Return the CC where the signature is present'''
        from chemicalchecker import ChemicalChecker
        cc_path = "/".join(self.signature_path.split("/")[:-5])
        return ChemicalChecker(cc_path)

    def get_sign(self, sign_type):
        '''Return the signature type for current dataset'''
        from .sign0 import sign0
        from .sign1 import sign1
        from .sign2 import sign2
        from .sign3 import sign3
        from .sign4 import sign4
        if sign_type not in ['sign%i' % i for i in range(5)]:
            raise ValueError('Wrong signature type: %s' % sign_type)
        folds = self.signature_path.split('/')
        folds[-1] = sign_type
        new_path = "/".join(folds)
        return eval(sign_type)(new_path, self.dataset)

    def remove_redundancy(self, cpu=2, overwrite=False):
        '''Remove redundancy of a signature (it generates) new data in the references folders.
        Only allowed for sign* (*not* sign1 or sign2).
    
        Args:
            cpu(int): Number of CPUs (default=2),
            overwrite(bool): Overwrite existing (default=False).
        '''
        if "sign" not in self.cctype:
            raise Exception("Only sign* are allowed")
        if self.cctype == "sign1" or self.cctype == "sign2":
            raise Exception("Not allowed for sign1 or sign2")
        from chemicalchecker.util.remove_near_duplicates import RNDuplicates
        rnd = RNDuplicates(cpu=cpu)
        sign_ref = self.get_molset("reference")
        if os.path.exists(sign_ref.data_path):
            if not overwrite:
                raise Exception("%s exists" % sign_ref.data_path)
        rnd.remove(self.data_path, save_dest=sign_ref.data_path)
        f5 = h5py.File(self.data_path, "r")
        if 'features' in f5.keys():
            features = f5['features'][:]
            f5.close()
            with h5py.File(sign_ref.data_path, 'a') as hf:
                hf.create_dataset('features', data=features)
        return sign_ref
