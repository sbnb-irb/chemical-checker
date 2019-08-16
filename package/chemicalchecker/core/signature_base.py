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
    """Decorator for properties calculated/stored on-demand on first use."""

    def __init__(self, func):
        self._attr_name = func.__name__
        self._func = func

    def __get__(self, instance, owner):
        attr = self._func(instance)
        setattr(instance, self._attr_name, attr)
        return attr


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
        script_name = '%s_%s_hpc.py' % (self.__class__.__name__, func_name)
        script_path = os.path.join(job_path, script_name)
        with open(script_path, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # pickle self and fit args
        pickle_file = '%s_%s_hpc.pkl' % (self.__class__.__name__, func_name)
        pickle_path = os.path.join(job_path, pickle_file)
        pickle.dump((self, args), open(pickle_path, 'w'))
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
        cluster = HPC(Config())
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

    def validate(self, apply_mappings=True):
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
                inchikey_mappings = dict(self.get_h5_dataset('mappings'))
        else:
            inchikey_mappings = None
        plot = Plot(self.dataset, self.stats_path)
        stats = {"molecules": len(self.keys)}
        results = dict()
        validation_path = Config().PATH.validation_path
        validation_files = os.listdir(validation_path)
        if len(validation_files) == 0:
            raise Exception("Validation dir %s is empty." % validation_path)
        for validation_file in validation_files:
            vset = validation_file.split('_')[0]
            cctype = self.__class__.__name__
            res = plot.vector_validation(self, cctype, prefix=vset,
                                         mappings=inchikey_mappings)
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

    def chunker(self, size=2000):
        """Iterate on signatures."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        for i in range(0, self.shape[0], size):
            yield slice(i, i + size)

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

    @property
    def shape(self):
        """Get the V matrix sizes."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        with h5py.File(self.data_path, 'r') as hf:
            if 'shape' not in hf.keys():
                self.__log.warn("HDF5 file has no 'shape' dataset.")
                return hf['V'].shape
            return hf['shape'][:]

    @cached_property
    def features(self):
        """Get the list of keys in the signature."""
        if not os.path.isfile(self.data_path):
            raise Exception("Data file %s not available." % self.data_path)
        with h5py.File(self.data_path, 'r') as hf:
            if 'features' not in hf.keys():
                self.__log.warn("No features available for this signature!")
                return None
            return hf['features'][:]

    # @cached_property
    # def keys(self):
    #     keys_name = 'keys'
    #     """Get the list of keys (usually inchikeys) in the signature."""
    #     if not os.path.isfile(self.data_path):
    #         raise Exception("Data file %s not available." % self.data_path)
    #     with h5py.File(self.data_path, 'r') as hf:
    #         if 'keys' not in hf.keys():
    #             if 'row_keys' in hf.keys():
    #                 keys_name = 'row_keys'
    #             else:
    #                 raise Exception("HDF5 file has no 'keys' field.")
    #         # if keys have a decode attriute they have been generated in py2
    #         # for compatibility with new format we decode them
    #         if hasattr(hf[keys_name][0], 'decode'):
    #             return [k.decode() for k in hf[keys_name][:]]
    #         else:
    #             return hf[keys_name][:]

    # @cached_property
    # def unique_keys(self):
    #     """Get the keys of the signature as a set."""
    #     return set(self.keys)

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
