"""Signature base.

Each signature class inherit from this base class. They will have to implement
the ``fit`` and ``predict`` methods.

At initialization this class enforce the signature internal directory
organization:

  * **signature_path**: the signature root (e.g. ``/root/full/A/A1/A1.001/sign2/``)
  * **model_path** ``./models``: where models learned at fit time are stored
  * **stats_path** ``./stats``: where statistic are collected
  * **diags_path** ``./diags``: where diagnostics are saved

Also implements the ``validate`` function, signature status, generic HPC
functions, and provide functions to "move" in the CC (e.g. getting same
signature for different space, different molset, CC instance, etc...).
"""
import os
import sys
import h5py
import json
import shutil
import pickle
import tempfile
import datetime
import numpy as np
from tqdm import tqdm
from bisect import bisect_left
from abc import abstractmethod

from chemicalchecker.core.diagnostics import Diagnosis
from chemicalchecker.util.hpc import HPC
from chemicalchecker.util.plot import Plot
from chemicalchecker.util import Config, logged
from chemicalchecker.util.remove_near_duplicates import RNDuplicates


@logged
class BaseSignature(object):
    """BaseSignature class."""

    @abstractmethod
    def __init__(self, signature_path, dataset, **params):
        """Initialize a BaseSignature instance."""
        self.dataset = dataset
        self.cctype = signature_path.split("/")[-1]
        self.molset = signature_path.split("/")[-5]
        self.signature_path = os.path.abspath(signature_path)
        self.readyfile = "fit.ready"

        if params:
            BaseSignature.__log.debug('PARAMS:')
            for k, v in params.items():
                BaseSignature.__log.debug('\t%s\t%s', str(k), str(v))

        # permissions 775 rwx for owner and group, rx for all
        if not os.path.isdir(self.signature_path):
            BaseSignature.__log.info(
                "New signature: %s" % self.signature_path)
            original_umask = os.umask(0)
            # Ns Does doing this change the sys umask?
            os.makedirs(self.signature_path, 0o775)
            os.umask(original_umask)
        else:
            BaseSignature.__log.info(
                "Loading signature: %s" % self.signature_path)
        # Creates the 'models', 'stats', 'diags' folders if they don't exist
        self.model_path = os.path.join(self.signature_path, "models")
        if not os.path.isdir(self.model_path):
            original_umask = os.umask(0)
            os.makedirs(self.model_path, 0o775)
            os.umask(original_umask)

        self.stats_path = os.path.join(self.signature_path, "stats")
        if not os.path.isdir(self.stats_path):
            original_umask = os.umask(0)
            os.makedirs(self.stats_path, 0o775)
            os.umask(original_umask)

        self.diags_path = os.path.join(self.signature_path, "diags")
        if not os.path.isdir(self.diags_path):
            original_umask = os.umask(0)
            os.makedirs(self.diags_path, 0o775)
            os.umask(original_umask)

    @abstractmethod
    def fit(self, **kwargs):
        """Fit a model."""
        self.update_status("FIT START")
        overwrite = kwargs.get('overwrite', False)
        if overwrite and self.is_fit():
            raise Exception("Signature has already been fitted. "
                            "Delete it manually, or call the `fit` method "
                            "passing overwrite=True")
        return True

    def fit_end(self, **kwargs):
        """Conclude fit method.

        We compute background distances, run validations (including diagnostic)
        and finally marking the signature as ready.
        """
        # save background distances
        self.update_status("Background distances")
        self.background_distances("cosine")
        self.background_distances("euclidean")
        validations = kwargs.get('validations', True)
        end_other_molset = kwargs.get('end_other_molset', True)
        # performing validations
        if validations:
            self.update_status("Validation")
            self.validate()
        # Marking as ready
        self.__log.debug("Mark as ready")
        self.mark_ready()
        # end fit for signature in the other molset
        if end_other_molset:
            other_molset = 'reference'
            if self.molset == 'reference':
                other_molset == 'full'
            other_self = self.get_molset(other_molset)
            if validations:
                self.update_status("Validation %s" % other_molset)
                other_self.validate()
            other_self.mark_ready()
        self.update_status("FIT END")

    @abstractmethod
    def predict(self):
        """Use the fitted models to predict."""
        BaseSignature.__log.debug('predict')
        if not self.is_fit():
            raise Exception("Signature is not fitted, cannot predict.")
        return True

    def clear(self):
        self.__log.debug("Clearing signature")
        if os.path.exists(self.data_path):
            self.__log.debug("Removing %s" % self.data_path)
            os.remove(self.data_path)
        if os.path.exists(self.model_path):
            self.__log.debug("Removing %s" % self.model_path)
            shutil.rmtree(self.model_path)
            original_umask = os.umask(0)
            os.makedirs(self.model_path, 0o775)
            os.umask(original_umask)
        if os.path.exists(self.stats_path):
            self.__log.debug("Removing %s" % self.stats_path)
            shutil.rmtree(self.stats_path)
            original_umask = os.umask(0)
            os.makedirs(self.stats_path, 0o775)
            os.umask(original_umask)

    def validate_versus_signature(self, sign, n_samples=1000, n_neighbors=5,
                                  apply_mappings=True, metric='cosine'):
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

    def validate(self, apply_mappings=True, metric='cosine', diagnostics=True):
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
            self.__log.warn(
                "Standard validation path does not exist, "
                "taking validations from examples")
            validation_path = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), "examples/validation_sets/")
        validation_files = os.listdir(validation_path)
        self.__log.info(validation_path)
        plot = Plot(self.dataset, self.stats_path, validation_path)
        if len(validation_files) == 0:
            raise Exception("Validation dir %s is empty." % validation_path)
        for validation_file in validation_files:
            vset = validation_file.split('_')[0]
            cctype = self.__class__.__name__
            res = plot.vector_validation(
                self, cctype, prefix=vset, mappings=inchikey_mappings,
                distance=metric)
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
        # run diagnostics
        if diagnostics:
            cc = self.get_cc()
            diag = cc.diagnosis(self)
            fig = diag.canvas()
            fig.savefig(os.path.join(self.diags_path, '%s.png' % diag.name))
        return results

    def diagnosis(self, sign, **kwargs):
        cc = self.get_cc()
        return Diagnosis(cc, self, **kwargs)

    def update_status(self, status):
        fname = os.path.join(self.signature_path, '.STATUS')
        self.__log.info('STATUS: %s' % status)
        sdate = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(fname, 'a') as fh:
            fh.write("{}\t{}\n".format(sdate, status))

    @property
    def status(self):
        fname = os.path.join(self.signature_path, '.STATUS')
        if not os.path.isfile(fname):
            status = ('N/A', 'STATUS file not found!')
        with open(fname, 'r') as fh:
            for line in fh.readlines():
                status = tuple(line.strip().split('\t'))
        return status

    def get_status_stack(self):
        fname = os.path.join(self.signature_path, '.STATUS')
        status_stack = list()
        if not os.path.isfile(fname):
            status_stack.append(('N/A', 'STATUS file not found!'))
        with open(fname, 'r') as fh:
            for line in fh.readlines():
                status_stack.append(tuple(line.strip().split('\t')))
        return status_stack

    def mark_ready(self):
        fname = os.path.join(self.model_path, self.readyfile)
        with open(fname, 'w') as fh:
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

    def func_hpc(self, func_name, *args, **kwargs):
        """Execute the *any* method on the configured HPC.

        Args:
            args(tuple): the arguments for of the fit method
            kwargs(dict): arguments for the HPC method.
        """
        # read config file,# NS: get the cc_config var otherwise set it to
        # os.environ['CC_CONFIG']
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

        # Write the hpc script
        with open(script_path, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')

        # pickle self (the data) and fit args
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

    def __repr__(self):
        """String representig the signature."""
        return self.data_path

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

    @property
    def qualified_name(self):
        """Signature qualified name (e.g. 'B1.001-sign1-full')."""
        return "%s_%s_%s" % (self.dataset, self.cctype, self.molset)

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

    def get_cc(self, cc_root=None):
        '''Return the CC where the signature is present'''
        from chemicalchecker import ChemicalChecker
        if cc_root is None:
            cc_root = "/".join(self.signature_path.split("/")[:-5])
        return ChemicalChecker(cc_root)

    def get_sign(self, sign_type):
        '''Return the signature type for current dataset'''
        from .sign0 import sign0
        from .sign1 import sign1
        from .sign2 import sign2
        from .sign3 import sign3
        if sign_type not in ['sign%i' % i for i in range(5)]:
            raise ValueError('Wrong signature type: %s' % sign_type)
        folds = self.signature_path.split('/')
        folds[-1] = sign_type
        new_path = "/".join(folds)
        return eval(sign_type)(new_path, self.dataset)

    def get_non_redundant_intersection(self, sign):
        """Return the non redundant intersection between two signatures.

        (i.e. keys and vectors that are common to both signatures.)
        N.B: to maximize overlap it's better to use signatures of type 'full'.
        N.B: Near duplicates are found in the first signature.
        """
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

    def save_reference(self, cpu=4, overwrite=False):
        """Save a non redundant signature in reference molset.

        It generates a new signature in the references folders.

        Args:
            cpu(int): Number of CPUs (default=4),
            overwrite(bool): Overwrite existing (default=False).
        """
        if "sign" not in self.cctype:
            raise Exception("Only sign* are allowed.")
        if self.molset == 'reference':
            raise Exception("This is already `reference` molset.")

        rnd = RNDuplicates(cpu=cpu)
        sign_ref = self.get_molset("reference")
        if os.path.exists(sign_ref.data_path):
            if not overwrite:
                raise Exception("%s exists" % sign_ref.data_path)
        rnd.remove(self.data_path, save_dest=sign_ref.data_path)
        f5 = h5py.File(self.data_path, "r")
        if 'features' in f5.keys():
            features = f5['features'].asstr()[:]
            f5.close()
            with h5py.File(sign_ref.data_path, 'a') as hf:
                hf.create_dataset('features', data=features)
        return sign_ref

    def background_distances(self, metric):
        """Return the background distances according to the selected metric.

        Args:
            metric(str): the metric name (cosine or euclidean).
        """
        sign_ref = self
        if self.molset != 'reference':
            sign_ref = self.get_molset("reference")
        bg_file = os.path.join(sign_ref.model_path,
                               "bg_%s_distances.h5" % metric)
        return sign_ref.compute_distance_pvalues(bg_file, metric)
