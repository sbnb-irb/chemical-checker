"""This class simplify and standardize access to the Chemical Checker.

Main tasks of this class are:

1. Check and enforce the directory structure.
2. Serve signatures to users or pipelines.
"""

import os
import h5py
import shutil
import itertools
from glob import glob

from .data import DataFactory
from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.util import Config
from chemicalchecker.util.hpc import HPC


@logged
class ChemicalChecker():
    """Explore the Chemical Checker."""

    def __init__(self, cc_root=None):
        """Initialize the Chemical Checker.

        If the CC_ROOT directory is empty a skeleton of CC is initialized.
        Otherwise the directory is explored and molset and datasets variables
        are discovered.

        Args:
            cc_root(str): The Chemical Checker root directory. If not specified
                            the root is taken from the config file. (default:None)
        """
        if not cc_root:
            self.cc_root = Config().PATH.CC_ROOT
        else:
            self.cc_root = cc_root
        self._basic_molsets = ['reference', 'full']
        self._datasets = set()
        self._molsets = set(self._basic_molsets)
        self.__log.debug("ChemicalChecker with root: %s", self.cc_root)
        if not os.path.isdir(self.cc_root):
            self.__log.warning("Empty root directory, creating dataset dirs")
            for molset in self._basic_molsets:
                for dataset in Dataset.get():
                    ds = dataset.dataset_code
                    new_dir = os.path.join(
                        self.cc_root, molset, ds[:1], ds[:2], ds)
                    self._datasets.add(ds)
                    self.__log.debug("Creating %s", new_dir)
                    original_umask = os.umask(0)
                    os.makedirs(new_dir, 0o775)
                    os.umask(original_umask)
        else:
            # if the directory exists get molsets and datasets
            paths = glob(os.path.join(self.cc_root, '*', '*', '*', '*'))
            self._molsets = set(x.split('/')[-4] for x in paths)
            self._datasets = set(x.split('/')[-1] for x in paths)
        self._molsets = sorted(list(self._molsets))
        self._datasets = sorted(list(self._datasets))

    @property
    def coordinates(self):
        """Iterator on Chemical Checker coordinates."""
        for name, code in itertools.product("ABCDE", "12345"):
            yield name + code

    @property
    def datasets(self, exemplary_only=False):
        """Iterator on Chemical Checker datasets."""
        for dataset in self._datasets:
            yield dataset

    def report_available(self, molset='*', dataset='*', signature='*'):
        """Report available signatures in the CC.

        Get the moleculeset/dataset combination where signatures are available.
        Use arguments to apply filters.
        Args:
            molset(str): Filter for the moleculeset e.g. 'full' or 'reference'
            dataset(str) Filter for the dataset e.g. A1.001
            signature(str): Filter for signature type e.g. 'sign1'
        Returns:
            Nested dictionary with molset, dataset and list of signatures
        """
        paths = glob(os.path.join(self.cc_root, molset, '*', '*', dataset,
                                  signature + '/*.h5'))
        molset_dataset_sign = dict()
        for path in paths:
            molset = path.split('/')[-6]
            dataset = path.split('/')[-3]
            sign = path.split('/')[-2]
            if molset not in molset_dataset_sign:
                molset_dataset_sign[molset] = dict()
            if dataset not in molset_dataset_sign[molset]:
                molset_dataset_sign[molset][dataset] = list()
            molset_dataset_sign[molset][dataset].append(sign)
        return molset_dataset_sign

    def report_sizes(self, molset='*', dataset='*', signature='*', matrix='V'):
        """Report sizes of available signatures in the CC.

        Get the moleculeset/dataset combination where signatures are available.
        Report the size of the 'V' matrix.
        Use arguments to apply filters.
        Args:
            molset(str): Filter for the moleculeset e.g. 'full' or 'reference'
            dataset(str) Filter for the dataset e.g. A1.001
            signature(str): Filter for signature type e.g. 'sign1'
        Returns:
            Nested dictionary with molset, dataset and list of signatures
        """
        paths = glob(os.path.join(self.cc_root, molset, '*', '*', dataset,
                                  signature + '/*.h5'))
        molset_dataset_sign = dict()
        for path in paths:
            molset = path.split('/')[-6]
            dataset = path.split('/')[-3]
            sign = path.split('/')[-2]
            if molset not in molset_dataset_sign:
                molset_dataset_sign[molset] = dict()
            if dataset not in molset_dataset_sign[molset]:
                molset_dataset_sign[molset][dataset] = dict()
            with h5py.File(path, 'r') as fh:
                if matrix not in fh.keys():
                    continue
                molset_dataset_sign[molset][dataset][sign] = fh[matrix].shape
        return molset_dataset_sign

    def get_signature_path(self, cctype, molset, dataset_code):
        """Return the signature path for the given dataset code.

        This should be the only place where we define the directory structure.
        The signature directory tipically contain the signature HDF5 file.

        Args:
            cctype(str): The Chemical Checker datatype i.e. one of the sign*.
            molset(str): The molecule set name.
            dataset_code(str): The dataset of the Chemical Checker.
        Returns:
            signature_path(str): The signature path.
        """
        signature_path = os.path.join(self.cc_root, molset, dataset_code[:1],
                                      dataset_code[:2], dataset_code, cctype)
        self.__log.debug("signature path: %s", signature_path)
        return signature_path

    def get_signature(self, cctype, molset, dataset_code, **params):
        """Return the signature for the given dataset code.

        Args:
            cctype(str): The Chemical Checker datatype (i.e. one of the sign*).
            molset(str): The molecule set name.
            dataset_code(str): The dataset code of the Chemical Checker.
            params(dict): Optional. The set of parameters to initialize and
                compute the signature. If the signature is already initialized
                this argument will be ignored.
        Returns:
            data(Signature): A `Signature` object, the specific type depends
                on the cctype passed.
        """
        signature_path = self.get_signature_path(cctype, molset, dataset_code)
        # the factory will return the signature with the right class
        data = DataFactory.make_data(
            cctype, signature_path, dataset_code, **params)
        return data

    def copy_signature_from(self, source_cc, cctype, molset, dataset_code,
                            overwrite=False):
        """Copy a signature file from another CC instance.

        Args:
            source_cc(ChemicalChecker): A different CC instance.
            cctype(str): The Chemical Checker datatype (i.e. one of the sign*).
            molset(str): The molecule set name.
            dataset_code(str): The dataset code of the Chemical Checker.
        """
        # initialize destination
        dst_signature_path = self.get_signature_path(
            cctype, molset, dataset_code)
        dst_sign = DataFactory.make_data(
            cctype, dst_signature_path, dataset_code)
        # initializa source
        src_signature_path = source_cc.get_signature_path(
            cctype, molset, dataset_code)
        src_sign = DataFactory.make_data(
            cctype, src_signature_path, dataset_code)
        # copy data
        src = src_sign.data_path
        dst = dst_sign.data_path
        self.__log.info("Copying signature from %s to %s", src, dst)
        if not os.path.isfile(src):
            raise Exception("Source file %s does not exists.", src)
        if os.path.isfile(dst):
            self.__log.info("File %s exists already.", dst)
            if not overwrite:
                raise Exception("File %s exists already.", dst)
        shutil.copyfile(src, dst)

    @staticmethod
    def remove_near_duplicates_hpc(job_path, cc_root, cctype, datasets):
        """Run HPC jobs to remove near duplicates of a signature.

        Args:
            job_path(str): Path (usually in scratch) where the script files are
                generated.
            cc_root(str): The Chemical Checker root directory.
            cctype(str): The Chemical Checker datatype (i.e. one of the sign*)
                for which duplicates will be removed.
        """
        # create job directory if not available
        if not os.path.isdir(job_path):
            os.mkdir(job_path)
        # create script file
        script_lines = [
            "import sys, os",
            "import pickle",
            "sys.path.append('%s')" % os.path.join(
                Config().PATH.CC_REPO, 'package'),  # allow package import
            "from chemicalchecker.util.remove_near_duplicates import RNDuplicates",
            "from chemicalchecker.core import ChemicalChecker",
            "cc = ChemicalChecker('%s')" % cc_root,
            "task_id = sys.argv[1]",  # <TASK_ID>
            "filename = sys.argv[2]",  # <FILE>
            "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
            "data = inputs[task_id]",  # elements for current job
            "for ds in data:",  # elements are indexes
            "    sign_full = cc.get_signature('%s', 'full', ds)" % cctype,
            "    sign_ref = cc.get_signature('%s', 'reference', ds)" % cctype,
            "    rnd = RNDuplicates()",
            "    rnd.remove(sign_full.data_path, save_dest=sign_ref.data_path)",
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'remove_near_duplicates.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        all_datasets = datasets
        params = {}
        params["num_jobs"] = len(all_datasets)
        params["jobdir"] = job_path
        params["job_name"] = "CC_REFERENCE"
        params["elements"] = all_datasets
        params["wait"] = True
        params["memory"] = 20 # writing to disk takes forever without enought
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC(Config())
        cluster.submitMultiJob(command, **params)
        return cluster

    @staticmethod
    def sign1_to_neig1_hpc(job_path, cc_root, molset):
        """Run HPC jobs to get nearest neighbor.

        Args:
            job_path(str): Path (usually in scratch) where the script files are
                generated.
            cc_root(str): The Chemical Checker root directory.
            molset(str): The Chemical Checker molset (eg. 'reference', 'full').
        """
        # create job directory if not available
        import chemicalchecker
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
            "for ds in data:",  # elements are indexes
            "    sign1_ref = cc.get_signature('sign1', '%s', ds)" % molset,
            "    neig1_ref = cc.get_signature('neig1', '%s', ds)" % molset,
            "    neig1_ref.fit(sign1_ref)",
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'sign1_to_neig1.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        all_datasets = [ds.code for ds in Dataset.get()]
        params = {}
        params["num_jobs"] = len(all_datasets)
        params["jobdir"] = job_path
        params["job_name"] = "CC_SIGN1_TO_NEIG1"
        params["elements"] = all_datasets
        params["wait"] = True
        params["memory"] = 1  # this avoids singularity segfault on some nodes
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC(Config())
        cluster.submitMultiJob(command, **params)
        return cluster

    @staticmethod
    def compute_sign2_hpc(job_path, cc_root, cpu=1):
        """Run HPC jobs to compute signature type 2.

        Args:
            job_path(str): Path (usually in scratch) where the script files are
                generated.
            cc_root(str): The Chemical Checker root directory.
            cpu(int): Number of cores to reserve.
        """
        # create job directory if not available
        import chemicalchecker
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
            "for ds in data:",  # elements are indexes
            "    sign1_ref = cc.get_signature('sign1', 'reference', ds)",
            "    neig1_ref = cc.get_signature('neig1', 'reference', ds)",
            "    sign2_ref = cc.get_signature('sign2', 'reference', ds)",
            "    sign2_ref.fit(sign1_ref, neig1_ref)",
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'sign2.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        all_datasets = [ds.code for ds in Dataset.get()]
        params = {}
        params["num_jobs"] = len(all_datasets)
        params["jobdir"] = job_path
        params["job_name"] = "CC_SIGN2"
        params["elements"] = all_datasets
        params["wait"] = True
        params["memory"] = 1  # this avoids singularity segfault on some nodes
        params["cpu"] = cpu  # Node2Vec parallelizes well
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC(Config())
        cluster.submitMultiJob(command, **params)
        return cluster

    @staticmethod
    def map_sign2_full_hpc(job_path, cc_root):
        """Run HPC jobs mapping signature from reference to full.

        Args:
            job_path(str): Path (usually in scratch) where the script files are
                generated.
            cc_root(str): The Chemical Checker root directory.
        """
        # create job directory if not available
        import chemicalchecker
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
            "for ds in data:",  # elements are indexes
            "    sign1_ref = cc.get_signature('sign1', 'reference', ds)",
            "    sign1_ref.consistency_check()",
            "    sign2_ref = cc.get_signature('sign2', 'reference', ds)",
            "    sign2_ref.consistency_check()",
            "    sign2_map = cc.get_signature('sign2', 'full_map', ds)",
            "    sign2_ref.copy_from(sign1_ref, 'mappings')",
            "    sign2_ref.map(sign2_map.data_path)",
            "    sign2_map.consistency_check()",
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'sign2.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        all_datasets = [ds.code for ds in Dataset.get()]
        params = {}
        params["num_jobs"] = len(all_datasets)
        params["jobdir"] = job_path
        params["job_name"] = "CC_SIGN2_FULL_MAP"
        params["elements"] = all_datasets
        params["wait"] = True
        params["memory"] = 1  # this avoids singularity segfault on some nodes
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC(Config())
        cluster.submitMultiJob(command, **params)
        return cluster

    @staticmethod
    def sign3_cross_fit(job_path, cc_root, pairs):
        """Run HPC jobs performing adanet cross fit between pairs.

        Args:
            job_path(str): Path (usually in scratch) where the script files are
                generated.
            cc_root(str): The Chemical Checker root directory.
            pairs(list): pairs of datasets.
        """
        # create job directory if not available
        import chemicalchecker
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
            "from chemicalchecker.core.sign3 import sign3",
            "cc = ChemicalChecker('%s')" % cc_root,
            "task_id = sys.argv[1]",  # <TASK_ID>
            "filename = sys.argv[2]",  # <FILE>
            "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
            "data = inputs[task_id]",  # elements for current job
            "for ds_from, ds_to in data:",  # elements are indexes
            "    s3 = cc.get_signature('sign3', 'full_map', ds_to)",
            "    sign3.cross_fit(cc, s3.model_path, ds_from, ds_to)",
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'sign3_cross.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        params = {}
        params["num_jobs"] = len(pairs)
        params["jobdir"] = job_path
        params["job_name"] = "CC_SIGN3_CROSS"
        params["elements"] = pairs
        params["wait"] = True
        params["memory"] = 16  # this avoids singularity segfault on some nodes
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC(Config())
        cluster.submitMultiJob(command, **params)
        return cluster

    @staticmethod
    def compute_sign3_hpc(job_path, cc_root, cpu=1):
        """Run HPC jobs to compute signature type 2.

        Args:
            job_path(str): Path (usually in scratch) where the script files are
                generated.
            cc_root(str): The Chemical Checker root directory.
            cpu(int): Number of cores to reserve.
        """
        # create job directory if not available
        import chemicalchecker
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
            "for ds in data:",  # elements are indexes
            "    sign3_ref = cc.get_signature('sign3', 'full_map', ds)",
            "    sign3_ref.fit(cc)",
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'sign3.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        all_datasets = [ds.code for ds in Dataset.get()]
        params = {}
        params["num_jobs"] = len(all_datasets)
        params["jobdir"] = job_path
        params["job_name"] = "CC_SIGN3"
        params["elements"] = all_datasets
        params["wait"] = True
        params["memory"] = 16  # this avoids singularity segfault on some nodes
        params["cpu"] = cpu  # Node2Vec parallelizes well
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC(Config())
        cluster.submitMultiJob(command, **params)
        return cluster
