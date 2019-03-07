"""This class simplify and standardize access to the Chemical Checker.

Main tasks of this class are:

1. Check and enforce the directory structure.
2. Serve signatures to users or pipelines.
"""

import os
import itertools
from glob import glob

from .data import DataFactory

from chemicalchecker.util import logged
from chemicalchecker.util import Config
from chemicalchecker.database import Dataset
from chemicalchecker.util.hpc import HPC


@logged
class ChemicalChecker():
    """Explore the Chemical Checker."""

    def __init__(self, cc_root):
        """Initialize the Chemical Checker.

        If the CC_ROOT directory is empty a skeleton of CC is initialized.
        Otherwise the directory is explored and molset and datasets variables
        are discovered.

        Args:
            cc_root(str): The Chemical Checker root directory. It's version
                dependendent.
        """
        self.cc_root = cc_root
        self._basic_molsets = ['reference', 'full']
        self._datasets = set()
        self._molsets = set(self._basic_molsets)
        self.__log.debug("ChemicalChecker with root: %s", cc_root)
        if not os.path.isdir(cc_root):
            self.__log.warning("Empty root directory, creating dataset dirs")
            for molset in self._basic_molsets:
                for dataset in Dataset.get():
                    ds = dataset.code
                    new_dir = os.path.join(cc_root, molset, ds[:1], ds[:2], ds)
                    self._datasets.add(ds)
                    self.__log.debug("Creating %s", new_dir)
                    original_umask = os.umask(0)
                    os.makedirs(new_dir, 0o775)
                    os.umask(original_umask)
        else:
            # if the directory exists get molsets and datasets
            paths = glob(os.path.join(cc_root, '*', '*', '*', '*'))
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

    def get_validation_path(self):
        """Return the validation path."""
        validation_path = os.path.join(
            self.cc_root, "tests", "validation_sets")
        self.__log.debug("validation path: %s", validation_path)
        return validation_path

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
        dataset = Dataset.get(dataset_code)
        if dataset is None:
            self.__log.warning(
                'Code %s returns no dataset', dataset_code)
            raise Exception("No dataset for code: " + dataset_code)
        signature_path = self.get_signature_path(cctype, molset, dataset_code)
        validation_path = self.get_validation_path()
        # the factory will return the signature with the right class
        data = DataFactory.make_data(
            cctype, signature_path, validation_path, dataset, **params)
        return data

    @staticmethod
    def remove_near_duplicates_hpc(job_path, cc_root, cctype):
        """Run HPC jobs to remove near duplicates of a signature.

        Args:
            job_path(str): Path (usually in scratch) where the script files are
                generated.
            cc_root(str): The Chemical Checker root directory.
            cctype(str): The Chemical Checker datatype (i.e. one of the sign*)
                for which duplicates will be removed.
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
            "from chemicalchecker.util import RNDuplicates",
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
            "    rnd.remove(sign_full.data_path.encode('ascii'))",
            "    rnd.save(sign_ref.data_path)",
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'remove_near_duplicates.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        all_datasets = [ds.code for ds in Dataset.get()]
        params = {}
        params["num_jobs"] = len(all_datasets)
        params["jobdir"] = job_path
        params["job_name"] = "CC_REFERENCE"
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
    def sign3_cross_fit(job_path, cc_root, pairs, mdl_root):
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
            "for ds1, ds2 in data:",  # elements are indexes
            "    sign_from = cc.get_signature('sign2', 'full_map', ds1)",
            "    sign_from.consistency_check()",
            "    sign_to = cc.get_signature('sign2', 'full_map', ds2)",
            "    sign_to.consistency_check()",
            "    suff = '%s_%s' % (ds1, ds2)",
            "    mdl_path = os.path.join('%s', suff)" % mdl_root,
            "    if not os.path.isdir(mdl_path):",
            "        os.makedirs(mdl_path)",
            "    sign3.cross_fit(sign_from, sign_to, mdl_path)",
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
        params["memory"] = 32  # this avoids singularity segfault on some nodes
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
            "    sign3_ref = cc.get_signature('sign3', 'reference', ds)",
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
