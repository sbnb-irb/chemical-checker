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
import pprint
import numpy as np

from .data import DataFactory
from .signature_data import DataSignature
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
        self.ds_sign3_full_map = "ZZ.001"
        self.ds_sign3_full_map_short = "ZZ.000"
        self.reference_code = "001"
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
            paths = glob(os.path.join(self.cc_root, '*', '*', '*', '*', '*',
                                      'sign*.h5'))
            self._molsets = set(x.split('/')[-6] for x in paths)
            self._datasets = set(x.split('/')[-3] for x in paths)
        self._molsets = sorted(list(self._molsets))
        self._datasets = [x for x in sorted(
            list(self._datasets)) if not x.endswith('000')]

    @property
    def coordinates(self):
        """Iterator on Chemical Checker coordinates."""
        for name, code in itertools.product("ABCDE", "12345"):
            yield name + code

    @property
    def datasets(self):
        """Iterator on Chemical Checker datasets."""
        for dataset in self._datasets:
            yield dataset

    def datasets_exemplary(self):
        """Iterator on Chemical Checker datasets."""
        for dataset in self.coordinates:
            yield dataset + '.001'

    @property
    def sign3_full_map_dataset(self):
        return self.ds_sign3_full_map

    @property
    def sign3_full_map_short_dataset(self):
        return self.ds_sign3_full_map_short

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
            molset_dataset_sign[molset][dataset].sort()
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

    def get_signature(self, cctype, molset, dataset_code, *args, **kwargs):
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
            cctype, signature_path, dataset_code, *args, **kwargs)
        return data

    def get_data_signature(self, cctype, dataset_code):
        """Return the data signature for the given dataset code.

        Args:
            cctype(str): The Chemical Checker datatype (i.e. one of the sign*).
            dataset_code(str): The dataset code of the Chemical Checker.
        Returns:
            data(Signature): A `DataSignature` object, the specific type depends
                on the cctype passed. It only allows access to the sign data.
        """
        args = ()
        kwargs = {}
        molset = "full"
        if len(dataset_code) == 2:
            dataset_code = dataset_code + "." + self.reference_code
        signature_path = self.get_signature_path(cctype, molset, dataset_code)
        # the factory will return the signature with the right class
        data = DataFactory.make_data(
            cctype, signature_path, dataset_code, *args, **kwargs)
        if not os.path.exists(data.data_path):
            self.__log.error(
                "There is no data for %s and dataset code %s" % (cctype, dataset_code))
            return None
        return DataSignature(data.data_path)

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

    def get_sign3_short_from_smiles(self, smiles, dest_file, chunk_size=1000):
        """Get the full signature3 short for a list of smiles.

        Args:
            smiles(list): A list of SMILES strings. We assume the user already
                standardized the SMILES string.
            dest_file(str): File where to save the short sign3.
        Returns:
            pred_s3(DataSignature): The predicted signatures as DataSignature
                object.
        """
        # initialize destination
        try:
            from chemicalchecker.tool.adanet import AdaNet
            from chemicalchecker.tool.autoencoder import AutoEncoder
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError as err:
            raise err
        ds_code_suffix = '001'
        predict_fn = {}

        sign3_short = self.get_signature(
            "sign3", "full", self.sign3_full_map_short_dataset)

        for coord in self.coordinates:
            sign3 = self.get_signature(
                "sign3", "full", coord + "." + ds_code_suffix)

            sign0_adanet_path = os.path.join(sign3.model_path,
                                             'adanet_sign0_A1.001_final',
                                             'savedmodel')

            predict_fn[coord + "." +
                       ds_code_suffix] = AdaNet.predict_fn(sign0_adanet_path)

        ds_codes = predict_fn.keys()
        ds_codes.sort()
        dest_dir = os.path.dirname(dest_file)
        # we return a simple DataSignature object (basic HDF5 access)
        temp_full_sign3 = os.path.join(dest_dir, "temp_sign3_full.h5")
        with h5py.File(temp_full_sign3, "w") as results:
            # initialize V (with NaN in case of failing rdkit) and smiles keys
            results.create_dataset('keys', data=np.array(
                smiles, DataSignature.string_dtype()))
            results.create_dataset(
                'V', (len(smiles), 128 * len(ds_codes)), dtype=np.float32)
            results.create_dataset("shape", data=(
                len(smiles), 128 * len(ds_codes)))
            # compute sign0 (i.e. Morgan fingerprint)
            nBits = 2048
            radius = 2
            # predict by chunk
            for i in range(0, len(smiles), chunk_size):
                chunk = slice(i, i + chunk_size)
                sign0s = list()
                sign3s = list()
                failed = list()
                for idx, mol_smiles in enumerate(smiles[chunk]):
                    try:
                        # read SMILES as molecules
                        mol = Chem.MolFromSmiles(mol_smiles)
                        if mol is None:
                            raise Exception("Cannot get molecule from smiles.")
                        info = {}
                        fp = AllChem.GetMorganFingerprintAsBitVect(
                            mol, radius, nBits=nBits, bitInfo=info)
                        bin_s0 = [fp.GetBit(i) for i in range(fp.GetNumBits())]
                        calc_s0 = np.array(bin_s0).astype(np.float32)
                    except Exception as err:
                        # in case of failure append a NaN vector
                        self.__log.warn("%s: %s", mol_smiles, str(err))
                        failed.append(idx)
                        calc_s0 = np.full((nBits, ), np.nan)
                    finally:
                        sign0s.append(calc_s0)
                # stack input signatures and generate predictions
                sign0s = np.vstack(sign0s)
                for ds in ds_codes:

                    preds = predict_fn[ds]({'x': sign0s})['predictions']
                    # add NaN when SMILES conversion failed
                    if failed:
                        preds[np.array(failed)] = np.full((128, ), np.nan)
                    sign3s.append(preds)
                # save chunk to H5
                results['V'][chunk] = np.hstack(sign3s)

        ae = AutoEncoder(sign3_short.model_path)

        pred_s3 = ae.encode(temp_full_sign3, dest_file)

        return pred_s3

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
        params["wait"] = False
        params["memory"] = 20  # writing to disk takes forever without enought
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC.from_config(Config())
        cluster.submitMultiJob(command, **params)
        return cluster

    @staticmethod
    def generate_ref_full_hpc(job_path, cc_root, datasets, from_data, to_data, **params):
        """Run HPC jobs to get new types of data.

        Args:
            job_path(str): Path (usually in scratch) where the script files are
                generated.
            cc_root(str): The Chemical Checker root directory.
            datasets(list): The list of datasets to run the scripts on.
            from_data(str): The type of data that we want to transform(e.g. sign1,sign2)
            to_data(str): The type of data that we want to create(e.g. clus1,sign2)
            cpu(int): Number of cores to use per dataset calculation. (default:10)
            memory(int): Number of G in RAM memory per dataset calculation. (default:24)
        """
        import chemicalchecker

        memory = 24
        cpu = 10
        for param, value in params.items():
            if "memory" in params:
                memory = params["memory"]
            if "cpu" in params:
                cpu = params["cpu"]

        # create job directory if not available
        if not os.path.isdir(job_path):
            os.mkdir(job_path)
        # create script file
        cc_config = os.environ['CC_CONFIG']
        cc_package = os.path.join(chemicalchecker.__path__[0], '../')
        script_lines1 = [
            "import sys, os",
            "import pickle",
            "os.environ['CC_CONFIG'] = '%s'" % cc_config,  # cc_config location
            "sys.path.append('%s')" % cc_package,  # allow package import
            "from chemicalchecker.core import ChemicalChecker",
            "cc = ChemicalChecker('%s')" % cc_root,
            "task_id = sys.argv[1]",  # <TASK_ID>
            "filename = sys.argv[2]",  # <FILE>
            "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
            "data = str(inputs[task_id][0])",  # elements for current job
            # start import
            '%s_full = cc.get_signature("%s","full",data)' % (
                from_data, from_data),
            # start import
            '%s_ref = cc.get_signature("%s","reference",data)' % (
                from_data, from_data),
            "pars = %s" % pprint.pformat(params),
            # start import
            '%s_ref = cc.get_signature("%s", "reference", data,**pars)' % (
                to_data, to_data)]
        script_lines3 = [
            "%s_full = cc.get_signature('%s', 'full', data,**pars)" % (
                to_data, to_data),
            "%s_ref.predict(%s_full, destination=%s_full.data_path, validations=True)" % (
                to_data, from_data, to_data),
            "%s_full.mark_ready()" % to_data,
            "print('JOB DONE')"
        ]

        if from_data == 'sign1' and to_data == 'sign2':

            script_lines2 = ['neig1_ref = cc.get_signature("neig1", "reference", data)',
                             "sign2_ref.fit(sign1_ref, neig1_ref, reuse=False)"]
        else:
            script_lines2 = [
                "%s_ref.fit(%s_ref)" % (to_data, from_data)]

        script_lines = script_lines1 + script_lines2 + script_lines3

        script_name = os.path.join(
            job_path, from_data + '_to_' + to_data + '.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')

        datasets.sort()
        # hpc parameters
        params = {}
        params["num_jobs"] = len(datasets)
        params["jobdir"] = job_path
        params["job_name"] = "CC_" + from_data + "_" + to_data
        params["elements"] = datasets
        params["wait"] = True
        params["memory"] = memory
        params["cpu"] = cpu
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC.from_config(Config())
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
        cluster = HPC.from_config(Config())
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
        cluster = HPC.from_config(Config())
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
        cluster = HPC.from_config(Config())
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
        cluster = HPC.from_config(Config())
        cluster.submitMultiJob(command, **params)
        return cluster

    @staticmethod
    def make_sign3_confidence_stratification(job_path, cc_root, cpu=1):
        """Run HPC jobs to copy signature with given threshold of confidence.

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
        cc = ChemicalChecker(cc_root)
        cc_config = os.environ['CC_CONFIG']
        cc_package = os.path.join(chemicalchecker.__path__[0], '../')
        script_lines = [
            "import sys, os",
            "import pickle",
            "import numpy as np",
            "os.environ['CC_CONFIG'] = '%s'" % cc_config,  # cc_config location
            "sys.path.append('%s')" % cc_package,  # allow package import
            "from chemicalchecker.core import ChemicalChecker",
            "cc = ChemicalChecker('%s')" % cc_root,
            "task_id = sys.argv[1]",  # <TASK_ID>
            "filename = sys.argv[2]",  # <FILE>
            "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
            "data = inputs[task_id]",  # elements for current job
            "for ds in data:",  # elements are indexes
            "    s3 = cc.get_signature('sign3', 'full', ds)",
            "    conf = s3.get_h5_dataset('confidence')",
            "    for thr in np.arange(0.1,1,0.1):",
            "        mask = conf > thr",
            "        s3_conf = cc.get_signature('sign3', 'conf%.1f' % thr, ds)",
            "        s3.make_filtered_copy(s3_conf.data_path, mask)",
            "        s3_conf.validate()",
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'sign3.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        all_datasets = sorted(list(cc.datasets_exemplary()))
        params = {}
        params["num_jobs"] = len(all_datasets)
        params["jobdir"] = job_path
        params["job_name"] = "CC_SIGN3_CONF"
        params["elements"] = all_datasets
        params["wait"] = False
        params["memory"] = 4  # this avoids singularity segfault on some nodes
        params["cpu"] = cpu  # Node2Vec parallelizes well
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC.from_config(Config())
        cluster.submitMultiJob(command, **params)
        return cluster
