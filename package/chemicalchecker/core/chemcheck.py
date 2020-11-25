"""Chemical Checker entry-point.

The most common starting point in a CC project is the
:class:`ChemicalChecker`::

    from chemicalchecker import ChemicalChecker
    cc = ChemicalChecker()

When initializing a CC instance we usually want provide a root directory.
If, like in the example above, we don't specify anything, the default path is
assumed (that is the ``CC_ROOT`` variable in the
:class:`~chemicalchecker.util.config.Config`).

If the specified ``CC_ROOT`` directory is already populated, we have
successfully initialized the CC instance and will have access to its
signatures.

If the ``CC_ROOT`` directory is empty we proceed generating the CC directory
structure and we'll have an empty CC instance optimal for handling our own
signatures.

The organization of signatures under the ``CC_ROOT`` follows hierarchy of
``molset``/``dataset``/``signature``.

    * The ``molset`` is mostly for internal usage, and its expected values are
      either "full" or "reference". In some steps of the pipeline is
      convenient to work with the non-redundant set of signatures
      ("reference") while at end we want to map back to the "full" set of
      molecules.

    * The ``dataset`` is the bioactivity space of interest and is described by
      the _level_ (e.g. ``A``) the _sublevel_ (e.g. ``1``) and a _code_ for
      each input dataset starting from ``.001``. The directory structure
      follow this hierarchy (e.g. ``/root/full/A/A1/A1.001``)

    * The ``signature`` is one of the possible type of signatures (see
      :doc:`Signaturization <../signaturization>`) and the final path is
      something like ``/root/full/A/A1/A1.001/sign2``


Main goals of this class are:
    1. Check and enforce the directory structure behind a CC instance.
    2. Serve signatures to users or pipelines.
"""
import re
import os
import h5py
import shutil
import itertools
import numpy as np
from glob import glob

from .molkit import Mol
from .data import DataFactory
from .preprocess import Preprocess
from .signature_data import DataSignature

from chemicalchecker.core.diagnostics import Diagnosis
from chemicalchecker.database import Dataset, Molecule
from chemicalchecker.util import logged, Config
from chemicalchecker.util.decorator import cached_property


@logged
class ChemicalChecker():
    """ChemicalChecker class."""

    def __init__(self, cc_root=None, custom_data_path=None):
        """Initialize a ChemicalChecker instance.

        If the CC_ROOT directory is empty a skeleton of CC is initialized.
        Otherwise the directory is explored and molset and datasets variables
        are discovered.

        Args:
            cc_root (None, str): The Chemical Checker root directory.
                If not specified the root is taken from the config file.
            custom_data_path (None, str): Path to one or more h5 files, detect
                their signature type, molset and dataset code form their
                'attrs' record.

        """
        # Default cc_root is taken from config file
        self.cc_root = cc_root
        if self.cc_root is None:
            self.cc_root = Config().PATH.CC_ROOT

        self._basic_molsets = ['reference', 'full']
        self._datasets = set()
        self._molsets = set(self._basic_molsets)
        self.reference_code = "001"
        self.__log.debug("ChemicalChecker with root: %s", self.cc_root)

        if custom_data_path is not None:
            # NS import one or several custom h5 files --> in any case a
            # cc_repo will exist afyter this block
            if '.' in custom_data_path.split('/')[-1]:
                # remove the file's name if provided (let it scan for h5 files
                # present there)
                custom_data_path = os.path.dirname(custom_data_path)

            self.custom_data_path = os.path.abspath(custom_data_path)
            self.__log.debug("Importing files from {}".format(
                self.custom_data_path))

            # Set a custom repo to avoid damaging ours
            self.cc_root = os.path.join(os.getcwd(), "cc_repo")

            if os.path.exists(self.cc_root):
                self.__log.warning(
                    "CC root {} exists, importing H5 files will "
                    "add signatures into it.".format(self.cc_root))

            else:
                self.__log.debug("Creating custom repo at {}".format(
                    self.cc_root))

                try:
                    original_umask = os.umask(0)
                    os.makedirs(self.cc_root, 0o775)
                    os.umask(original_umask)

                except Exception as e:
                    self.__log.error(
                        "Problem in creating cc_repo: {}".format(e))

            # Create  the cc_repo directory structure and symbolic link to
            # files
            self.import_h5()

        # If non-existing CC_root
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
            # NS: also valid for imported h5 datasets
            paths = glob(os.path.join(self.cc_root, '*',
                                      '*', '*', '*', '*', 'sign*.h5'))
            self._molsets = set(x.split('/')[-6] for x in paths)
            self._datasets = set(x.split('/')[-3] for x in paths)

        # In case
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

    @property
    def name(self):
        """Return the name of the Chemical Checker."""
        return os.path.basename(os.path.normpath(self.cc_root))

    def datasets_exemplary(self):
        """Iterator on Chemical Checker exemplary datasets."""
        for dataset in self.coordinates:
            yield dataset + '.001'

    @cached_property
    def universe(self):
        """Get the list of molecules in the CC universe.

        We define the CC universe as the union of all molecules found in sign0
        for any of the bioactivity datasets that are 'derived' and that are
        'essential'.
        """
        universe = set()
        dataset_accepted = []
        for ds in Dataset.get():
            if not ds.derived:
                self.__log.debug("Dataset '%s' not derived", ds)
                continue
            if not ds.essential:
                self.__log.debug("Dataset '%s' not essential", ds)
                continue
            s0 = self.get_signature('sign0', 'full', ds.code)
            dataset_accepted.append(ds.code)
            try:
                universe.update(s0.unique_keys)
            except Exception as ex:
                self.__log.warning(str(ex))
        self.__log.debug("Datasets defining universe: %s",
                         ' '.join(dataset_accepted))
        return sorted(list(universe))

    def get_universe_inchi(self):
        self.__log.debug("Fetching InChI of universe")
        ink_inchi = Molecule.get_inchikey_inchi_mapping(self.universe)
        return [ink_inchi[k] for k in self.universe]

    @staticmethod
    def set_verbosity(level='warning', logger_name='chemicalchecker'):
        '''Set the verbosity for logging module.'''
        import logging
        level = level.upper()
        levels = {'DEBUG': logging.DEBUG,
                  'INFO': logging.INFO,
                  'WARNING': logging.WARNING,
                  'ERROR': logging.ERROR,
                  'CRITICAL': logging.CRITICAL}
        log_fn = {'DEBUG': ChemicalChecker.__log.debug,
                  'INFO': ChemicalChecker.__log.info,
                  'WARNING': ChemicalChecker.__log.warning,
                  'ERROR': ChemicalChecker.__log.error,
                  'CRITICAL': ChemicalChecker.__log.critical}
        logger = logging.getLogger(logger_name)
        logger.setLevel(levels[level])
        log_fn[level]("Logging level %s for logger '%s'." %
                      (level.upper(), logger_name))

    def report_available(self, molset='*', dataset='*', signature='*'):
        """Report available signatures in the CC.

        Get the moleculeset/dataset combination where signatures are available.
        Use arguments to apply filters.

        Args:
            molset (str, optional): Filter for the moleculeset e.g. 'full' or
                'reference'
            dataset (str, optional): Filter for the dataset e.g. A1.001
            signature (str, optional): Filter for signature type e.g. 'sign1'

        Returns:
            Nested dictionary with molset, dataset and list of signatures
        """
        paths = glob(os.path.join(self.cc_root, molset, '*',
                                  '*', dataset, signature + '/*.h5'))
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
            try:
                with h5py.File(path, 'r') as fh:
                    if matrix not in fh.keys():
                        continue
                    molset_dataset_sign[molset][
                        dataset][sign] = fh[matrix].shape
            except Exception as ex:
                self.__log.warning(
                    'problem reading file %s: %s' % (path, str(ex)))
        return molset_dataset_sign

    def report_status(self, molset='*', dataset='*', signature='*'):
        """Report status of signatures in the CC.

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
            try:
                status_file = os.path.join(os.path.dirname(path), '.STATUS')
                if not os.path.isfile(status_file):
                    status = ('N/A', 'STATUS file not found!')
                else:
                    with open(status_file, 'r') as fh:
                        for line in fh.readlines():
                            status = line.strip().split('\t')
                molset_dataset_sign[molset][dataset][sign] = status
            except Exception as ex:
                self.__log.warning(
                    'problem reading file %s: %s' % (path, str(ex)))
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

        # self.__log.debug("signature path: %s", signature_path)
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
            data(Signature): A `DataSignature` object, the specific type
                depends on the cctype passed.
                It only allows access to the sign data.
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
                "There is no data for %s and dataset code %s" %
                (cctype, dataset_code))
            return None
        return DataSignature(data.data_path)

    def preprocess(self, sign):
        """Return the file with the raw data preprocessed.

        Args:
            sign: signature object obtained from cc.get_signature)
        Returns:
            datafile(str): The name of the file where the data in pairs is
                saved.
        """

        prepro = Preprocess(sign.signature_path, sign.dataset)
        if not prepro.is_fit():
            self.__log.info(
                "No preprocessed file found, calling the preprocessing script")
            prepro.fit()
        else:
            self.__log.info("Found {}".format(prepro.data_path))
        return prepro.data_path

        # ex:os.path.join(self.raw_path, "preprocess.h5")

    def preprocess_predict(self, sign, input_file, destination):
        """Runs the preprocessing script 'predict'.

        Run on an input file of raw data formatted correctly for the space of
        interest

        Args:
            sign: signature object obtained from cc.get_signature)
            input_file(str): path to the h5 file containning the data on which
                to apply 'predict'
            destination(str): Path to a .h5 file where the predicted signature
                will be saved.
        Returns:
            datafile(str): The h5 file containing the predicted data after
                preprocess
        """

        input_file = os.path.abspath(input_file)
        destination = os.path.abspath(destination)

        # Checking the provided paths

        if not os.path.exists(input_file):
            raise Exception("Error, {} does not exist!".format(input_file))

        ext = destination[-2:].lower()
        if not ext == 'h5':
            destination += '.h5'

        prepro = Preprocess(sign.signature_path, sign.dataset)
        prepro.predict(input_file, destination)

        return destination

    def signature(self, dataset, cctype):
        return self.get_signature(cctype=cctype, molset="full",
                                  dataset_code=dataset)

    def diagnosis(self, sign, **kwargs):
        return Diagnosis(self, sign, **kwargs)

    def import_h5(self):
        """Recovers h5 files from a given custom directory.

        Creates links to them in a CC skeleton arborescence.
        """

        h5files = glob(os.path.join(self.custom_data_path, "*.h5"))

        if len(h5files) == 0:
            self.__log.info(
                "No h5 file found in {},"
                " creating an empty CC structure.".format(
                    self.custom_data_path))

        else:

            available_files = ", ".join([os.path.basename(f) for f in h5files])
            self.__log.debug("Found h5 files {}: in {}".format(
                available_files, self.custom_data_path))

            # check the format of the imported info data
            # dataset code (ex: A1.001)
            formatDC = re.compile(r"[A-Z]\d\.\d\d\d")
            formatCCTYPE = re.compile(r"sign\d")
            formatMolset = re.compile(r"(full|reference)", re.IGNORECASE)

            # mapping info and required format
            formatDict = dict(dataset_code=formatDC,
                              cctype=formatCCTYPE, molset=formatMolset)

            def filter_dataset(path2h5file):
                """ returns a tuple of the type ('full', 'A', 'A1', 'A1.001',
                'sign3', path_to_h5file') or None if something's wrong
                """

                out = []
                with h5py.File(path2h5file, 'a') as ccfile:

                    # check if the required info is presents in the h5 file
                    # attrs dict
                    # iterates over ('dataset_code', 'cctype', 'molset') and
                    # the required format for each of them
                    for requiredKey, requiredFormat in formatDict.items():
                        if requiredKey not in ccfile.attrs:
                            self.__log.debug(
                                "Attribute {} cannot be retrieved from {},"
                                " skipping this file".format(
                                    requiredKey, ccfile))
                            return None

                        else:
                            # check the format of the provided info
                            if requiredFormat.match(ccfile.attrs[requiredKey]) is None:
                                self.__log.debug(
                                    "Problem with format",
                                    ccfile.attrs[requiredKey])
                                return None

                    #-------Now that the format is correct, output the info
                    # so that we just have to iterate over it to create the
                    # directory substructure
                    # full or reference
                    out.append(ccfile.attrs['molset'].lower())
                    out.append(ccfile.attrs['dataset_code'][0])  # i.e A
                    out.append(ccfile.attrs['dataset_code'][:2])  # i.e 1
                    out.append(ccfile.attrs['dataset_code'])     # i.e A1.001
                    # i.e sign3
                    out.append(ccfile.attrs['cctype'].lower())
                    out.append(path2h5file)

                return tuple(out)

            # Keep only h5 files that contain the required info in the correct
            # format
            h5tuples = [filter_dataset(
                f) for f in h5files if filter_dataset(f) is not None]

            if len(h5tuples) == 0:
                raise Exception(
                    "None of the provided h5 datasets have sufficient info in"
                    " its attributes! Please ensure myh5file.attrs has the"
                    "folllowing keys: 'dataset_code', 'cctype', 'molset'")

            # Now creating the cc_repo skeleton
            original_umask = os.umask(0)
            for h5t in h5tuples:

                # i.e ../../full/A/A1/A1.001/sign3
                path2sign = os.path.join(self.cc_root, '/'.join(h5t[:-1]))
                self.__log.debug("Attempting to create %s", path2sign)

                # If the signature already exists then propose to rename it
                # (ex: 00X) or skip it
                skip_signature = False
                while os.path.exists(path2sign):
                    self.__log.debug(
                        "Signature {} already exists for dataset {}".format(
                            h5t[4], h5t[3]))
                    resp = input("Rename it (r) or skip it (any other key)?")

                    if resp.lower() != 'r':
                        skip_signature = True
                        break

                    else:

                        # Check that the user entered the correct format
                        formatok = False
                        while not formatok:
                            newcode = input("New dataset code? (ex: 002)")

                            # I put A1 because all that matters is the 00x part
                            formatok = formatDict[
                                'dataset_code'].match('A1.' + newcode)

                            # True/False easier to deal with than None in this
                            # case
                            formatok = True if (
                                formatok is not None) else False
                            if not formatok:
                                self.__log.error(
                                    "Bad format, please try again.")

                        newtup = (h5t[0], h5t[1], h5t[2], h5t[
                                  2] + '.' + newcode, h5t[4])
                        # i.e ../../full/A/A1/A1.001/sign3
                        path2sign = os.path.join(
                            self.cc_root, '/'.join(newtup))
                        self.__log.debug(
                            "New signature path: {}".format(path2sign))

                if not skip_signature:
                    try:
                        os.makedirs(os.path.join(
                            self.cc_root, path2sign), 0o775)
                        # symbolic link to the h5 file in the cc_repo as
                        # signx.h5
                        os.symlink(
                            h5t[-1], os.path.join(
                                self.cc_root, path2sign, h5t[-2] + '.h5'))

                    except Exception as e:
                        os.umask(original_umask)
                        self.__log.error(
                            "Problem in creating "
                            "the cc custom repo: {}".format(e))

            os.umask(original_umask)  # after the loop to be sure

    def export(self, destination, signature, h5_filter=None,
               h5_names_map=None, overwrite=False, version=None):
        """Export a signature h5 file to a given path. 

        Which dataset to copy can be specified as well as how to rename some
        dataset.

        Args:
            destination(str): A destination path.
            signature(sign): A signature object.
            h5_filter(list): List of h5 dataset name to export.
            h5_names_map(dict): Dictionary of current to final h5 dataset name.
            overwrite(boo): Wether to allow overwriting the export.
            version(int): Mark the exported signature with a version number.
        """
        src_file = signature.data_path

        if not os.path.isfile(src_file):
            raise Exception('Signature must have an H5 file to export!')
        if not overwrite and os.path.isfile(destination):
            raise Exception('File %s already exists!' % destination)

        src = h5py.File(signature.data_path, 'r')
        dst = h5py.File(destination, 'w')
        if h5_filter is None:
            h5_filter = src.keys()
        if h5_names_map is None:
            h5_names_map = {x: x for x in h5_filter}
        for h5ds in h5_filter:
            if h5ds not in src:
                raise Exception('%s not available in %s' % (h5ds, src_file))
            h5src = src[h5ds][:]
            dst.create_dataset(h5_names_map[h5ds], data=h5src)
        if version is not None:
            dst.create_dataset('version', data=[version])

        # NS: Adding metadata so that they can be opened on local instances of
        # the checker:
        attributes = dict(dataset_code=signature.dataset,
                          cctype=signature.cctype, molset=signature.molset)
        if len(dst.attrs) != 3:
            for k, v in attributes.items():
                dst.attrs[k] = v

        src.close()
        dst.close()

    def symlink_to(self, source_cc, cctypes=['sign0'],
                   molsets=['reference', 'full'], datasets='exemplary'):
        """Link current CC instane to other via symlinks.

        When experimenting with signature parameters it's usefull to have
        low cctype (e.g. sign0, sign1) not copied but simply linked.

        Args:
            source_cc(ChemicalChecker): A different CC instance to link.
            cctypes(list): The signature (i.e. sign*) to link.
            molsets(list): The molecule set name to link .
            datasets(list): The codes of dataset to link.
        """
        if datasets == 'exemplary':
            datasets = list(self.datasets_exemplary())

        for molset in molsets:
            for ds in datasets:
                dst_ds_dir = os.path.join(
                    self.cc_root, molset, ds[:1], ds[:2], ds)
                src_ds_dir = os.path.join(
                    source_cc.cc_root, molset, ds[:1], ds[:2], ds)
                for cctype in cctypes:
                    dst_dir = os.path.join(dst_ds_dir, cctype)
                    src_dir = os.path.join(src_ds_dir, cctype)
                    self.__log.debug("Link %s --> %s", dst_dir, src_dir)
                    if os.path.isdir(dst_dir):
                        self.__log.warning("%s already present", dst_dir)
                        continue
                    os.symlink(src_dir, dst_dir)

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

    def _assert_equal(self, other_cc, cctypes=['sign1', 'sign2', 'sign3'],
                      molsets=['full', 'reference']):
        """Compare two ChemicalChecker instances."""
        for ds in self.datasets_exemplary():
            for cctype in ['sign1', 'sign2']:
                for molset in ['full', 'reference']:
                    s1 = self.get_signature(cctype, molset, ds)
                    s2 = other_cc.get_signature(cctype, molset, ds)
                    assert(all(s1[0] == s2[0]))
                    assert(all(s1[-1] == s2[-1]))
                    assert(s1.info_h5 == s2.info_h5)
        return True

    def get_molecule(self, mol_str, str_type=None):
        """Return a molecule `Mol` object.

        Args:
            mol_str: Compound identifier (e.g. SMILES string)
            str_type: Type of identifier ('inchikey', 'inchi' and 'smiles' are
                accepted) if 'None' we do our best to guess.
        """
        return Mol(self, mol_str, str_type=str_type)


    def get_global_signature(self, mol_str, str_type=None):
        """Return the (stacked) global signature
        If the given molecule belongs to the universe.


        Args:
            mol_str: Compound identifier (e.g. SMILES string)
            str_type: Type of identifier ('inchikey', 'inchi' and 'smiles' are
                accepted) if 'None' we do our best to guess.
        """
        try:
            mol = self.get_molecule(mol_str, str_type)
        except Exception as e:
            self.__log.warning(
                "Problem with generating molecule object from " + mol_str)
            self.__log.warning(e)
            return None

        if mol.inchikey in set(self.universe):
            spaces = [''.join(t) for t in itertools.product(
                'ABCDE', '12345', ['.001'])]
            try:
                global_sign = np.concatenate(
                    [mol.signature('sign3', sp) for sp in spaces], axis=0)
            except Exception as e2:
                self.__log.warning(
                    "Problem with generating global signature from " + mol_str)
                self.__log.warning(e2)
            else:
                return global_sign

        else:
            self.__log.warning(mol_str+" NOT IN UNIVERSE")


        return None
