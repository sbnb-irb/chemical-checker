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
import gzip
import h5py
import json
import shutil
import itertools
import numpy as np
from glob import glob
from pathlib import Path

from .molkit import Mol
from .data import DataFactory
from .signature_data import DataSignature

from chemicalchecker.database import Dataset, Molecule
from chemicalchecker.database.database import test_connection
from chemicalchecker.util import logged, Config
from chemicalchecker.util.decorator import cached_property


@logged
class ChemicalChecker():
    """ChemicalChecker class."""

    def __init__(self, cc_root=None, custom_data_path=None, dbconnect=True):
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
            dbconnect (True, Bool): if True, try to connect to the DB
        """
        # Default cc_root is taken from config file
        self.cc_root = cc_root
        if self.cc_root is None:
            self.cc_root = Config().PATH.CC_ROOT

        self._basic_molsets = ['reference', 'full']
        self._datasets = set()
        self._molsets = set(self._basic_molsets)
        self._exemplary_id = "001"
        self._metadata = None
        self.__log.debug("ChemicalChecker with root: %s", self.cc_root)

        # If non-existing CC_root
        if not os.path.isdir(self.cc_root):
            self.__log.debug("Empty root directory,"
                             "creating root and dataset dirs")
            original_umask = os.umask(0)
            os.makedirs(self.cc_root, 0o775)
            os.umask(original_umask)

            # Test connection to DB, the database is not necessarily present
            self.__log.debug("Testing DB connection.")
            if dbconnect and test_connection():
                for molset in self._basic_molsets:
                    for dataset in Dataset.get():
                        ds = dataset.dataset_code
                        new_dir = os.path.join(
                            self.cc_root, molset, ds[:1], ds[:2], ds)
                        self._datasets.add(ds)
                        original_umask = os.umask(0)
                        os.makedirs(new_dir, 0o775)
                        os.umask(original_umask)
            else:
                self.__log.debug("No database found, working locally.")

        else:
            # if the directory exists get molsets and datasets
            paths = self._available_sign_paths(filename='sign*.h5')
            self._molsets = set(x.split('/')[-6] for x in paths)
            self._datasets = set(x.split('/')[-3] for x in paths)
            if custom_data_path is not None:
                self.__log.info("CC root directory exists: "
                                "ignoring 'custom_data_path'.")
                custom_data_path = None

        # import one or several custom h5 files
        if custom_data_path is not None:
            if os.path.isfile(custom_data_path):
                raise Exception("'custom_data_path' must be a directory")

            custom_data_path = os.path.abspath(custom_data_path)
            self.__log.debug("Linking files from: %s" % custom_data_path)
            self.link_h5(custom_data_path)

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

    @property
    def metadata(self):
        """Return the metadata of the Chemical Checker."""
        if self._metadata is None:
            self._metadata = self._get_metadata()
        return self._metadata

    def sign_metadata(self, key, molset, dataset, cctype):
        """Return the metadata of the Chemical Checker."""
        if self._metadata is None:
            self._metadata = self._get_metadata()
        if key not in self._metadata:
            raise Exception("Key `%s` not found in metadata" % key)
        if molset in self._metadata[key]:
            if dataset in self._metadata[key][molset]:
                if cctype in self._metadata[key][molset][dataset]:
                    return self._metadata[key][molset][dataset][cctype]
        try:
            sign = self.get_signature(cctype, molset, dataset)
        except Exception:
            self.__log.debug(
                "Signature not found, skipping: %s %s %s %s" %
                (self.name, *cctype, molset, dataset))
            return None
        if key == 'dimensions':
            return sign.shape
        if key == 'keys':
            return sign.keys

    def _get_metadata(self, overwrite=False):
        fn = os.path.join(self.cc_root, "metadata.json.zip")
        metadata = dict()
        if not os.path.exists(fn) or overwrite:
            metadata['dimensions'] = self.report_dimensions()
            metadata['keys'] = self.report_keys()
            json_str = json.dumps(metadata) + "\n"
            json_bytes = json_str.encode('utf-8')
            with gzip.open(fn, "w") as fh:
                fh.write(json_bytes)
            return metadata
        with gzip.open(fn, "r") as fh:
            json_bytes = fh.read()
        json_str = json_bytes.decode('utf-8')
        metadata = json.loads(json_str)
        return metadata

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
    def set_verbosity(level='warning', logger_name='chemicalchecker',
                      format=None):
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
        if level == 'DEBUG':
            ch = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s %(name)-12s [%(levelname)-8s] %(message)s')
            ch.setFormatter(formatter)
            logger.handlers = []
            logger.addHandler(ch)
        else:
            ch = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(levelname)-8s] %(message)s')
            ch.setFormatter(formatter)
            logger.handlers = []
            logger.addHandler(ch)
        logger.setLevel(levels[level])
        log_fn[level]("Logging level %s for logger '%s'." %
                      (level.upper(), logger_name))

    def _available_sign_paths(self, molset='*', dataset='*', signature='*',
                              filename='*.h5'):
        paths = glob(os.path.join(self.cc_root, molset, '*',
                                  '*', dataset, signature, filename))
        return paths

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
        paths = self._available_sign_paths(molset, dataset, signature)
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

    def report_keys(self, molset='full', dataset='*', signature='sign1'):
        """Report keys of all available signatures in the CC.

        Get the moleculeset/dataset combination where signatures are available.
        Report the list of keys. Use arguments to apply filters.
        Args:
            molset(str): Filter for the moleculeset e.g. 'full' or 'reference'
            dataset(str) Filter for the dataset e.g. A1.001
            signature(str): Filter for signature type e.g. 'sign1'
        Returns:
            Nested dictionary with molset, dataset and list of signatures
        """
        paths = self._available_sign_paths(molset, dataset, signature)
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
                    if 'keys' not in fh.keys():
                        continue
                    decoder = np.vectorize(lambda x: x.decode())
                    molset_dataset_sign[molset][
                        dataset][sign] = decoder(fh['keys'][:]).tolist()
            except Exception as ex:
                self.__log.warning(
                    'problem reading file %s: %s' % (path, str(ex)))
        return molset_dataset_sign

    def report_dimensions(self, molset='*', dataset='*', signature='*',
                          matrix='V'):
        """Report dimensions of all available signatures in the CC.

        Get the moleculeset/dataset combination where signatures are available.
        Report the size of the 'V' matrix. Use arguments to apply filters.
        Args:
            molset(str): Filter for the moleculeset e.g. 'full' or 'reference'
            dataset(str) Filter for the dataset e.g. A1.001
            signature(str): Filter for signature type e.g. 'sign1'
        Returns:
            Nested dictionary with molset, dataset and list of signatures
        """
        paths = self._available_sign_paths(molset, dataset, signature)
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
        paths = self._available_sign_paths(molset, dataset, signature)
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
            dataset_code = dataset_code + "." + self._exemplary_id
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

    def signature(self, dataset, cctype):
        return self.get_signature(cctype=cctype, molset="full",
                                  dataset_code=dataset)

    def link_h5(self, custom_data_path):
        """Link H5 files from a given custom directory.

        Populates local CC instance with symlinks to external signatures H5s.

        Args:
            custom_data_path(str): Path to a directory signature containing H5s
        """
        h5files = glob(os.path.join(custom_data_path, "*.h5"))

        if len(h5files) == 0:
            self.__log.warning("No h5 file found in %s, "
                               "CC instance will be empty." % custom_data_path)
            return

        available_files = ", ".join([os.path.basename(f) for f in h5files])
        self.__log.debug("Found h5 files {}: in {}".format(
            available_files, custom_data_path))

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
            with h5py.File(path2h5file, 'r') as ccfile:

                # check if the required info is presents in the h5 file
                # attrs dict
                # iterates over ('dataset_code', 'cctype', 'molset') and
                # the required format for each of them
                for requiredKey, requiredFormat in formatDict.items():
                    if requiredKey not in ccfile.attrs:
                        self.__log.warning(
                            "Attribute {} cannot be retrieved from {},"
                            " skipping this file".format(
                                requiredKey, ccfile))
                        return None

                    else:
                        # check the format of the provided info
                        matching = requiredFormat.match(
                            ccfile.attrs[requiredKey])
                        if matching is None:
                            self.__log.warning(
                                "Problem with format",
                                ccfile.attrs[requiredKey])
                            return None

                # Prepare the signature file name and path
                out.append(ccfile.attrs['molset'].lower())
                out.append(ccfile.attrs['dataset_code'][0])
                out.append(ccfile.attrs['dataset_code'][:2])
                out.append(ccfile.attrs['dataset_code'])
                out.append(ccfile.attrs['cctype'].lower())
                out.append(path2h5file)
            return tuple(out)

        # Keep h5 files containing the required info in the correct format
        h5tuples = list()
        for fn in h5files:
            res = filter_dataset(fn)
            if res:
                h5tuples.append(res)
            else:
                # guess from filename
                name = Path(fn).stem
                cctype, dataset_code, molset = name.split('_')
                res = []
                res.append(molset.lower())
                res.append(dataset_code[0])
                res.append(dataset_code[:2])
                res.append(dataset_code)
                res.append(cctype.lower())
                res.append(fn)
                h5tuples.append(res)

        if len(h5tuples) == 0:
            raise Exception(
                "None of the provided h5 datasets have sufficient info in"
                " its attributes! Please ensure myh5file.attrs has the"
                "folllowing keys: 'dataset_code', 'cctype', 'molset'")

        # Now creating the instance folder structure
        original_umask = os.umask(0)
        for h5t in h5tuples:
            # ex: ../../full/A/A1/A1.001/sign3
            path2sign = os.path.join(self.cc_root, *h5t[:-1])
            self.__log.debug("Creating: %s", path2sign)

            # The signature should not exist
            if os.path.exists(path2sign):
                self.__log.debug("Skipping as path exists: %s", path2sign)
                continue

            # create dir
            os.makedirs(path2sign, 0o775)
            # symbolic link to the h5 file in the cc_repo as signx.h5
            os.symlink(h5t[-1], os.path.join(path2sign, h5t[-2] + '.h5'))
        os.umask(original_umask)

    def export(self, destination, signature, h5_filter=None,
               h5_names_map={}, overwrite=False, version=None):
        """Export a signature h5 file to a given path.

        Which dataset to copy can be specified as well as how to rename some
        dataset.

        Args:
            destination(str): A destination path.
            signature(sign): A signature object.
            h5_filter(list): List of h5 dataset name to export.
            h5_names_map(dict): Dictionary of current to final h5 dataset name.
            overwrite(boo): Whether to allow overwriting the export.
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
        h5_names_map = {x: x for x in h5_filter}
        h5_names_map.update(h5_names_map)
        for h5ds in h5_filter:
            if h5ds not in src:
                raise Exception('%s not available in %s' % (h5ds, src_file))
            h5src = src[h5ds][:]
            dst.create_dataset(h5_names_map[h5ds], data=h5src)
        if version is not None:
            dst.create_dataset('version', data=[version])

        # adding metadata
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
            self.__log.warning(mol_str + " NOT IN UNIVERSE")

        return None

    def export_symlinks(self, dest_path=None, essential_only=True):
        """Creates symlinks for all available signatures in a single folder.

        Args:
            dest_path (str): The destination for symlink, if None then the
                default under the cc_root is generated.
        """
        if dest_path is None:
            dest_path = os.path.join(self.cc_root, 'sign_links')

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        for molset in ['full', 'reference']:
            for cctype in ['sign0', 'sign1', 'sign2', 'sign3', 'sign4']:
                for ds in self.coordinates:
                    dataset_code = ds + '.001'
                    sign = self.get_signature(cctype, molset, dataset_code)
                    sign_file = sign.data_path

                    if os.path.isfile(sign_file):
                        dst_name = "_".join([cctype, dataset_code, molset])
                        dst_name += ".h5"
                        # Make a symlink into the destination
                        symlink = os.path.join(dest_path, dst_name)
                        try:
                            os.symlink(sign_file, symlink)
                        except Exception as ex:
                            self.__log.warning(
                                "Error creating %s: %s" % (symlink, str(ex)))

    def add_sign_metadata(self, molset='*', dataset='*', signature='*'):
        """Add metadata to available signatures.

        Args:
            molset (str, optional): Filter for the moleculeset e.g. 'full' or
                'reference'
            dataset (str, optional): Filter for the dataset e.g. A1.001
            signature (str, optional): Filter for signature type e.g. 'sign1'
        """
        paths = self._available_sign_paths(molset, dataset, signature)
        for path in paths:
            molset = path.split('/')[-6]
            dataset = path.split('/')[-3]
            sign = path.split('/')[-2]
            metadata = dict(cctype=sign, dataset_code=dataset, molset=molset)
            try:
                with h5py.File(path, 'a') as fh:
                    for k, v in metadata.items():
                        if k in fh.attrs:
                            del fh.attrs[k]
                        fh.attrs.create(name=k, data=v)
            except Exception:
                self.__log.warning("Could not add metadata to: %s" % path)
                continue
