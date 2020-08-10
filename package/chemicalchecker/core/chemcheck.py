"""Standardize internal structure and access to the Chemical Checker
methods and signatures.

When initializing a CC instance we must provide a root directory. If the
CC_ROOT is empty we proceed generating the CC directory structure and
we'll have an empty CC instance. If the CC_ROOT directory is not empty,
we discover signatures assuming a _molset_, _dataset_, _signature_
hierarchy.

- The _molset_ is mostly for internal usage, and it expected values are
either "full" or "reference". In some steps of the pipeline is
convenient to work with the non-redundant set of signatures
("reference") while at end we want to map back to the "full" set of
molecules.

- The _dataset_ is the bioactivity space of interest and is described by
the _level_ (e.g. "A") the _sublevel_ (e.g. "1") and a _code_ for each
input dataset starting from .001. The directory structure follow this
hierarchy (e.g. "/root/full/A/A1/A1.001" )

- The _signature_ is one of the possible type of signatures (`sign0`,
`sign1`, `sign2` and `sign3`), for each of these special signatures
types with precomputed data can be available. Namely: `neig` for
nearest neighbor, `clus` for clustered signatures, `proj` for the
2D-projections.


Main tasks of this class are:

1. Check and enforce the directory structure behind a CC instance.
2. Serve signatures to users or pipelines.
"""

import os
import h5py
import shutil
import itertools
import re
from glob import glob

from .molkit import Mol
from .data import DataFactory
from .preprocess import Preprocess
from .signature_data import DataSignature

from chemicalchecker.util import logged, Config
from chemicalchecker.database import Dataset


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
class ChemicalChecker():
    """Explore the Chemical Checker."""

    def __init__(self, cc_root=None, custom_data_path=None):
        """Initialize the Chemical Checker.

        If the CC_ROOT directory is empty a skeleton of CC is initialized.
        Otherwise the directory is explored and molset and datasets variables
        are discovered.

        Args:
            cc_root(str): The Chemical Checker root directory. If not specified
                          the root is taken from the config file.
                          (default:None)

            custom_data_path: Path to one or more h5 files, detect their signature
                          type, molset and dataset code form their 'attrs' record.

        """
        if not cc_root:
            self.cc_root = Config().PATH.CC_ROOT

        else:
            self.cc_root = cc_root


        self._basic_molsets = ['reference', 'full']
        self._datasets = set()
        self._molsets = set(self._basic_molsets)
        self.reference_code = "001"
        self.__log.debug("ChemicalChecker with root: %s", self.cc_root)


        if custom_data_path is not None:
            #NS import one or several custom h5 files --> in any case a cc_repo will axist afyter this block
            if '.' in custom_data_path.split('/')[-1]:
                # remove the file's name if provided (let it scan for h5 files present there)
                custom_data_path = os.path.dirname(custom_data_path)

            self.custom_data_path = os.path.abspath(custom_data_path)
            print("Importing files from {}".format(self.custom_data_path))

            # Set a custom repo to avoid damaging ours
            self.cc_root= os.path.join(os.getcwd(), "cc_repo")

            if os.path.exists(self.cc_root):
                print("\nWARNING--> CC repo {} exists, importing H5 files will add signatures into it\n".format(self.cc_root))

            else:            
                print("\n---> Creating custom repo at {}\n".format(self.cc_root))
                
                try:
                    original_umask = os.umask(0)
                    os.makedirs(self.cc_root, 0o775)
                    os.umask(original_umask)

                except Exception as e:
                    print("Problem in creating cc_repo: {}".format(e))

            # Create  the cc_repo directory structure and symbolic link to files
            self.import_h5()


        # If non-existing CC_root
        if not os.path.isdir(self.cc_root):
            self.__log.warning("Empty root directory, creating dataset dirs")

            for molset in self._basic_molsets:
                for dataset in Dataset.get():
                    ds = dataset.dataset_code
                    new_dir = os.path.join(self.cc_root, molset, ds[:1], ds[:2], ds)
                    self._datasets.add(ds)
                    self.__log.debug("Creating %s", new_dir)
                    original_umask = os.umask(0)
                    os.makedirs(new_dir, 0o775)
                    os.umask(original_umask)
        else:
            # if the directory exists get molsets and datasets
            # NS: also valid for imported h5 datasets
            paths = glob(os.path.join(self.cc_root, '*', '*', '*', '*', '*', 'sign*.h5'))
            self._molsets = set(x.split('/')[-6] for x in paths)
            self._datasets = set(x.split('/')[-3] for x in paths)

        # In case 
        self._molsets = sorted(list(self._molsets))
        self._datasets = [x for x in sorted(list(self._datasets)) if not x.endswith('000')]

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
        """Iterator on Chemical Checker exemplary datasets."""
        for dataset in self.coordinates:
            yield dataset + '.001'

    @cached_property
    def universe(self):
        """Get the list of molecules in the CC universe."""
        universe = set()
        for ds in self.datasets_exemplary():
            s1 = self.get_signature('sign1', 'full', ds)
            universe.update(s1.unique_keys)
        return sorted(list(universe))

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
            molset(str): Filter for the moleculeset e.g. 'full' or 'reference'
            dataset(str) Filter for the dataset e.g. A1.001
            signature(str): Filter for signature type e.g. 'sign1'
        Returns:
            Nested dictionary with molset, dataset and list of signatures
        """
        paths = glob(os.path.join(self.cc_root, molset, '*', '*', dataset, signature + '/*.h5'))
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
        data = DataFactory.make_data(cctype, signature_path, dataset_code, *args, **kwargs)
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
            dataset_code(str): The dataset code of the Chemical Checker.
        Returns:
            datafile(str): The name of the file where the data in pairs is
                saved.
        """
        prepro = Preprocess(sign.signature_path, sign.dataset)
        prepro.fit()

        return prepro.data_path

    def sign_name(self, sign):
        """Get a signature name (e.g. 'B1.001-sign1-full')"""
        folds = sign.data_path.split("/")
        cctype = folds[-2]
        dataset = folds[-3]
        molset = folds[-6]
        return "%s_%s_%s" % (dataset, cctype, molset)

    def signature(self, dataset, cctype):
        return self.get_signature(cctype=cctype, molset="full", dataset_code=dataset)

    def diagnosis(self, sign, save=True, plot=True, overwrite=False, n=10000):
        return self.get_diagnosis(sign=sign, save=save, plot=plot,
                                  overwrite=overwrite, n=n)

    def import_h5(self):
        """ NS. Recovers h5 files from a given custom directory 
            and creates links to them in a CC skeleton arborescence
        """


        h5files= glob(os.path.join(self.custom_data_path, "*.h5"))
        if len(h5files) == 0:
            raise Exception("No h5 files found in {}".format(self.custom_data_path))

        available_files= ", ".join([os.path.basename(f) for f in h5files])
        print("Found h5 files {}: in {}".format(available_files, self.custom_data_path))

        # check the format of the imported info data
        formatDC= re.compile(r"[A-Z]\d\.\d\d\d")  # dataset code (ex: A1.001)
        formatCCTYPE= re.compile(r"sign\d")
        formatMolset= re.compile(r"(full|reference)", re.IGNORECASE)

        formatDict= dict(dataset_code=formatDC, cctype=formatCCTYPE, molset=formatMolset) # mapping info and required format

        def filter_dataset(path2h5file):
            """ returns a tuple of the type ('full', 'A', 'A1', 'A1.001', 'sign3', path_to_h5file') or None if something's wrong"""

            out =[]
            with h5py.File(path2h5file, 'a') as ccfile:

                # check if the required info is presents in the h5 file attrs dict
                # iterates over ('dataset_code', 'cctype', 'molset') and the required format for each of them
                for requiredKey, requiredFormat in formatDict.items(): 
                    if requiredKey  not in  ccfile.attrs:
                        print("Attribute {} cannot be retrieved from {}, skipping this file".format(requiredKey, ccfile))
                        return None

                    else:
                        # check the format of the provided info
                        if requiredFormat.match(ccfile.attrs[requiredKey]) is None:
                            print("Problem with format", ccfile.attrs[requiredKey])
                            return None

                #-------Now that the format is correct, output the info
                # so that we just have to iterate over it to create the directory substructure
                out.append(ccfile.attrs['molset'].lower())           # full or reference
                out.append(ccfile.attrs['dataset_code'][0])  # i.e A
                out.append(ccfile.attrs['dataset_code'][:2]) # i.e 1
                out.append(ccfile.attrs['dataset_code'])     # i.e A1.001
                out.append(ccfile.attrs['cctype'].lower())            # i.e sign3
                out.append(path2h5file)


            return tuple(out)




        # Keep only h5 files that contain the required info in the correct format
        h5tuples = [filter_dataset(f) for f in h5files if filter_dataset(f) is not None]

        if len(h5tuples) == 0:
            raise Exception("None of the provided h5 datasets have sufficient info in its attributes! Please ensure myh5file.attrs has the folllowing keys: 'dataset_code', 'cctype', 'molset'")

        # Now creating the cc_repo skeleton
        original_umask = os.umask(0)
        for h5t in h5tuples:

            path2sign=os.path.join(self.cc_root,'/'.join(h5t[:-1]))   # i.e ../../full/A/A1/A1.001/sign3
            print("Attempting to create", path2sign)




            # If the signature already exists then propose to rename it (ex: 00X) or skip it
            skip_signature=False
            while os.path.exists(path2sign):
                print("Signature {} already exists for dataset {}".format(h5t[4], h5t[3]))
                resp = input("Rename it (r) or skip it (any other key)?")

                if resp.lower() != 'r':
                    skip_signature=True
                    break

                else:                   
                    
                    # Check that the user entered the correct format
                    formatok=False
                    while not formatok: 
                        newcode=input("New dataset code? (ex: 002)")

                        # I put A1 because all that matters is the 00x part
                        formatok= formatDict['dataset_code'].match('A1.'+newcode)

                        # True/False easier to deal with than None in this case
                        formatok= True if (formatok is not None) else False
                        if not formatok: print("Bad format, please try again.")

                    newtup= (h5t[0], h5t[1], h5t[2], h5t[2]+'.'+newcode, h5t[4])
                    path2sign= os.path.join(self.cc_root,'/'.join(newtup))   # i.e ../../full/A/A1/A1.001/sign3
                    print("New signature path: {}".format(path2sign))

                        

            if not skip_signature:     
                try:
                    os.makedirs(os.path.join(self.cc_root, path2sign), 0o775)
                    os.symlink(h5t[-1], os.path.join(self.cc_root, path2sign, h5t[-2]+'.h5'))  # symbolic link to the h5 file in the cc_repo as signx.h5

                except Exception as e:
                    os.umask(original_umask)
                    print("Problem in creating the cc custom repo: {}".format(e))

        os.umask(original_umask) # after the loop to be sure




                

    def export(self, destination, signature, h5_filter=None,
               h5_names_map=None, overwrite=False, version=None):
        """Export a signature h5 file to a given path. Which dataset to copy
           can be specified as well as how to rename some dataset.

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

        # NS: Adding metadata so that they can be opened on local instances of the checker:
        attributes = dict(dataset_code=signature.dataset, cctype=signature.cctype, molset=signature.molset)
        if len(dst.attrs) !=3:
            for k,v in attributes.items():
                dst.attrs[k]=v

        src.close()
        dst.close()

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

    def get_diagnosisplot(self):
        from chemicalchecker.util.plot.diagnosticsplot import DiagnosisPlot
        return DiagnosisPlot(cc=self)

    def get_diagnosis(self, sign, save=True, plot=True, overwrite=False,
                      n=10000):
        from chemicalchecker.core.diagnostics import Diagnosis
        return Diagnosis(cc=self, sign=sign, save=save,
                         plot=plot, overwrite=overwrite, n=n)
