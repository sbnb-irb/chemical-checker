"""Molecule representation.

A simple class to inspect small molecules, easily convert between different
identifier, and ultimately find if signatures available.
"""
import os
import numpy as np
import collections
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from chemicalchecker.util import logged
from .signature_data import DataSignature
from chemicalchecker.util.psql import psql
from chemicalchecker.util.keytype import KeyTypeDetector
from chemicalchecker.util.parser.converter import Converter


@logged
class Molset(object):
    """Molset class.

    Given a CC instance, provides access to features of a input set of
    molecules for one or more dataset of interest.
    The data is organized in a DataFrame which will be annotated with
    observed and/or predicted features by simple kNN inference.
    """

    def __init__(self, cc, molecules, mol_type=None, add_image=True):
        """Initialize a Mol instance.

        Args:
            cc (Chemicalchecker): Chemical Checker instance.
            molecules (list): A list of molecules in homogenous format
               'mol_type'.
            mol_type (str): Type of identifier options are 'inchi',
               'smiles' or 'name'. if 'name' is used we query externally
               (cactus.nci and pubchem) and some molecules might be missing.
               We will immediately add other identifiers to the DataFrame
               if not present yet.
            mol_col (str): The name of the columns in the DataFrame.
            add_image (bool): If True a molecule image is added
        """
        if mol_type is None:
            mol_type = KeyTypeDetector.type(molecules[0])
            if mol_type is None:
                self.__log.warning(
                    "Molecule '%s' not recognized as either"
                    " 'inchikey', 'inchi' or 'smiles'. "
                    "Considering type as 'name'." %
                    molecules[0])
                mol_type = 'name'
        self.cc = cc
        molecules = sorted([x.strip() for x in set(molecules)])
        self.__log.info('%s unique molecules provided' % len(molecules))
        self.df = pd.DataFrame()
        if mol_type.lower() == 'inchi':
            mol_col = 'InChI'
        elif mol_type.lower() == 'smiles':
            mol_col = 'SMILES'
        elif mol_type.lower() == 'name':
            mol_col = 'Name'
        elif mol_type.lower() == 'inchikey':
            mol_col = 'InChIKey'
        else:
            raise Exception("Molecule type '%' not supported" % mol_type)
        self.df[mol_col] = molecules
        self.conv = Converter()

        if mol_type.lower() == 'name':
            name_inchi = self.get_name_inchi_map(molecules)
            self.df['InChI'] = self.df[mol_col].map(name_inchi)
            self._add_inchikeys()
            self._add_smiles()
            self.df = self.df[['Name', 'InChIKey', 'InChI', 'SMILES']]
        elif mol_type.lower() == 'inchikey':
            ink_inchi = self.get_inchikey_inchi_map(molecules)
            self.df['InChI'] = self.df[mol_col].map(ink_inchi)
            self._add_smiles()
            self.df = self.df[['InChIKey', 'InChI', 'SMILES']]
        elif mol_type.lower() == 'inchi':
            self._add_inchikeys()
            self._add_smiles()
            self.df = self.df[['InChIKey', 'InChI', 'SMILES']]
        elif mol_type.lower() == 'smiles':
            self._add_inchi(mol_col)
            self._add_inchikeys()
            self.df = self.df[['InChIKey', 'InChI', 'SMILES']]
        self.inchikeys = self.df[self.df['InChIKey']
                                 != '']['InChIKey'].tolist()
        self.df = self.df.sort_values('InChIKey')
        self.df.reset_index(inplace=True, drop=True)
        if add_image:
            from rdkit.Chem import PandasTools
            self.df.fillna('', inplace=True)
            PandasTools.AddMoleculeColumnToFrame(
                self.df, smilesCol='SMILES', molCol='Structure')

    def get_name_inchi_map(self, names):
        cpd_inchi = dict()
        for cpd in tqdm(names, desc='getting Name-InChI map'):
            try:
                inchi = self.conv.chemical_name_to_inchi(cpd)
                cpd_inchi[cpd] = inchi
            except Exception as ex:
                self.__log.warning(str(ex))
        return cpd_inchi

    def get_inchikey_inchi_map(self, inks):
        ink_inchi = dict()
        for ink in tqdm(inks, desc='getting InChIKey-InChI map'):
            try:
                inchi = self.conv.inchikey_to_inchi(ink)[0]['standardinchi']
            except Exception:
                self.__log.warning('%s has no InChI' % ink)
                continue
            ink_inchi[ink] = inchi
        return ink_inchi

    def _add_inchikeys(self):
        if 'InChIKey' not in self.df.columns:
            self.df['InChIKey'] = self.df['InChI'].dropna().apply(
                self.conv.inchi_to_inchikey)

    def _add_smiles(self):
        if 'SMILES' not in self.df.columns:
            self.df['SMILES'] = self.df['InChI'].dropna().apply(
                self.conv.inchi_to_smiles)

    def _add_inchi(self, mol_col):
        if 'InChI' not in self.df.columns:
            self.df['InChI'] = self.df[mol_col].dropna().apply(
                lambda x: self.conv.smiles_to_inchi(x)[1])

    def annotate(self, dataset_code, shorten_dscode=True,
                 include_features=False, feature_map=None,
                 include_values=False, features_from_raw=False,
                 filter_values=True):
        """Annotate the DataFrame with features fetched CC spaces.

        The minimal annotation if whether the molecule is present or not in the
        dataset of interest. The features, values and predictions are
        optionally available.  Features and values are fetched from the raw
        preprocess file of a given space (before dropping any data). 
        The prediction is based on NN at the sign4 level and depends on
        several parameters: 1) the applicability threshold for considering the 
        sign4 as reliable, 2) the p-value threshold (on sign4 distance) to
        define the neighbors 3) the number of NN to consider.

        Args:
            dataset_code (str): the CC dataset code: e.g. B4.001.
            include_features (bool): Include features of the molecules from
                their raw preprocess sign0.
            feature_map (dict): A dictionary that will be used to map features
                to a human interpretable format.
            include_values (bool): if True an additional column is added with
                a list of (feature, value) pairs.
            include_prediction (bool): include NN derived predictions.
            mapping_fn (dict): A dictionary performing the mapping between
                features ids and their meaning.
            shorten_dscode (bool): If True get rid of the .001 part of the
                dataset code
            features_from_raw (bool): If True the features are fetched from the
                raw sign0 which includes all features for all molecules. If
                False sign0 features are used, so after the Sanitizer step
                which possibly removes features or molecules.
            filter_values (bool): If True values == 0 are filtered. False is
                recomended when the dataset is continuous data.
        """
        # get raw preprocess data
        s0 = self.cc.signature(dataset_code, 'sign0')
        s0_raw = DataSignature(os.path.join(
            s0.signature_path, 'raw', 'preprocess.h5'))
        incc = np.isin(self.inchikeys, s0_raw.keys)
        mol_ds = sum(incc)
        tot_mol = len(self.inchikeys)
        self.__log.info(
            f'{mol_ds}/{tot_mol} molecules found in {dataset_code}')
        dscode = dataset_code
        if shorten_dscode:
            dscode = dataset_code[:2]
        # get presence in space
        self.df[dscode] = 0
        available_inks = np.array(self.inchikeys)[incc]
        idx_df = self.df[self.df['InChIKey'].isin(available_inks)].index
        self.df.at[idx_df, dscode] = 1
        # get the features if requested
        if include_features:
            if features_from_raw:
                sorted_inks, values = s0_raw.get_vectors(available_inks,
                                                         dataset_name='X')
                feat_ref = s0_raw
            else:
                sorted_inks, values = s0.get_vectors(available_inks)
                feat_ref = s0
            if values is None or len(values) == 0:
                self.__log.warning(f'No features found for {dataset_code}!')
            else:
                ink_feat = defaultdict(list)
                for ink, val in zip(sorted_inks, values):
                    feat_idxs = np.argwhere(val).flatten()
                    ink_feat[ink] = feat_ref.features[feat_idxs].tolist()
                feat_name = dscode + '_features'
                self.df[feat_name] = self.df['InChIKey'].map(ink_feat)
                if feature_map is not None:
                    self.df[feat_name] = self.df[feat_name].apply(
                        lambda x: [feature_map[i] for i in x if isinstance(
                            x, list)]).tolist()
                    self.df[feat_name] = self.df[feat_name].apply(
                        lambda x: [i for i in x if i is not None]).tolist()
        # add the corresponding feature value if required
        if include_values:
            if not include_features:
                if features_from_raw:
                    sorted_inks, values = s0_raw.get_vectors(available_inks,
                                                             dataset_name='X')
                    feat_ref = s0_raw
                else:
                    sorted_inks, values = s0.get_vectors(available_inks)
                    feat_ref = s0
            ink_val = defaultdict(list)
            for ink, val in zip(sorted_inks, values):
                feat_idxs = np.arange(len(val))
                if filter_values:
                    feat_idxs = np.argwhere(val != 0).flatten()
                ink_val[ink] = list(zip(
                    feat_ref.features[feat_idxs].tolist(),
                    val[feat_idxs]))
            val_name = dscode + '_values'
            self.df[val_name] = self.df['InChIKey'].map(ink_val)
            if feature_map is not None:
                self.df[val_name] = self.df[val_name].apply(
                    lambda x: [(feature_map[k], v) for k, v in x if
                               isinstance(x, list)]).tolist()

    def predict(self, dataset_code, shorten_dscode=True,
                applicability_thr_query=0, applicability_thr_nn=0,
                max_nr_nn=1000, pvalue_thr_nn=1e-4, limit_top_nn=1000,
                return_stats=False):
        """Annotate the DataFrame with predicted features based on neighbors.

        In this case we can potentially get annotation for every molecules.
        The molecules are first signaturized (sign4) filtered by applicability
        and then searched against molecules within the CC sign4. 
        Nearest neighbors are selected for each molecule based on user
        specified parameters. Only NN that are found in the sign0 of the space
        are preserved. The annotations of these NN are aggregated (multiple
        strategies are possible) and finally assigned to the molecule.

        Args:
            dataset_code (str): the CC dataset code: e.g. B4.001.
            shorten_dscode (bool): If True get rid of the .001 part of the
                dataset code
            applicability_thr_query (float): Only query with a sign4 
                applicability above this threshold will be searched for 
                neighbors.
            applicability_thr_nn (float): Only neighbors with a sign4 
                applicability above this threshold will be used for features
                inference.
            max_nr_nn (int): Maximum number of neighbors to possibly consider.
            pvalue_thr (float): Filter neighbors based on distance.
            limit_top_nn (int): Only keep topN neighbors for feature 
                predictions.
            return_stats (bool): if True return a dataframe with statistics
                on the NN search filtering steps.
        """
        dscode = dataset_code
        if shorten_dscode:
            dscode = dataset_code[:2]
        # signaturize get sign4
        s4 = self.cc.signature(dataset_code, 'sign4')
        s4_app = dict(zip(s4.keys, s4.get_h5_dataset('applicability').ravel()))
        tmp_pred_file = './tmp.h5'
        query_inks = self.df['InChIKey'].tolist()
        query_inchies = self.df['InChI'].tolist()
        pred = s4.predict_from_string(query_inchies, tmp_pred_file,
                                      keytype='InChI', keys=query_inks)
        pred_sign = pred[:]
        pred_keys = pred.keys
        pred_appl = pred.get_h5_dataset('applicability').ravel()
        os.remove(tmp_pred_file)

        # filter queries by applicability
        appl_mask = pred_appl > applicability_thr_query
        self.__log.info(
            'Filtering %i queries for applicability thr' % sum(~appl_mask))
        query_inks = pred_keys[appl_mask]
        pred_sign = pred_sign[appl_mask]

        # faiss sign4 NN search
        neig4 = self.cc.get_signature('neig4', 'reference', dataset_code)
        nn = neig4.get_kth_nearest(pred_sign, k=max_nr_nn)
        self.__log.info("nn distances shape %s" % (str(nn['distances'].shape)))

        # identify distance threshold based on requested pvalue
        metric = neig4.get_h5_dataset('metric')[0]
        bg_dist = s4.background_distances(metric)
        dst_thr_idx = np.argwhere(bg_dist['pvalue'] == pvalue_thr_nn).ravel()
        dst_thr = bg_dist['distance'][dst_thr_idx][0]
        self.__log.info("bg_distance '%s' thr %.4f" % (metric, dst_thr))

        # precomputing dictionary with the list of significant neighbors for
        # each query speeds up things
        query_nn_dict = defaultdict(list)
        for query_idx, hit_idx in np.argwhere(nn['distances'] < dst_thr):
            query_ink = query_inks[query_idx]
            query_nn_dict[query_ink].append(nn['keys'][query_idx, hit_idx])

        # analyze each query individually
        s4_ref = self.cc.get_signature('sign4', 'reference', dataset_code)
        s0 = self.cc.signature(dataset_code, 'sign0')
        query_nn_dict_map = defaultdict(list)
        stats = list()
        # for rid in tqdm(range(len(query_inks)), desc='get NN'):
        for query_ink, nn_inks in tqdm(query_nn_dict.items(), desc='get NN'):
            # filter by distance, now precomputed
            #query_ink = query_inks[rid]
            #nn_inks = nn['keys'][rid]
            #nn_dists = nn['distances'][rid]
            #len_orig = len(nn_inks)
            #nn_inks = nn_inks[nn_dists < dst_thr]
            len_dist = len(nn_inks)
            # expand neighbors list to include redundant molecules
            nn_inks_mapped_idx = np.isin(s4_ref.mappings[:, 1], nn_inks)
            nn_inks = s4_ref.mappings[:, 0][nn_inks_mapped_idx]
            len_mapp = len(nn_inks)
            # remove the query itself in case is found
            nn_inks = np.delete(nn_inks, np.argwhere(nn_inks == query_ink))
            len_mapp_noself = len(nn_inks)
            # limit to neighbors with good applicability
            nn_inks = [i for i in nn_inks if s4_app[i] > applicability_thr_nn]
            nn_inks = np.array(nn_inks)
            len_app = len(nn_inks)
            # limit to neighbors with sign0
            s0_mask = np.isin(nn_inks, s0.keys)
            nn_inks = nn_inks[s0_mask]
            len_s0 = len(nn_inks)
            # limit to top N neighbors
            nn_inks = nn_inks[:limit_top_nn]
            query_nn_dict_map[query_ink] = nn_inks.tolist()
            len_topn = len(nn_inks)
            stats.append({'query_ink': query_ink, 'distance': len_dist,
                          'full': len_mapp_noself, 'applicability': len_app,
                          'in_sign0': len_s0, 'limit_topN': len_topn})
        stats_df = pd.DataFrame(stats)
        self.df['%s_NN' % dscode] = self.df['InChIKey'].map(query_nn_dict_map)

        # generate table with sign0 annotation for each neighbor found
        # we can recursively use a molset for this purpose!
        all_nn_inks = set.union(*[set(n) for n in query_nn_dict_map.values()])
        all_nn_inks = sorted(list(all_nn_inks))
        conv = Converter()
        inchies = list()
        for ink in tqdm(all_nn_inks, desc='get NN InChI'):
            try:
                inchi = conv.inchikey_to_inchi(ink)[0]['standardinchi']
            except Exception:
                self.__log.warning('%s has no InChI' % ink)
                continue
            inchies.append(inchi)
        nn_molset = Molset(self.cc, inchies, mol_type='inchi')
        nn_molset.annotate(dataset_code, include_features=True,
                           features_from_raw=False)

        # aggregate NN features
        aggregated = defaultdict(list)
        for ink, nn_inks in tqdm(query_nn_dict_map.items(), desc='preds'):
            if len(nn_inks) == 0:
                continue
            neighs = nn_molset.df[nn_molset.df['InChIKey'].isin(nn_inks)]
            features = neighs['%s_features' % dscode].tolist()
            all_feats = set.union(*[set(f) for f in features])
            all_feats = sorted(list(all_feats))
            aggregated[ink] = all_feats
        self.df['%s_predicted' % dscode] = self.df['InChIKey'].map(aggregated)
        if return_stats:
            return stats_df, nn_molset
        else:
            return nn_molset

    @staticmethod
    def get_uniprot_annotation(entries):
        """Get information on Uniprot entries."""
        from bioservices import UniProt
        uniprot = UniProt()
        df = list()
        for up in tqdm(entries):
            try:
                df.append(uniprot.get_df(f'accession:{up}'))
            except Exception:
                print(up, 'NOT FOUND!')
        df = pd.concat(df)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def get_chembl_protein_classes(chembldb='chembl_27'):
        """Get protein class id to name dictionary."""
        R = psql.qstring(
            "SELECT protein_class_id, parent_id, pref_name "
            "FROM protein_classification", chembldb)
        class_names = defaultdict(lambda: None)
        class_names.update({'Class:%i' % r[0]: r[2] for r in R})
        return class_names

    @staticmethod
    def get_chebi(chebi_obo):
        """Get CHEBI id to name dictionary."""
        chebi_dict = defaultdict(str)
        f = open(chebi_obo, "r")
        terms = f.read().split("[Term]\n")
        for term in terms[1:]:
            term = term.split("\n")
            chebi_id = term[0].split("id: ")[1]
            chebi_name = term[1].split("name: ")[1]
            chebi_dict[chebi_id] = chebi_name
        f.close()
        return chebi_dict


@logged
class Mol(object):
    """Mol class.

    Given a CC instance, provides access to signatures of a input molecule. 
    """

    def __init__(self, cc, mol_str, str_type=None):
        """Initialize a Mol instance.

        Args:
            cc: Chemical Checker instance.
            mol_str: Compound identifier (e.g. SMILES string)
            str_type: Type of identifier ('inchikey', 'inchi' and 'smiles' are
                accepted) if 'None' we do our best to guess.
        """
        if str_type is None:
            str_type = KeyTypeDetector.type(mol_str)
            if str_type is None:
                raise Exception(
                    "Molecule '%s' not recognized as valid format"
                    " (options are: 'inchikey', 'inchi' and 'smiles')" %
                    mol_str)
        conv = Converter()
        if str_type == "inchikey":
            self.inchikey = mol_str
            self.inchi = conv.inchikey_to_inchi(
                self.inchikey)[0]["standardinchi"]
        if str_type == "inchi":
            self.inchi = mol_str
            self.inchikey = conv.inchi_to_inchikey(self.inchi)
        if str_type == "smiles":
            self.inchikey, self.inchi = conv.smiles_to_inchi(mol_str)
        self.smiles = conv.inchi_to_smiles(self.inchi)
        self.mol = conv.inchi_to_mol(self.inchi)
        self.cc = cc

    def isin(self, cctype, dataset_code, molset="full"):
        """Check if the molecule is in the dataset of interest"""
        sign = self.cc.get_signature(cctype, molset, dataset_code)
        try:
            keys = set(sign.keys)
        except Exception:
            keys = set(sign.row_keys)
        return self.inchikey in keys

    def signature(self, cctype, dataset_code, molset="full"):
        """Check if the molecule is in the dataset of interest"""
        sign = self.cc.get_signature(cctype, molset, dataset_code)
        return sign[self.inchikey]

    def report_available(self, dataset_code="*", cctype="*", molset="full"):
        """Check in what datasets the key is present"""
        available = self.cc.report_available(molset, dataset_code, cctype)
        d0 = {}
        for molset, datasets in available.items():
            d1 = collections.defaultdict(list)
            for dataset, cctypes in datasets.items():
                for cctype in cctypes:
                    if self.isin(cctype, dataset, molset):
                        d1[dataset] += [cctype]
            if d1:
                d0[molset] = dict((k, v) for k, v in d1.items())
        return d0

    def show(self):
        """Simply display the molecule in a Jupyter notebook"""
        from rdkit.Chem.Draw import IPythonConsole
        return self.mol
