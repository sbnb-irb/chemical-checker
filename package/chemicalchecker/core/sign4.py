"""Signature type 4.

First attempt on Siamese network to derive a new signature.
"""
import os
import h5py
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from functools import partial
from scipy.spatial.distance import pdist
from sklearn.preprocessing import robust_scale
from scipy.stats import pearsonr, norm, rankdata

from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import QuantileTransformer

from .signature_base import BaseSignature
from .signature_data import DataSignature

from chemicalchecker.util.plot import Plot
from chemicalchecker.util import logged
from chemicalchecker.util.splitter import Traintest, NeighborTripletTraintest
from chemicalchecker.util.splitter import NeighborErrorTraintest
from chemicalchecker.tool.siamese import SiameseTriplets


@logged
class sign4(BaseSignature, DataSignature):
    """Signature type 4 class."""

    def __init__(self, signature_path, dataset, **params):
        """Initialize the signature.

        Args:
            signature_path(str): The signature root directory.
            dataset(`Dataset`): `chemicalchecker.database.Dataset` object.
            params(): Parameters, expected keys are:
                * 'sign2' for learning based on sign2
                * 'sign0' for learning based on sign0 (Morgan Fingerprint)
                * 'sign0_conf' for learning confidences based on MFP
                * 'error' for learning error in predictions
        """
        # Calling init on the base class to trigger file existence checks
        BaseSignature.__init__(self, signature_path,
                               dataset, **params)
        # generate needed paths
        self.data_path = os.path.join(self.signature_path, 'sign4.h5')
        DataSignature.__init__(self, self.data_path)
        self.model_path = os.path.join(self.signature_path, 'models')
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        self.stats_path = os.path.join(self.signature_path, 'stats')
        if not os.path.isdir(self.model_path):
            os.makedirs(self.stats_path)
        # assign dataset
        self.dataset = dataset
        # logging
        self.__log.debug('dataset: %s', dataset)
        self.__log.debug('data_path: %s', self.data_path)
        self.__log.debug('model_path: %s', self.model_path)
        self.__log.debug('stats_path: %s', self.stats_path)
        # get parameters or default values
        self.params = dict()
        # parameters to learn from sign2
        default_sign2 = {
            'epochs': 1,
            'batch_size': 1000,
            'augment_fn': None,
            'num_triplets': 1000000,
            'augment_scale': 1,
            'augment_kwargs': {
                'dataset': [dataset]
            },
        }
        default_sign2.update(params.get('sign2', {}))
        self.params['sign2'] = default_sign2
        # parameters to learn from sign0
        default_sign0 = {
        }
        default_sign0.update(params.get('sign0', {}))
        self.params['sign0'] = default_sign0
        # parameters to learn confidence from sign0
        default_sign0_conf = {
        }
        default_sign0_conf.update(params.get('sign0_conf', {}))
        self.params['sign0_conf'] = default_sign0_conf
        # predictors and parameters to learn error
        n_jobs = params.get('cpu', 4)
        default_err = {
            'LinearRegression': LinearRegression(),
            'LinearSVR': LinearSVR(),
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=5,
                n_jobs=n_jobs
            ),
            'KNeighborsRegressor': KNeighborsRegressor(
                n_jobs=n_jobs),
            'GaussianProcessRegressor': GaussianProcessRegressor(),
            'MLPRegressor': MLPRegressor()
        }
        default_err.update(params.get('error', {}))
        self.params['error'] = default_err

    @staticmethod
    def save_sign2_universe(sign2_list, destination):
        """Create a file with all signatures 2 for each molecule in the CC.

        Args:
            sign2_list(list): List of signature 2 objects to learn from.
            destination(str): Path where the H5 is saved.
        """
        # get sorted universe inchikeys and CC signatures
        sign4.__log.info("Generating signature 2 universe matrix.")
        inchikeys = set()
        for sign in sign2_list:
            inchikeys.update(sign.unique_keys)
        inchikeys = sorted(list(inchikeys))
        tot_inks = len(inchikeys)
        tot_ds = len(sign2_list)
        # build matrix stacking horizontally signature
        if os.path.isfile(destination):
            sign4.__log.warning("Skipping as destination %s already exists." %
                                destination)
            return
        with h5py.File(destination, "w") as fh:
            fh.create_dataset('x_test', (tot_inks, 128 * tot_ds),
                              dtype=np.float32)
            for idx, sign in enumerate(sign2_list):
                sign4.__log.info("Fetching from %s" % sign.data_path)
                # including NaN we have the correct number of molecules
                _, vectors = sign.get_vectors(inchikeys, include_nan=True)
                fh['x_test'][:, idx * 128:(idx + 1) * 128] = vectors
                del vectors

    '''
    @staticmethod
    def complete_sign2_universe(sign2_list, sign2_self, partial_universe,
                                sign2_universe):

        # get sorted universe inchikeys and CC signatures
        sign4.__log.info("Completing signature 2 universe matrix.")
        inchikeys = set()
        for sign in sign2_list:
            if sign == sign2_self:
                continue
            inchikeys.update(sign.unique_keys)
        inchikeys = sorted(list(inchikeys))

        space_inks = sign2_self.unique_keys - set(inchikeys)
        tot_inks = len(inchikeys)
    '''

    @staticmethod
    def save_sign2_coverage(sign2_list, destination):
        """Create a file with all signatures 2 coverage of molecule in the CC.

        Args:
            sign2_list(list): List of signature 2 objects to learn from.
            destination(str): Path where the H5 is saved.
        """
        # get sorted universe inchikeys and CC signatures
        sign4.__log.info("Generating signature 2 coverage matrix.")
        inchikeys = set()
        for sign in sign2_list:
            inchikeys.update(sign.unique_keys)
        inchikeys = sorted(list(inchikeys))
        tot_inks = len(inchikeys)
        tot_ds = len(sign2_list)
        sign4.__log.info("Saving coverage for %s dataset and %s molecules." %
                         (tot_ds, tot_inks))
        # build matrix stacking horizontally signature
        if os.path.isfile(destination):
            sign4.__log.warning("Skipping as destination %s already exists." %
                                destination)
            return
        with h5py.File(destination, "w") as fh:
            fh.create_dataset('x_test', (tot_inks, tot_ds), dtype=np.float32)
            for idx, sign in enumerate(sign2_list):
                sign4.__log.info("Fetching from %s" % sign.data_path)
                # including NaN we have the correct number of molecules
                coverage = np.isin(
                    list(inchikeys), sign.keys, assume_unique=True)
                sign4.__log.info("%s has %s Signature 2." %
                                 (sign.dataset, np.count_nonzero(coverage)))
                fh['x_test'][:, idx:(idx + 1)] = np.expand_dims(coverage, 1)

    def save_sign2_matrix(self, sign2_list, destination):
        """Save matrix of pairs of horizontally stacked signature 2.

        This is the matrix for training the signature 4. It is defined for all
        molecules or which we have a signature 2 in the current space.
        The X is the collections of signature 2 from other spaces
        horizontally stacked (and NaN filled) for 2 molecules.
        The Y is 0/1 depending whether the two signatures are similar as
        derived from the nearest neighbors.

        Args:
            sign2_list(list): List of signature 2 objects to learn from.
            destination(str): Path where to save the matrix (HDF5 file).
        """
        self.__log.debug('Saving sign2 traintest to: %s' % destination)
        # get current space on which we'll train
        ref_dimension = self.sign2_self.shape[1]
        feat_shape = (self.sign2_self.shape[0],
                      ref_dimension * len(self.src_datasets))
        with h5py.File(destination, 'w') as hf:
            hf.create_dataset('x', feat_shape, dtype=np.float32)
            # for each dataset fetch signatures for the molecules of current
            # space, if missing we add NaN.
            for idx, (ds, sign) in enumerate(zip(self.src_datasets, sign2_list)):
                _, signs = sign.get_vectors(
                    self.sign2_self.keys, include_nan=True)
                col_slice = slice(ref_dimension * idx,
                                  ref_dimension * (idx + 1))
                hf['x'][:, col_slice] = signs
                available = np.isin(list(self.sign2_self.keys), sign.keys)
                self.__log.info('%s shared molecules between %s and %s',
                                sum(available), self.dataset, ds)
                del signs

    def learn_sign2(self, params, reuse=True, suffix=None, evaluate=True):
        """Learn the signature 3 model.

        This method is used twice. First to evaluate the performances of the
        Siamese model. Second to train the final model on the full set of data.

        Args:
            params(dict): Dictionary with algorithm parameters.
            reuse(bool): Whether to reuse intermediate files (e.g. the
                aggregated signature 2 matrix).
            suffix(str): A suffix for the Siamese model path (e.g.
                'sign4/models/siamese_<suffix>').
            evaluate(bool): Whether we are performing a train-test split and
                evaluating the performances (N.B. this is required for complete
                confidence scores)
        """
        # get params and set folder
        if suffix:
            siamese_path = os.path.join(self.model_path, 'siamese_%s' % suffix)
            traintest_file = os.path.join(self.model_path,
                                          'traintest_%s.h5' % suffix)
            params['traintest_file'] = params.get(
                'traintest_file', traintest_file)
        else:
            siamese_path = os.path.join(self.model_path, 'siamese')
        if 'model_dir' in params:
            siamese_path = params.get('model_dir')
        if not reuse or not os.path.isdir(siamese_path):
            os.makedirs(siamese_path)
        # generate input matrix
        sign2_matrix = os.path.join(self.model_path, 'train.h5')
        if not reuse or not os.path.isfile(sign2_matrix):
            self.save_sign2_matrix(self.sign2_list, sign2_matrix)
        # if evaluating, perform the train-test split
        traintest_file = params.get('traintest_file')
        if evaluate:
            num_triplets = params.get('num_triplets', 1e6)
            if not reuse or not os.path.isfile(traintest_file):
                X = DataSignature(sign2_matrix)
                NeighborTripletTraintest.create(
                    X, traintest_file, self.neig_sign,
                    split_names=['train', 'test'],
                    split_fractions=[.8, .2],
                    suffix=suffix,
                    num_triplets=num_triplets)
        else:
            num_triplets = params.get('num_triplets', 1e6)
            if not reuse or not os.path.isfile(traintest_file):
                X = DataSignature(sign2_matrix)
                NeighborTripletTraintest.create(
                    X, traintest_file, self.neig_sign,
                    split_names=['train'],
                    split_fractions=[1.0],
                    suffix=suffix,
                    num_triplets=num_triplets)
        # update the subsampling parameter
        if 'augment_kwargs' in params:
            ds = params['augment_kwargs']['dataset']
            dataset_idx = np.argwhere(np.isin(self.src_datasets, ds)).flatten()
            params['augment_kwargs']['dataset_idx'] = self.dataset_idx
            # compute probabilities for subsampling
            p_nr, p_keep = subsampling_probs(self.sign2_coverage, dataset_idx)
            params['augment_kwargs']['p_nr'] = p_nr
            params['augment_kwargs']['p_keep'] = p_keep
        # init siamese NN
        siamese = SiameseTriplets(siamese_path,
                                  evaluate=evaluate,
                                  **params)
        self.__log.debug('Siamese training on %s' % traintest_file)
        siamese.fit()
        self.__log.debug('model saved to %s' % siamese_path)
        # save known NN faiss index
        known_mask = np.arange(min(100000, X.info_h5['x'][0]))
        known_x = X.get_h5_dataset('x', mask=known_mask)
        self.save_known_distributions(siamese, known_x)
        # evaluate distance distributions
        self.plot_validations(siamese, dataset_idx, traintest_file)
        # when evaluating we update the nr of epoch to avoid
        # overfitting during final
        if evaluate:
            # update the parameters with the new nr_of epochs
            self.params['sign2']['epochs'] = siamese.last_epoch

    def save_known_distributions(self, siamese, known_x):
        try:
            import faiss
        except ImportError as err:
            raise err
        # save known neighbors faiss index
        known_pred = siamese.predict(known_x)
        neig_index = faiss.IndexFlatL2(known_pred.shape[1])
        neig_index.add(known_pred.astype(np.float32))
        known_neig_file = os.path.join(self.model_path, 'known_neig.index')
        faiss.write_index(neig_index, known_neig_file)
        # predict with final model
        preds = siamese.predict(known_x)
        # measure average distance from known
        distances = np.mean(neig_index.search(preds, 50), axis=1).flatten()
        intensities, stddevs, consensus = self.conformal_prediction(
            siamese, known_x)
        # get prediction with only self
        self_preds = siamese.predict(mask_keep([self.dataset_idx], known_x))
        # get correlation between prediction and only self predictions
        accuracies = row_wise_correlation(preds, self_preds, scaled=True)
        know_dist_file = os.path.join(self.model_path, 'known_dist.h5')
        with h5py.File(know_dist_file, "w") as hf:
            hf.create_dataset('stddev', data=stddevs, dtype=np.float32)
            hf.create_dataset('intensity', data=intensities, dtype=np.float32)
            hf.create_dataset('consensus', data=consensus, dtype=np.float32)
            hf.create_dataset('distance', data=distances, dtype=np.float32)
            hf.create_dataset('accuracy', data=accuracies, dtype=np.float32)

    def conformal_prediction(self, siamese, featues, nan_pred=None):
        # reference prediction (based on no information)
        if nan_pred is None:
            nan_feat = np.full((1, featues.shape[1]), np.nan, dtype=np.float32)
            nan_pred = siamese.predict(nan_feat)
        # draw prediction with sub-sampling (dropout)
        dropout_fn = partial(subsample, dataset_idx=self.dataset_idx)
        samples = siamese.predict(featues, dropout_fn=dropout_fn,
                                  dropout_samples=10)
        # summarize the predictions as consensus
        consensus = np.mean(samples, axis=1)
        # zeros input (no info) as intensity reference
        centered = consensus - nan_pred
        # measure the intensity (mean of absolute comps)
        intensities = np.mean(np.abs(centered), axis=1).flatten()
        # summarize the standard deviation of components
        stddevs = np.mean(np.std(samples, axis=1), axis=1).flatten()
        return intensities, stddevs, consensus

    def plot_validations(self, siamese, dataset_idx, chunk_size=10000,
                         limit=1000, dist_limit=1000):

        def no_mask(idxs, x1_data):
            return x1_data

        import matplotlib.pyplot as plt
        import seaborn as sns
        import itertools

        mask_fns = {
            'ALL': partial(no_mask, dataset_idx),
            'NOT-SELF': partial(mask_exclude, dataset_idx),
            'ONLY-SELF': partial(mask_keep, dataset_idx),
        }

        # get molecules where space is available
        cov = DataSignature(self.sign2_coverage).get_h5_dataset('V')
        known_idxs = np.argwhere(cov[:, dataset_idx[0]] == 1).flatten()
        unknown_idxs = np.argwhere(cov[:, dataset_idx[0]] == 0).flatten()
        self.__log.info('VALIDATION: total %s known, %s unknown' %
                        (known_idxs.shape[0], unknown_idxs.shape[0]))
        full_x = DataSignature(self.sign2_universe)
        known = full_x.get_h5_dataset('x_test', mask=known_idxs[:limit])
        unknown = full_x.get_h5_dataset('x_test', mask=unknown_idxs[:limit])

        # predict
        self.__log.info('VALIDATION: Predicting.')
        known_pred = dict()
        unknown_pred = dict()
        for name, mask_fn in mask_fns.items():
            known_pred[name] = siamese.predict(mask_fn(known))
            if name == 'ONLY-SELF':
                unknown_pred[name] = []
            else:
                unknown_pred[name] = siamese.predict(mask_fn(unknown))

        self.__log.info(known_pred['ALL'][0])
        self.__log.info(unknown_pred['ALL'][0])

        self.__log.info('VALIDATION: Plot correlations.')
        fig, axes = plt.subplots(
            1, 3, sharex=True, sharey=False, figsize=(10, 3))
        combos = itertools.combinations(mask_fns, 2)
        for ax, (n1, n2) in zip(axes.flatten(), combos):
            corrs = row_wise_correlation(known_pred[n1], known_pred[n2])
            scaled_corrs = row_wise_correlation(
                known_pred[n1], known_pred[n2], scaled=True)
            sns.distplot(corrs, label='%s %s' % (n1, n2), ax=ax)
            sns.distplot(scaled_corrs, label='scaled %s %s' % (n1, n2), ax=ax)
            ax.legend()
        fname = 'known_unknown_correlations.png'
        plot_file = os.path.join(siamese.model_dir, fname)
        plt.savefig(plot_file)
        plt.close()

        for metric in ['euclidean', 'cosine']:
            self.__log.info('VALIDATION: Plot %s distances.' % metric)
            fig, axes = plt.subplots(
                1, 3, sharex=True, sharey=True, figsize=(10, 3))
            for ax, name in zip(axes.flatten(), mask_fns):
                ax.set_title(name)
                dist_known = pdist(known_pred[name][:dist_limit],
                                   metric=metric)
                sns.distplot(dist_known, label='known', ax=ax)
                if len(unknown_pred[name]) > 0:
                    dist_unknown = pdist(unknown_pred[name][:dist_limit],
                                         metric=metric)
                    sns.distplot(dist_unknown, label='unknown', ax=ax)
                ax.legend()
            fname = 'known_unknown_dist_%s.png' % metric
            plot_file = os.path.join(siamese.model_dir, fname)
            plt.savefig(plot_file)
            plt.close()

    def plot_validations_2(self, siamese, dataset_idx, traintest_file,
                           chunk_size=10000, limit=1000, dist_limit=1000):
        def mask_keep(idxs, x1_data):
            # we will fill an array of NaN with values we want to keep
            x1_data_transf = np.zeros_like(x1_data, dtype=np.float32) * np.nan
            for idx in idxs:
                # copy column from original data
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x1_data_transf[:, col_slice] = x1_data[:, col_slice]
            # keep rows containing at least one not-NaN value
            #not_nan = np.isfinite(x1_data_transf).any(axis=1)
            #x1_data_transf = x1_data_transf[not_nan]
            return x1_data_transf

        def mask_exclude(idxs, x1_data):
            x1_data_transf = np.copy(x1_data)
            for idx in idxs:
                # set current space to nan
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x1_data_transf[:, col_slice] = np.nan
            # drop rows that only contain NaNs
            #not_nan = np.isfinite(x1_data_transf).any(axis=1)
            #x1_data_transf = x1_data_transf[not_nan]
            return x1_data_transf

        def no_mask(idxs, x1_data):
            return x1_data

        from MulticoreTSNE import MulticoreTSNE
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import seaborn as sns
        import itertools
        from sklearn.preprocessing import robust_scale
        from sklearn.linear_model import LogisticRegressionCV
        import faiss
        from chemicalchecker.util.remove_near_duplicates import RNDuplicates
        from tqdm import tqdm

        mask_fns = {
            'ALL': partial(no_mask, dataset_idx),
            'NOT-SELF': partial(mask_exclude, dataset_idx),
            'ONLY-SELF': partial(mask_keep, dataset_idx),
        }

        # get molecules where space is available
        if self.sign2_coverage is None:
            self.__log.info('VALIDATION: fetching Xs.')
            known = list()
            unknown = list()
            enough_known = False
            enough_unknown = False
            with h5py.File(self.sign2_universe, "r") as features:
                # read input in chunks
                for idx in range(0, features['x_test'].shape[0], chunk_size):
                    chunk = slice(idx, idx + chunk_size)
                    feat = features['x_test'][chunk]
                    nan_ds = np.isnan(feat[:, dataset_idx[0] * 128])
                    if len(known) < limit:
                        chunk_known = feat[~nan_ds]
                        if len(chunk_known) == 0:
                            continue
                        known.extend(chunk_known)
                    else:
                        enough_known = True
                    if len(unknown) < limit:
                        chunk_unknown = feat[nan_ds]
                        if len(chunk_unknown) == 0:
                            continue
                        unknown.extend(chunk_unknown)
                    else:
                        enough_unknown = True
                    if enough_known and enough_unknown:
                        break
            known = np.vstack(known[:limit])
            unknown = np.vstack(unknown[:limit])
        else:
            # get known and unknown idxs
            cov = DataSignature(self.sign2_coverage).get_h5_dataset('V')
            unknown_idxs = np.argwhere(cov[:, dataset_idx[0]] == 0).flatten()

            if len(unknown_idxs) > 50000:
                unknown_idxs = np.random.choice(
                    len(unknown_idxs), 50000, replace=False)
            # get train and test idxs
            all_inchikeys = self.get_universe_inchikeys()
            traintest = DataSignature(traintest_file)
            train_inks = traintest.get_h5_dataset('keys_train')[:4000]
            test_inks = traintest.get_h5_dataset('keys_test')[:1000]
            train_idxs = np.argwhere(
                np.isin(all_inchikeys, train_inks)).flatten()
            test_idxs = np.argwhere(
                np.isin(all_inchikeys, test_inks)).flatten()
            train_idxs.sort()
            test_idxs.sort()
            tr_nn_idxs = np.argwhere(
                np.isin(self.neig_sign.keys, train_inks)).flatten()
            ts_nn_idxs = np.argwhere(
                np.isin(self.neig_sign.keys, test_inks)).flatten()

        # 6.6 GB - 3.4GB = 3.2 GB
        # predict FIXME we don't really need all the predictions
        self.__log.info('VALIDATION: Predicting.')
        preds = dict()
        pred_signs_name = os.path.join(self.model_path, 'pred_signs.pkl')

        if not os.path.isfile(pred_signs_name):
            for name, mask_fn in mask_fns.items():
                with h5py.File(self.sign2_universe, "r") as features:
                    # read input in chunks
                    preds[name] = np.zeros((features['x_test'].shape[0], 128))
                    for idx in tqdm(range(0, features['x_test'].shape[0], chunk_size)):
                        chunk = slice(idx, idx + chunk_size)
                        feat = features['x_test'][chunk]
                        preds[name][chunk] = siamese.predict(mask_fn(feat))

            with open(pred_signs_name, 'wb') as f:
                pickle.dump(preds, f)
        else:
            with open(pred_signs_name, 'rb') as f:
                preds = pickle.load(f)

        # we only need train test and a bunch of unknowns
        preds.setdefault('tr', {})
        preds.setdefault('ts', {})
        preds.setdefault('unk', {})
        preds['tr']['ALL'] = preds['ALL'][train_idxs].astype(np.float32)
        preds['ts']['ALL'] = preds['ALL'][test_idxs].astype(np.float32)
        preds['unk']['ALL'] = preds['ALL'][unknown_idxs].astype(np.float32)
        del preds['ALL']
        preds['tr'][
            'NOT-SELF'] = preds['NOT-SELF'][train_idxs].astype(np.float32)
        preds['ts'][
            'NOT-SELF'] = preds['NOT-SELF'][test_idxs].astype(np.float32)
        del preds['NOT-SELF']
        preds['tr'][
            'ONLY-SELF'] = preds['ONLY-SELF'][train_idxs].astype(np.float32)
        preds['ts'][
            'ONLY-SELF'] = preds['ONLY-SELF'][test_idxs].astype(np.float32)
        del preds['ONLY-SELF']

        # find nearest neighbors for train and test
        neig_pred = faiss.IndexFlatL2(preds['unk']['ALL'].shape[1])
        neig_pred.add(preds['unk']['ALL'])
        unknown_n_dis, unknown_n_idx = neig_pred.search(
            np.vstack([preds['tr']['ALL'], preds['ts']['ALL']]), 5)
        K1_unknown = preds['unk']['ALL'][unknown_n_idx[:, 0]]
        K5_unknown = preds['unk']['ALL'][unknown_n_idx[:, -1]]

        metrics = dict()
        feat_type = 'continuos'
        fname = '%s_known_unknown' % feat_type
        self.__log.info(
            'VALIDATION: Plot correlations %s.' % feat_type)

        fig, axes = plt.subplots(
            1, 3, sharex=True, sharey=True, figsize=(10, 3))
        combos = itertools.combinations(mask_fns, 2)
        for ax, (n1, n2) in zip(axes, combos):
            corrs_train = row_wise_correlation(
                preds['tr'][n1], preds['tr'][n2], scaled=True)
            corrs_test = row_wise_correlation(
                preds['ts'][n1], preds['ts'][n2], scaled=True)
            ax.set_title('Scaled  %s %s' % (n1, n2))
            sns.distplot(corrs_train, ax=ax, label='train')
            sns.distplot(corrs_test, ax=ax, label='test')
            metrics['corr train %s %s' % (n1, n2)] = np.mean(corrs_train)
            metrics['corr test %s %s' % (n1, n2)] = np.mean(corrs_test)
            del corrs_train
            del corrs_test
        plot_file = os.path.join(siamese.model_dir,
                                 '%s_correlations.png' % fname)
        plt.savefig(plot_file)
        plt.close()

        for m in ['euclidean', 'cosine']:
            self.__log.info(
                'VALIDATION: Plot %s distances %s.' % (m, feat_type))
            fig, axes = plt.subplots(
                3, 2, sharex=True, sharey=True, figsize=(8, 10))
            dist_unknown = pdist(preds['unk']['ALL'][:dist_limit], metric=m)
            mean_d_unknown = np.mean(dist_unknown)
            for ax_row, name in zip(axes, ['Train-Train', 'Train-Test', 'Test-Test']):
                if name == 'Train-Train':
                    dist_all = pdist(preds['tr']['ALL'][np.random.choice(
                        len(preds['tr']['ALL']), dist_limit, replace=False)], metric=m)
                    dist_only = pdist(preds['tr']['ONLY-SELF'][np.random.choice(
                        len(preds['tr']['ONLY-SELF']), dist_limit, replace=False)], metric=m)
                    mean_dist = np.mean(dist_all)
                    metrics['%s_trtr_all' % m] = abs(
                        mean_dist - mean_d_unknown)
                    metrics['%s_trtr_only' % m] = abs(
                        mean_dist - np.mean(dist_only))
                elif name == 'Train-Test':
                    known_all = np.vstack(
                        [preds['tr']['ALL'], preds['ts']['ALL']])
                    dist_all = pdist(known_all[np.random.choice(len(known_all), min(
                        dist_limit, len(known_all)), replace=False)], metric=m)
                    del known_all
                    known_only = np.vstack(
                        [preds['tr']['ONLY-SELF'], preds['ts']['ONLY-SELF']])
                    dist_only = pdist(known_only[np.random.choice(len(known_only), min(
                        dist_limit, len(known_only)), replace=False)], metric=m)
                    del known_only
                    mean_dist = np.mean(dist_all)
                    metrics['%s_trts_all' % m] = abs(
                        mean_dist - mean_d_unknown)
                    metrics['%s_trts_only' % m] = abs(
                        mean_dist - np.mean(dist_only))
                else:
                    dist_all = pdist(preds['ts']['ALL'][np.random.choice(len(preds['ts']['ALL']), min(
                        dist_limit, len(preds['ts']['ALL'])), replace=False)], metric=m)
                    dist_only = pdist(preds['ts']['ONLY-SELF'][np.random.choice(len(preds['ts'][
                                      'ONLY-SELF']), min(dist_limit, len(preds['ts']['ALL'])), replace=False)], metric=m)
                    mean_dist = np.mean(dist_all)
                    metrics['%s_tsts' % m] = abs(mean_dist - mean_d_unknown)
                    metrics['%s_tsts_only' % m] = abs(
                        mean_dist - np.mean(dist_only))
                ax_row[0].set_title(name + ' ALL')
                sns.distplot(dist_all, label='known', ax=ax_row[0])
                sns.distplot(dist_unknown, label='unknown', ax=ax_row[0])
                ax_row[0].legend()
                ax_row[1].set_title(name + ' ONLY-SELF')
                sns.distplot(dist_only, ax=ax_row[1])
                del dist_all
                del dist_only
            plot_file = os.path.join(siamese.model_dir,
                                     '%s_dists_%s.png' % (fname, m))
            plt.savefig(plot_file)
            plt.close()

        return True
        self.__log.info('VALIDATION: Plot intensities %s.' % feat_type)
        plt.figure(figsize=(10, 3))
        data = np.array([np.sum(np.abs(preds['tr']['ALL']), axis=1),
                         np.sum(np.abs(preds['ts']['ALL']), axis=1),
                         np.sum(np.abs(
                             np.vstack([preds['tr']['ONLY-SELF'], preds['ts']['ONLY-SELF']])), axis=1),
                         np.sum(np.abs(preds['unk']['ALL']), axis=1),
                         np.sum(np.abs(K1_unknown), axis=1),
                         np.sum(np.abs(K5_unknown), axis=1)])
        plt.title('Intensities', fontsize=20)
        sns.violinplot(data=data, inner="quartile")
        labels = ['tr', 'te', 'ONLY-SELF', 'unk_naive', 'unk_K1', 'unk_K5']
        plt.xticks(range(len(labels)), labels)
        plot_file = os.path.join(
            siamese.model_dir, '%s_intensity.png' % fname)
        plt.savefig(plot_file)
        plt.close()
        del data

        self.__log.info('VALIDATION: Plot NN accuracy %s.' % feat_type)

        def get_nn_matrix(in_data, unk_data=np.array([])):
            NN = faiss.IndexFlatL2(in_data.shape[1])
            NN.add(in_data.astype(np.float32))
            neig_dis, neig_idx = NN.search(in_data.astype(np.float32), 20)
            # remove self
            NN_neig = np.zeros((neig_idx.shape[0], neig_idx.shape[1] - 1))
            NN_dis = np.zeros((neig_idx.shape[0], neig_idx.shape[1] - 1))
            for idx, row in enumerate(neig_idx):
                NN_neig[idx] = row[np.argwhere(row != idx).flatten()][
                    :NN_neig.shape[1]]
                NN_dis[idx] = neig_dis[idx][np.argwhere(row != idx).flatten()][
                    :NN_dis.shape[1]]
            unk_neig_dist = False
            if unk_data.any():
                unk_neig_dist, _ = NN.search(unk_data.astype(np.float32), 5)
            return NN_dis, NN_neig, unk_neig_dist

        def overlap(n1, n2):
            res = list()
            for r1, r2 in zip(n1, n2):
                s1 = set(r1)
                s2 = set(r2)
                inter = len(set.intersection(s1, s2))
                res.append(inter / float(len(s1)))
            return np.array(res)

        fig, axes = plt.subplots(
            1, 3, sharex=True, sharey=True, figsize=(10, 3))

        nn_range = range(5, 21, 5)

        NN_true = dict()
        NN_pred = dict()
        dis_unk = dict()
        dis_knw = dict()

        nn_matrix = self.neig_sign.get_h5_dataset('V')

        axes[0].set_title('Accuracy Train-Train')
        _, NN_true['trtr'], _ = get_nn_matrix(nn_matrix[tr_nn_idxs])
        dis_knw['trtr'], NN_pred['trtr'], dis_unk['trtr'] = get_nn_matrix(
            preds['tr']['ALL'], unk_data=preds['unk']['ALL'])
        assert(len(NN_true['trtr']) == len(NN_pred['trtr']))
        trtr_jaccs = []
        for top in nn_range:
            jaccs = overlap(NN_pred['trtr'][:, :top], NN_true['trtr'][:, :top])
            trtr_jaccs.append(jaccs)
            sns.distplot(jaccs, ax=axes[0],
                         label='top %s NN' % top, hist=False)
            axes[0].set_xlim(0, 1)
            axes[0].legend()
        metrics['KNNacc_trtr'] = np.mean(trtr_jaccs)
        axes[1].set_title('Accuracy Train-Test')
        trts_jaccs = []
        _, NN_true['trts'], _ = get_nn_matrix(
            np.vstack([nn_matrix[tr_nn_idxs], nn_matrix[ts_nn_idxs]]))
        dis_knw['trts'], NN_pred['trts'], dis_unk['trts'] = get_nn_matrix(np.vstack(
            [preds['tr']['ALL'], preds['ts']['ALL']]), unk_data=preds['unk']['ALL'])
        assert(len(NN_true['trts']) == len(NN_pred['trts']))
        for top in nn_range:
            jaccs = overlap(NN_pred['trts'][:, :top], NN_true['trts'][:, :top])
            trts_jaccs.append(jaccs)
            sns.distplot(jaccs, ax=axes[1],
                         label='top %s NN' % top, hist=False)
            axes[1].set_xlim(0, 1)
            axes[1].legend()
        metrics['KNNacc_trts'] = np.mean(trts_jaccs)
        axes[2].set_title('Accuracy Test-Test')
        tsts_jaccs = []
        _, NN_true['tsts'], _ = get_nn_matrix(nn_matrix[ts_nn_idxs])
        dis_knw['tsts'], NN_pred['tsts'], dis_unk['tsts'] = get_nn_matrix(
            preds['ts']['ALL'], unk_data=preds['unk']['ALL'])
        assert(len(NN_true['tsts']) == len(NN_pred['tsts']))
        for top in nn_range:
            jaccs = overlap(NN_pred['tsts'][:, :top], NN_true['tsts'][:, :top])
            tsts_jaccs.append(jaccs)
            sns.distplot(jaccs, ax=axes[2],
                         label='top %s NN' % top, hist=False)
            axes[2].set_xlim(0, 1)
            axes[2].legend()
        metrics['KNNacc_tsts'] = np.mean(tsts_jaccs)
        plot_file = os.path.join(
            siamese.model_dir, '%s_NN_accuracy.png' % fname)
        plt.savefig(plot_file)
        plt.close()

        del nn_matrix
        del NN_true

        self.__log.info('VALIDATION: Plot KNN %s.' % feat_type)
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for ax_row, k in zip(axes, ['K1', 'K5']):

            ax_row[0].set_title('%s Train-Train' % k)
            if k == 'K1':
                sns.distplot(dis_knw['trtr'][:, 0],
                             label='known', ax=ax_row[0])
                sns.distplot(dis_unk['trtr'][:, 0],
                             label='unknown', ax=ax_row[0])
            else:
                sns.distplot(dis_knw['trtr'][:, 4],
                             label='known', ax=ax_row[0])
                sns.distplot(dis_unk['trtr'][:, 4],
                             label='unknown', ax=ax_row[0])
            ax_row[0].legend()
            ax_row[1].set_title('%s Train-Test' % k)
            if k == 'K1':
                sns.distplot(dis_knw['trts'][:, 0],
                             label='known', ax=ax_row[1])
                sns.distplot(dis_unk['trts'][:, 0],
                             label='unknown', ax=ax_row[1])
            else:
                sns.distplot(dis_knw['trts'][:, 4],
                             label='known', ax=ax_row[1])
                sns.distplot(dis_unk['trts'][:, 4],
                             label='unknown', ax=ax_row[1])
            ax_row[1].legend()
            ax_row[2].set_title('%s Test-Test' % k)
            if k == 'K1':
                sns.distplot(dis_knw['tsts'][:, 0],
                             label='known', ax=ax_row[2])
                sns.distplot(dis_unk['tsts'][:, 0],
                             label='unknown', ax=ax_row[2])
            else:
                sns.distplot(dis_knw['tsts'][:, 4],
                             label='known', ax=ax_row[2])
                sns.distplot(dis_unk['tsts'][:, 4],
                             label='unknown', ax=ax_row[2])
            ax_row[2].legend()
        filename = os.path.join(
            siamese.model_dir, "%s_KNN.png" % fname)
        plt.savefig(filename, dpi=100)
        plt.close()

        del NN_pred
        del dis_unk
        del dis_knw

        self.__log.info(
            'VALIDATION: Plot feature distribution 1 %s.' % feat_type)
        fig = plt.figure(figsize=(20, 6), dpi=100)
        fig, axes = plt.subplots(
            4, 1, sharex=True, sharey=True, figsize=(20, 12))
        order = np.argsort(
            np.mean(np.vstack([preds['tr']['ALL'], preds['ts']['ALL']]), axis=0))[::-1]
        for ax, name in zip(axes.flatten(), mask_fns):
            df = pd.DataFrame(np.vstack([preds['tr'][name], preds['ts'][name]])[
                              :, order]).melt().dropna()
            sns.pointplot(x='variable', y='value', ci='sd',
                          data=df, ax=ax, label=name)
            ax.set_ylabel('known %s' % name)
            ax.axhline(0)
        df = pd.DataFrame(preds['unk']['ALL'][:, order]).melt().dropna()
        ax2 = axes[-1]
        sns.pointplot(x='variable', y='value', ci='sd', data=df, ax=ax2)
        ax2.set_ylabel('unknown ALL')
        ax2.axhline(0)
        filename = os.path.join(
            siamese.model_dir, "%s_features_1.png" % fname)
        plt.savefig(filename, dpi=100)
        plt.close()

        self.__log.info(
            'VALIDATION: Plot feature distribution 2 %s.' % feat_type)
        plt.figure(figsize=(10, 3))
        data = np.array([
            preds['tr']['ALL'].flatten(),
            preds['ts']['ALL'].flatten(),
            np.vstack([preds['tr']['ONLY-SELF'],
                       preds['tr']['ONLY-SELF']]).flatten(),
            preds['unk']['ALL'].flatten(),
            K1_unknown.flatten(),
            K5_unknown.flatten()
        ])
        plt.title('Feature distribution', fontsize=20)
        sns.violinplot(data=data, inner="quartile")
        labels = ['tr', 'te', 'ONLY-SELF', 'unk_naive', 'unk_K1', 'unk_K5']
        plt.xticks(range(len(labels)), labels)
        plot_file = os.path.join(
            siamese.model_dir, '%s_features_2.png' % fname)
        plt.savefig(plot_file)
        plt.close()
        del data

        self.__log.info('VALIDATION: Plot Projections 1 %s.' % feat_type)
        fig, axes = plt.subplots(2, 3, figsize=(10, 7), dpi=200)
        for ax_row, name in zip(axes, ['TSNE', 'PCA']):
            if name == 'TSNE':
                proj_model = MulticoreTSNE(n_components=2)
            else:
                proj_model = PCA(n_components=2)

            tr_idx = np.random.choice(len(preds['tr']['ALL']), min(
                1600, len(preds['tr']['ALL'])), replace=False)
            ts_idx = np.random.choice(len(preds['ts']['ALL']), min(
                400, len(preds['ts']['ALL'])), replace=False)
            uk_idx = np.random.choice(len(K1_unknown), min(
                2000, len(K1_unknown)), replace=False)

            proj_train = np.vstack([
                preds['tr']['ALL'][tr_idx],
                preds['ts']['ALL'][ts_idx],
                K1_unknown[uk_idx]
            ])
            proj = proj_model.fit_transform(proj_train)
            dist_tr = proj[:len(tr_idx)]
            dist_ts = proj[len(tr_idx):len(tr_idx) + len(ts_idx)]
            dist_uk = proj[-len(uk_idx):]
            ax_row[0].set_title('%s K1' % name)
            ax_row[0].scatter(dist_tr[:, 0], dist_tr[:, 1],
                              alpha=.9, label='tr', s=1)
            ax_row[0].scatter(dist_ts[:, 0], dist_ts[:, 1],
                              alpha=.6, label='ts', s=1)
            ax_row[0].scatter(dist_uk[:, 0], dist_uk[:, 1],
                              alpha=.4, label='unk', s=1)
            ax_row[0].legend()

            uk_idx = np.random.choice(len(K5_unknown), min(
                2000, len(K5_unknown)), replace=False)
            proj_train = np.vstack([
                preds['tr']['ALL'][tr_idx],
                preds['ts']['ALL'][ts_idx],
                K5_unknown[uk_idx]
            ])
            proj = proj_model.fit_transform(proj_train)
            dist_tr = proj[:len(tr_idx)]
            dist_ts = proj[len(tr_idx):len(tr_idx) + len(ts_idx)]
            dist_uk = proj[-len(uk_idx):]
            ax_row[1].set_title('%s K5' % name)
            ax_row[1].scatter(dist_tr[:, 0], dist_tr[:, 1],
                              alpha=.9, label='tr', s=1)
            ax_row[1].scatter(dist_ts[:, 0], dist_ts[:, 1],
                              alpha=.6, label='ts', s=1)
            ax_row[1].scatter(dist_uk[:, 0], dist_uk[:, 1],
                              alpha=.4, label='unk', s=1)
            ax_row[1].legend()

            uk_idx = np.random.choice(len(preds['unk']['ALL']), min(
                2000, len(preds['unk']['ALL'])), replace=False)
            proj_train = np.vstack([
                preds['tr']['ALL'][tr_idx],
                preds['ts']['ALL'][ts_idx],
                preds['unk']['ALL'][uk_idx]
            ])
            proj = proj_model.fit_transform(proj_train)
            dist_tr = proj[:len(tr_idx)]
            dist_ts = proj[len(tr_idx):len(tr_idx) + len(ts_idx)]
            dist_uk = proj[-len(uk_idx):]
            ax_row[2].set_title('%s Naive unknown' % name)
            ax_row[2].scatter(dist_tr[:, 0], dist_tr[:, 1],
                              alpha=.9, label='tr', s=1)
            ax_row[2].scatter(dist_ts[:, 0], dist_ts[:, 1],
                              alpha=.6, label='ts', s=1)
            ax_row[2].scatter(dist_uk[:, 0], dist_uk[:, 1],
                              alpha=.4, label='unk', s=1)
            ax_row[2].legend()

            del proj
            del dist_tr
            del dist_ts
            del dist_uk

        plot_file = os.path.join(
            siamese.model_dir, '%s_proj_1.png' % fname)
        plt.savefig(plot_file)
        plt.close()

        self.__log.info('VALIDATION: Plot Projections 2 %s.' % feat_type)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        middle = int(np.ceil(len(tr_idx) / 2))
        tr_all = preds['tr']['ALL'][tr_idx[:middle]]
        tr_not = preds['tr']['NOT-SELF'][tr_idx[middle:]]
        middle = int(np.ceil(len(ts_idx) / 2))
        ts_all = preds['ts']['ALL'][ts_idx[:middle]]
        ts_not = preds['ts']['NOT-SELF'][ts_idx[middle:]]

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.flatten()
        proj_model = MulticoreTSNE(n_components=2)
        proj_train = np.vstack([
            tr_all,
            tr_not,
            ts_all,
            ts_not
        ])
        proj = proj_model.fit_transform(proj_train)

        axes[0].set_title('TSNE tr ALL', fontsize=15)
        p = proj[:len(tr_all)]
        p_esle = proj[len(tr_all):]
        axes[0].scatter(p_esle[:, 0], p_esle[:, 1], color='grey', s=20)
        axes[0].scatter(p[:, 0], p[:, 1], label='tr ALL', s=20)
        axes[0].legend()

        axes[1].set_title('TSNE tr NOT-SELF', fontsize=15)
        p = proj[len(tr_all):len(tr_all) + len(tr_not)]
        p_esle = np.vstack(
            [proj[:len(tr_all)], proj[len(tr_all) + len(tr_not):]])
        axes[1].scatter(p_esle[:, 0], p_esle[:, 1], color='grey', s=20)
        axes[1].scatter(p[:, 0], p[:, 1], label='tr NOT-SELF',
                        s=20, color='orange')
        axes[1].legend()

        axes[2].set_title('TSNE ts ALL', fontsize=15)
        p = proj[len(tr_all) + len(tr_not):len(tr_all) +
                 len(tr_not) + len(ts_all)]
        p_esle = np.vstack(
            [proj[:len(tr_all) + len(tr_not)], proj[-len(ts_not):]])
        axes[2].scatter(p_esle[:, 0], p_esle[:, 1], color='grey', s=20)
        axes[2].scatter(p[:, 0], p[:, 1], label='tr NOT-SELF',
                        s=20, color='green')
        axes[2].legend()

        axes[3].set_title('TSNE ts NOT-SELF', fontsize=15)
        p = proj[-len(ts_not):]
        p_esle = proj[:-len(ts_not):]
        axes[3].scatter(p_esle[:, 0], p_esle[:, 1], color='grey', s=20)
        axes[3].scatter(p[:, 0], p[:, 1], label='tr NOT-SELF',
                        s=20, color='red')
        axes[3].legend()

        plot_file = os.path.join(
            siamese.model_dir, '%s_proj_2.png' % fname)
        plt.savefig(plot_file)
        plt.close()

        metrics_file = os.path.join(siamese.model_dir,
                                    '%s_metrics.pkl' % fname)
        with open(metrics_file, 'wb') as output:
            pickle.dump(metrics, output, pickle.HIGHEST_PROTOCOL)

    def save_sign0_matrix(self, sign0, destination, include_confidence=True,
                          chunk_size=1000):
        """Save matrix of signature 0 and confidence values.

        Args:
            sign0(list): Signature 0 to learn from.
            destination(str): Path where to save the matrix (HDF5 file).
            include_confidence(bool): whether to include confidences.
        """
        self.__log.debug('Saving sign0 traintest to: %s' % destination)
        mask = np.isin(list(self.keys), sign0.keys, assume_unique=True)
        # the following work only if sign0 keys is a subset (or ==) of sign4
        assert(np.all(np.isin(list(sign0.keys), self.keys, assume_unique=True)))
        # shapes?
        common_keys = np.count_nonzero(mask)
        x_shape = (common_keys, sign0.shape[1])
        y_shape = (common_keys, self.shape[1])
        if include_confidence:
            y_shape = (common_keys, self.shape[1] + 5)
        with h5py.File(destination, 'w') as hf_out:
            hf_out.create_dataset('x', x_shape, dtype=np.float32)
            hf_out.create_dataset('y', y_shape, dtype=np.float32)
            with h5py.File(self.data_path, 'r') as hf_in:
                out_start = 0
                for i in tqdm(range(0, self.shape[0], chunk_size)):
                    chunk = slice(i, i + chunk_size)
                    labels = hf_in['V'][chunk][mask[chunk]]
                    if include_confidence:
                        stddev = hf_in['stddev_norm'][chunk][mask[chunk]]
                        stddev = np.expand_dims(stddev, 1)
                        intensity = hf_in['intensity_norm'][chunk][mask[chunk]]
                        intensity = np.expand_dims(intensity, 1)
                        exp_error = hf_in['exp_error_norm'][chunk][mask[chunk]]
                        exp_error = np.expand_dims(exp_error, 1)
                        novelty = hf_in['novelty_norm'][chunk][mask[chunk]]
                        novelty = np.expand_dims(novelty, 1)
                        confidence = hf_in['confidence'][chunk][mask[chunk]]
                        confidence = np.expand_dims(confidence, 1)
                        labels = np.hstack((labels, stddev, intensity,
                                            exp_error, novelty, confidence))
                    out_size = labels.shape[0]
                    out_chunk = slice(out_start, out_start + out_size)
                    hf_out['y'][out_chunk] = labels
                    del labels
                    hf_out['x'][out_chunk] = sign0[out_chunk]
                    out_start += out_size

    def learn_sign0(self, sign0, params, reuse=True, suffix=None,
                    evaluate=True, include_confidence=True):
        """Learn the signature 3 from sign0.

        This method is used twice. First to evaluate the performances of the
        AdaNet model. Second to train the final model on the full set of data.

        Args:
            sign0(list): Signature 0 object to learn from.
            params(dict): Dictionary with algorithm parameters.
            reuse(bool): Whether to reuse intermediate files (e.g. the
                aggregated signature 2 matrix).
            suffix(str): A suffix for the AdaNet model path (e.g.
                'sign4/models/adanet_<suffix>').
            evaluate(bool): Whether we are performing a train-test split and
                evaluating the performances (N.B. this is required for complete
                confidence scores)
            include_confidence(bool): whether to include confidences.
        """
        try:
            from chemicalchecker.tool.adanet import AdaNet
        except ImportError as err:
            raise err
        # adanet parameters
        self.__log.debug('AdaNet fit sign0 %s based on %s', self.dataset,
                         sign0.dataset)
        # get params and set folder
        if suffix:
            adanet_path = os.path.join(self.model_path, 'adanet_%s' % suffix)
        else:
            adanet_path = os.path.join(self.model_path, 'adanet')
        if 'model_dir' in params:
            adanet_path = params.pop('model_dir')
        if not reuse or not os.path.isdir(adanet_path):
            os.makedirs(adanet_path)
        # generate input matrix
        sign0_matrix = os.path.join(self.model_path, 'train_sign0.h5')
        if not reuse or not os.path.isfile(sign0_matrix):
            self.save_sign0_matrix(sign0, sign0_matrix, include_confidence)
        # if evaluating, perform the train-test split
        if evaluate:
            traintest_file = os.path.join(self.model_path,
                                          'traintest_sign0.h5')
            traintest_file = params.pop(
                'traintest_file', traintest_file)
            if not reuse or not os.path.isfile(traintest_file):
                Traintest.split_h5_blocks(sign0_matrix, traintest_file)
        else:
            traintest_file = sign0_matrix
            traintest_file = params.pop(
                'traintest_file', traintest_file)
        # initialize adanet and start learning
        ada = AdaNet(model_dir=adanet_path,
                     traintest_file=traintest_file,
                     **params)
        self.__log.debug('AdaNet training on %s' % traintest_file)
        ada.train_and_evaluate(evaluate=evaluate)
        self.__log.debug('model saved to %s' % adanet_path)
        # when evaluating also save the performances
        if evaluate:
            # save AdaNet performances and plots
            sign2_plot = Plot(self.dataset, adanet_path)
            ada.save_performances(adanet_path, sign2_plot, suffix)

    def save_sign0_conf_matrix(self, sign0, destination, chunk_size=1000):
        """Save matrix of signature 0 confidence values.

        Args:
            sign0(list): Signature 0 to learn from.
            destination(str): Path where to save the matrix (HDF5 file).
            include_confidence(bool): whether to include confidences.
        """
        self.__log.debug('Saving confidence traintest to: %s' % destination)
        mask = np.isin(list(self.keys), list(sign0.keys), assume_unique=True)
        # the following work only if sign0 keys is a subset (or ==) of sign4
        assert(np.all(np.isin(list(sign0.keys), list(self.keys), assume_unique=True)))
        # shapes?
        common_keys = np.count_nonzero(mask)
        x_shape = (common_keys, sign0.shape[1])
        y_shape = (common_keys, 5)
        self.__log.debug('Shapes X: %s Y: %s' % (str(x_shape), str(y_shape)))
        with h5py.File(destination, 'w') as hf_out:
            hf_out.create_dataset('x', x_shape, dtype=np.float32)
            hf_out.create_dataset('y', y_shape, dtype=np.float32)
            with h5py.File(self.data_path, 'r') as hf_in:
                out_start = 0
                for i in tqdm(range(0, self.shape[0], chunk_size)):
                    chunk = slice(i, i + chunk_size)
                    stddev = hf_in['stddev_norm'][chunk][mask[chunk]]
                    stddev = np.expand_dims(stddev, 1)
                    intensity = hf_in['intensity_norm'][chunk][mask[chunk]]
                    intensity = np.expand_dims(intensity, 1)
                    exp_error = hf_in['exp_error_norm'][chunk][mask[chunk]]
                    exp_error = np.expand_dims(exp_error, 1)
                    novelty = hf_in['novelty_norm'][chunk][mask[chunk]]
                    novelty = np.expand_dims(novelty, 1)
                    confidence = hf_in['confidence'][chunk][mask[chunk]]
                    confidence = np.expand_dims(confidence, 1)
                    conf_scores = np.hstack((stddev, intensity,
                                             exp_error, novelty, confidence))
                    out_size = conf_scores.shape[0]
                    out_chunk = slice(out_start, out_start + out_size)
                    hf_out['y'][out_chunk] = conf_scores
                    hf_out['x'][out_chunk] = sign0[out_chunk]
                    out_start += out_size

    def learn_sign0_conf(self, sign0, params, reuse=True, suffix=None,
                         evaluate=True):
        """Learn the signature 3 confidence from sign0.

        This method is used twice. First to evaluate the performances of the
        AdaNet model. Second to train the final model on the full set of data.

        Args:
            sign0(list): Signature 0 object to learn from.
            params(dict): Dictionary with algorithm parameters.
            reuse(bool): Whether to reuse intermediate files (e.g. the
                aggregated signature 2 matrix).
            suffix(str): A suffix for the AdaNet model path (e.g.
                'sign4/models/adanet_<suffix>').
            evaluate(bool): Whether we are performing a train-test split and
                evaluating the performances (N.B. this is required for complete
                confidence scores)
            include_confidence(bool): whether to include confidences.
        """
        try:
            from chemicalchecker.tool.adanet import AdaNet
        except ImportError as err:
            raise err
        # adanet parameters
        self.__log.debug('AdaNet fit confidence %s based on %s', self.dataset,
                         sign0.dataset)
        # get params and set folder
        if suffix:
            adanet_path = os.path.join(self.model_path, 'adanet_%s' % suffix)
        else:
            adanet_path = os.path.join(self.model_path, 'adanet')
        if 'model_dir' in params:
            adanet_path = params.pop('model_dir')
        if not reuse or not os.path.isdir(adanet_path):
            os.makedirs(adanet_path)
        # generate input matrix
        sign0_matrix = os.path.join(self.model_path, 'train_sign0_conf.h5')
        if not reuse or not os.path.isfile(sign0_matrix):
            self.save_sign0_conf_matrix(sign0, sign0_matrix)
        # if evaluating, perform the train-test split
        if evaluate:
            traintest_file = os.path.join(self.model_path,
                                          'traintest_sign0_conf.h5')
            traintest_file = params.pop(
                'traintest_file', traintest_file)
            if not reuse or not os.path.isfile(traintest_file):
                Traintest.split_h5_blocks(sign0_matrix, traintest_file)
        else:
            traintest_file = sign0_matrix
            traintest_file = params.pop(
                'traintest_file', traintest_file)
        # initialize adanet and start learning
        ada = AdaNet(model_dir=adanet_path,
                     traintest_file=traintest_file,
                     **params)
        self.__log.debug('AdaNet training on %s' % traintest_file)
        ada.train_and_evaluate(evaluate=evaluate)
        self.__log.debug('model saved to %s' % adanet_path)

        if evaluate:
            # save AdaNet performances and plots
            sign2_plot = Plot(self.dataset, adanet_path)
            ada.save_performances(adanet_path, sign2_plot, suffix)

    def learn_error(self, predict_fn, params, reuse=True, suffix=None,
                    evaluate=True):
        """Learn the signature 3 prediction error.

        This method is used twice. First to evaluate the performances of the
        model. Second to train the final model on the full set of data.

        Args:
            sign2_list(list): List of signature 2 objects to learn from.
            reuse(bool): Whether to reuse intermediate files (e.g. the
                aggregated signature 2 matrix).
            suffix(str): A suffix for the AdaNet model path (e.g.
                'sign4/models/adanet_<suffix>').
            evaluate(bool): Whether we are performing a train-test split and
                evaluating the performances (N.B. this is required for complete
                confidence scores)
        """
        # get params and set folder
        if suffix:
            model_dir = os.path.join(self.model_path, 'error_%s' % suffix)
            traintest_file = os.path.join(self.model_path,
                                          'traintest_error_%s.h5' % suffix)
        else:
            model_dir = os.path.join(self.model_path, 'adanet')
            traintest_file = os.path.join(self.model_path,
                                          'traintest_error.h5')
        if params:
            if 'model_dir' in params:
                model_dir = params.pop('model_dir')
        if not reuse or not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        # generate input matrix
        sign2_matrix = os.path.join(self.model_path, 'train.h5')
        if not reuse or not os.path.isfile(sign2_matrix):
            self.save_sign2_matrix(self.sign2_list, sign2_matrix)

        # update the subsampling function
        dataset_idx = np.argwhere(np.isin(self.src_datasets, ds)).flatten()
        p_nr, p_keep = subsampling_probs(self.sign2_coverage, dataset_idx)
        subsample_fn = partial(subsample, dataset_idx=dataset_idx,
                               p_nr=p_nr, p_keep=p_keep)

        # if evaluating, perform the train-test split
        traintest_file = params.pop('traintest_file', traintest_file)
        if evaluate:
            if not reuse or not os.path.isfile(traintest_file):
                NeighborErrorTraintest.create(
                    sign2_matrix, traintest_file, predict_fn, subsample_fn,
                    split_names=['train', 'test'],
                    split_fractions=[.8, .2],
                    neigbors_matrix=self.neig_sign[:])
        else:
            if not reuse or not os.path.isfile(traintest_file):
                NeighborErrorTraintest.create(
                    sign2_matrix, traintest_file, predict_fn, subsample_fn,
                    split_names=['train'],
                    split_fractions=[.1],
                    neigbors_matrix=self.neig_sign[:])

        if evaluate:
            self.train_predictors(params, model_dir, traintest_file)
            #self.save_performances(model_dir, sign2_plot, suffix, others)
        else:
            self.train_predictors(
                params, model_dir, traintest_file, train_only=True)

    def train_predictors(self, predictors, save_path, traintest_file,
                         train_only=False):
        """Train predictors for comparison."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import pearsonr

        def train_predict_save(name, model, x_true, y_true, split, save_path):
            result = dict()
            result['time'] = 0.
            # train and save
            if split is None or split == 'train':
                self.__log.info('Training model: %s' % name)
                t0 = time()
                model.fit(x_true, y_true)
                model_path = os.path.join(save_path, '%s.pkl' % name)
                pickle.dump(model, open(model_path, 'wb'))
                result['time'] = time() - t0
                self.__log.info('Training took: %.1f sec' % result['time'])
            # call predict
            self.__log.info("Predicting for: %s", name)
            y_pred = model.predict(x_true)
            y_pred = np.expand_dims(y_pred, 1)
            y_true = np.expand_dims(y_true, 1)
            self.__log.info("%s Y: %s", name, y_pred.shape)
            pr, pval = pearsonr(y_pred.ravel(), y_true.ravel())
            self.__log.info("%s pearson r: %.2f pval: %.E" % (name, pr, pval))
            if y_pred.shape[0] < 4:
                return
            file_true = os.path.join(
                save_path, "_".join([name] + [str(split), 'true']))
            np.save(file_true, y_true)
            file_pred = os.path.join(
                save_path, "_".join([name] + [str(split), 'pred']))
            np.save(file_pred, y_pred)
            result['true'] = file_true
            result['pred'] = file_pred
            result['coverage'] = 1.0
            return result

        # get results for each split or none if only training
        results = dict()
        if train_only:
            splits = [None]
        else:
            splits = ['train', 'test']
        for split in splits:
            traintest = Traintest(traintest_file, split)
            x_shape, y_shape = traintest.get_xy_shapes()
            self.__log.info("%s X: %s Y: %s", split, x_shape, y_shape)
            self.__log.info("%s Y: %s", split, y_shape)
            traintest.open()
            x_data = traintest.get_all_x()
            y_data = traintest.get_all_y().flatten()
            traintest.close()
            for name, model in predictors.items():
                if name not in results:
                    results[name] = dict()
                # predict and save
                results[name][str(split)] = train_predict_save(
                    name, model, x_data, y_data, split, save_path)
        # save prerformances
        if not train_only:
            for name in sorted(results):

                fig, axes = plt.subplots(
                    2, len(splits), sharex=True, sharey=False, figsize=(7, 7))
                for ax, split in zip(axes[0], splits):
                    preds = results[name]
                    if split not in preds:
                        self.__log.info("Skipping %s on %s", name, split)
                        continue
                    if preds[split] is None:
                        self.__log.info("Skipping %s on %s", name, split)
                        continue
                    y_pred = np.load(preds[split]['pred'] + ".npy")
                    y_true = np.load(preds[split]['true'] + ".npy")
                    sns.regplot(y_true.flatten(), y_pred.flatten(),
                                # x_estimator=np.mean,
                                marker='.',
                                truncate=False,
                                scatter_kws={'alpha': .5}, ax=ax)
                    ax.set_xlabel('True')
                    ax.set_ylabel('Predicted')
                    ax.set_xlim(-1, 1)
                    ax.set_ylim(-1, 1)
                    ax.set_title(split)
                for ax, split in zip(axes[1], splits):
                    preds = results[name]
                    if split not in preds:
                        self.__log.info("Skipping %s on %s", name, split)
                        continue
                    if preds[split] is None:
                        self.__log.info("Skipping %s on %s", name, split)
                        continue
                    y_pred = np.load(preds[split]['pred'] + ".npy")
                    y_true = np.load(preds[split]['true'] + ".npy")
                    sns.distplot(y_pred, label='Pred', ax=ax)
                    sns.distplot(y_true, label='True', ax=ax)
                    ax.set_xlabel('Pearsons rho')
                    ax.set_xlim(-1, 1)
                    ax.set_title(split)
                plt.tight_layout()
                plot_file = os.path.join(save_path,
                                         '%s_correlations.png' % name)
                plt.savefig(plot_file)
                plt.close()

        return results

    def fit_sign0(self, sign0, include_confidence=True, extra_confidence=False):
        """Train an AdaNet model to predict sign4 from sign0.

        This method is fitting a model that uses Morgan fingerprint as features
        to predict signature 3. In future other featurization approaches can be
        tested.

        Args:
            chemchecker(ChemicalChecker): The CC object used to fetch input
                signature 0.
            sign0_traintest(str): Path to the train file.
            include_confidence(bool): Whether to include confidence score in
                regression problem.
            extra_confidence(bool): Whether to train an additional regressor
                exclusively devoted to confidence.
        """

        # check if performance evaluations need to be done
        s0_code = sign0.dataset
        eval_adanet_path = os.path.join(self.model_path,
                                        'adanet_sign0_%s_eval' % s0_code)
        eval_stats = os.path.join(eval_adanet_path,
                                  'stats_sign0_%s_eval.pkl' % s0_code)
        if not os.path.isfile(eval_stats):
            self.learn_sign0(sign0, self.params['sign0'],
                             suffix='sign0_%s_eval' % s0_code,
                             evaluate=True,
                             include_confidence=include_confidence)
        # learn confidence predictor
        if extra_confidence:
            conf_eval_adanet_path = os.path.join(
                self.model_path, 'adanet_sign0_%s_conf_eval' % s0_code)
            conf_eval_stats = os.path.join(
                conf_eval_adanet_path, 'stats_conf_eval.pkl')
            if not os.path.isfile(conf_eval_stats):
                self.learn_sign0_conf(sign0, self.params['sign0_conf'],
                                      suffix='sign0_%s_conf_eval' % s0_code,
                                      evaluate=True)

        # check if we have the final trained model
        final_adanet_path = os.path.join(self.model_path,
                                         'adanet_sign0_%s_final' % s0_code,
                                         'savedmodel')
        if not os.path.isdir(final_adanet_path):
            self.learn_sign0(sign0, self.params['sign0'],
                             suffix='sign0_%s_final' % s0_code,
                             evaluate=False,
                             include_confidence=include_confidence)
        # learn the final confidence predictor
        if extra_confidence:
            conf_final_adanet_path = os.path.join(
                self.model_path, 'adanet_sign0_%s_conf_final' % s0_code)
            conf_final_stats = os.path.join(
                conf_final_adanet_path, 'stats_conf_final.pkl')
            if not os.path.isfile(conf_final_stats):
                self.learn_sign0_conf(sign0, self.params['sign0_conf'],
                                      suffix='sign0_%s_conf_final' % s0_code,
                                      evaluate=False)

    def get_predict_fn(self, model='adanet_sign0_A1.001_final'):
        try:
            from chemicalchecker.tool.adanet import AdaNet
        except ImportError as err:
            raise err
        modelpath = os.path.join(self.model_path, model, 'savedmodel')
        return AdaNet.predict_fn(modelpath)

    def predict_from_smiles(self, smiles, dest_file, chunk_size=1000,
                            predict_fn=None, accurate_novelty=False,
                            keys=None, components=128, include_confidence=True):
        """Given SMILES generate sign0 and predict sign4.

        Args:
            smiles(list): A list of SMILES strings. We assume the user already
                standardized the SMILES string.
            dest_file(str): File where to save the predictions.
        Returns:
            pred_s3(DataSignature): The predicted signatures as DataSignature
                object.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError as err:
            raise err
        # load NN
        if predict_fn is None:
            predict_fn = self.get_predict_fn()
        # we return a simple DataSignature object (basic HDF5 access)
        pred_s3 = DataSignature(dest_file)
        # load novelty model for more accurate novelty scores (slower)
        if accurate_novelty:
            novelty_path = os.path.join(self.model_path, 'novelty', 'lof.pkl')
            try:
                novelty_model = pickle.load(open(novelty_path, 'rb'))
            except Exception:
                import dill
                dill._dill._reverse_typemap["ObjectType"] = object
                novelty_model = pickle.load(open(novelty_path, 'rb'),
                                            encoding="bytes")

            nov_qtr_path = os.path.join(self.model_path, 'novelty', 'qtr.pkl')
            try:
                nov_qtr = pickle.load(open(nov_qtr_path, 'rb'))
            except Exception:
                import dill
                dill._dill._reverse_typemap["ObjectType"] = object
                nov_qtr = pickle.load(open(nov_qtr_path, 'rb'),
                                      encoding="bytes")
        with h5py.File(dest_file, "w") as results:
            # initialize V (with NaN in case of failing rdkit) and smiles keys
            results.create_dataset('smiles', data=np.array(
                smiles, DataSignature.string_dtype()))
            if keys is not None:
                results.create_dataset('keys', data=np.array(
                    keys, DataSignature.string_dtype()))
            else:
                results.create_dataset('keys', data=np.array(
                    smiles, DataSignature.string_dtype()))
            results.create_dataset(
                'V', (len(smiles), components), dtype=np.float32)
            if include_confidence:
                results.create_dataset(
                    'stddev_norm', (len(smiles), ), dtype=np.float32)
                results.create_dataset(
                    'intensity_norm', (len(smiles), ), dtype=np.float32)
                results.create_dataset(
                    'exp_error_norm', (len(smiles), ), dtype=np.float32)
                results.create_dataset(
                    'novelty_norm', (len(smiles), ), dtype=np.float32)
                results.create_dataset(
                    'confidence', (len(smiles), ), dtype=np.float32)
            results.create_dataset("shape", data=(len(smiles), components))
            # compute sign0 (i.e. Morgan fingerprint)
            nBits = 2048
            radius = 2
            # predict by chunk
            for i in tqdm(range(0, len(smiles), chunk_size)):
                chunk = slice(i, i + chunk_size)
                sign0s = list()
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
                        calc_s0 = np.full((nBits, ),  np.nan)
                    finally:
                        sign0s.append(calc_s0)
                # stack input signatures and generate predictions
                sign0s = np.vstack(sign0s)
                preds = predict_fn({'x': sign0s})['predictions']
                # add NaN when SMILES conversion failed
                if failed:
                    if include_confidence:
                        preds[np.array(failed)] = np.full(
                            (components + 5, ),  np.nan)
                    else:
                        preds[np.array(failed)] = np.full(
                            (components, ),  np.nan)
                # save chunk to H5
                results['V'][chunk] = preds[:, :components]
                if include_confidence:
                    results['stddev_norm'][chunk] = preds[:, components]
                    results['intensity_norm'][chunk] = preds[:, components + 1]
                    results['exp_error_norm'][chunk] = preds[:, components + 2]
                    results['novelty_norm'][chunk] = preds[:, components + 3]
                    results['confidence'][chunk] = preds[:, components + 4]
                    if accurate_novelty:
                        novelty = novelty_model.score_samples(
                            preds[:, :components])
                        abs_novelty = np.abs(np.expand_dims(novelty, 1))
                        results['novelty_norm'][chunk] = nov_qtr.transform(
                            abs_novelty).flatten()
        return pred_s3

    def get_universe_inchikeys(self):
        # get sorted universe inchikeys
        if hasattr(self, 'universe_inchikeys'):
            return self.universe_inchikeys
        inchikeys = set()
        for ds, sign in zip(self.src_datasets, self.sign2_list):
            inchikeys.update(sign.unique_keys)
        inchikeys = sorted(list(inchikeys))
        return inchikeys

    def fit(self, sign2_list, sign2_self, sign1_self, sign2_universe=None,
            partial_universe=None,
            sign2_coverage=None, sign0=None,
            model_confidence=False, save_correlations=False,
            predict_novelty=False, update_preds=True, normalize_scores=False,
            mask_features=False,
            validations=True, chunk_size=1000):
        """Fit the model to predict the signature 3.

        Args:
            sign2_list(list): List of signature 2 objects to learn from.
            sign2_self(sign2): Signature 2 of the current space.
            sign2_universe(str): Path to the union of all signatures 2 for all
                molecules in the CC universe. (800k x 3200)
            sign2_coverage(str): Path to the coverage of all signatures 2 for
                all molecules in the CC universe. (800k x 25)
            sign0(sign): The signature 0 or any direct vector representation
                of the molecule used to train a sign2 independendent sign4
                predictor.
            model_confidence(bool): Whether to model confidence. That is based
                on standard deviation of prediction with dropout.
            save_correlations(bool) Whether to save the correlation (average,
                tertile, max) for the given input dataset (result of the
                evaluation).
            predict_novelty(bool) Whether to predict molecule novelty score.
            update_preds(bool): Whether to write or update the sign4.h5
            normalize_scores(bool): Whether to normalize confidence scores.
            validations(bool): Whether to perform validation.
            chunk_size(int): Chunk size when writing to sign4.h5
        """
        try:
            import faiss
        except ImportError as err:
            raise err

        # define datasets that will be used
        self.src_datasets = [sign.dataset for sign in sign2_list]
        self.neig_sign = sign1_self
        self.sign2_self = sign2_self
        self.sign2_list = sign2_list
        self.sign2_coverage = sign2_coverage
        self.dataset_idx = np.argwhere(
            np.isin(self.src_datasets, self.dataset)).flatten()
        if self.sign2_coverage is None:
            self.sign2_coverage = os.path.join(self.model_path,
                                               'all_sign2_coverage.h5')
        if not os.path.isfile(self.sign2_coverage):
            sign4.save_sign2_coverage(sign2_list, self.sign2_coverage)

        self.__log.debug('Siamese fit %s based on %s', self.dataset,
                         str(self.src_datasets))

        # build input matrix if not provided (should be since is shared)
        self.sign2_universe = sign2_universe
        if self.sign2_universe is None:
            self.sign2_universe = os.path.join(self.model_path, 'all_sign2.h5')
        if partial_universe is not None:
            sign4.complete_sign2_universe(
                sign2_list, sign2_self,
                partial_universe, self.sign2_universe)
        if not os.path.isfile(self.sign2_universe):
            sign4.save_sign2_universe(sign2_list, self.sign2_universe)

        # check if performance evaluations need to be done
        eval_model_path = os.path.join(self.model_path, 'siamese_eval')
        eval_file = os.path.join(eval_model_path, 'siamese.h5')
        if not os.path.isfile(eval_file):
            self.learn_sign2(self.params['sign2'],
                             suffix='eval', evaluate=True)

        # check if we have the final trained model
        final_model_path = os.path.join(self.model_path, 'siamese_final')
        final_file = os.path.join(final_model_path, 'siamese.h5')
        if not os.path.isfile(final_file):
            self.learn_sign2(self.params['sign2'],
                             suffix='final', evaluate=False)
        siamese = SiameseTriplets(final_model_path)

        if model_confidence:
            # part of confidence is the expected error/accuracy
            # generate prediction, measure error, fit regressor
            predict_fn = siamese.predict
            eval_err_path = os.path.join(self.model_path, 'error_eval')
            if not os.path.isdir(eval_err_path):
                # step1 learn dataset availability to error predictor
                self.learn_error(predict_fn, self.params['error'],
                                 suffix='eval', evaluate=True)
            # final error predictor
            final_err_path = os.path.join(self.model_path, 'error_final')
            if not os.path.isdir(final_err_path):
                self.learn_error(predict_fn, self.params['error'],
                                 suffix='final', evaluate=False)
            self.__log.debug('Loading model for error prediction')
            rf = pickle.load(
                open(os.path.join(final_err_path, 'RandomForest.pkl')), 'rb')
            # another part of confidence is the intensity
            # that is proportional to the distance to prediction of knowns
            NN_file = os.path.join(self.model_path, 'known_nn.h5')
            NN = faiss.read_index(NN_file)

        # get sorted universe inchikeys
        self.universe_inchikeys = self.get_universe_inchikeys()
        tot_inks = len(self.universe_inchikeys)
        tot_ds = len(self.src_datasets)

        # save universe sign4
        if update_preds:
            with h5py.File(self.data_path, "a") as results:
                # initialize V and keys datasets
                safe_create(results, 'V', (tot_inks, 128), dtype=np.float32)
                safe_create(results, 'keys',
                            data=np.array(self.universe_inchikeys,
                                          DataSignature.string_dtype()))
                safe_create(results, 'datasets',
                            data=np.array(self.src_datasets,
                                          DataSignature.string_dtype()))
                safe_create(results, 'shape', data=(tot_inks, 128))
                if model_confidence:
                    # the actual confidence value will be stored here
                    safe_create(results, 'confidence',
                                (tot_inks,), dtype=np.float32)
                    safe_create(results, 'confidence_raw',
                                (tot_inks,), dtype=np.float32)
                    # this is to store standard deviation
                    safe_create(results, 'stddev',
                                (tot_inks,), dtype=np.float32)
                    safe_create(results, 'stddev_norm',
                                (tot_inks,), dtype=np.float32)
                    # this is to store intensity
                    safe_create(results, 'intensity',
                                (tot_inks,), dtype=np.float32)
                    safe_create(results, 'intensity_norm',
                                (tot_inks,), dtype=np.float32)
                    # this is to store distance
                    safe_create(results, 'distance',
                                (tot_inks,), dtype=np.float32)
                    safe_create(results, 'distance_norm',
                                (tot_inks,), dtype=np.float32)
                    # this is to store error prediction
                    safe_create(results, 'exp_error',
                                (tot_inks,), dtype=np.float32)
                    safe_create(results, 'exp_error_norm',
                                (tot_inks,), dtype=np.float32)
                    # this is to store the consensus prediction
                    safe_create(results, 'consensus',
                                (tot_inks, 128), dtype=np.float32)
                if predict_novelty:
                    safe_create(results, 'novelty',
                                (tot_inks, ), dtype=np.float32)
                    safe_create(results, 'novelty_norm',
                                (tot_inks, ), dtype=np.float32)
                    safe_create(results, 'outlier',
                                (tot_inks, ), dtype=np.float32)
                if save_correlations:
                    # read the correlations obtained evaluating on single
                    # spaces
                    df = pd.read_pickle(eval_stats)
                    test_eval = df[(df.split != 'train') & (
                        df.algo == 'AdaNet_eval')]
                    train_eval = df[(df.split == 'train') & (
                        df.algo == 'AdaNet_eval')]
                    avg_pearsons = np.zeros(tot_ds, dtype=np.float32)
                    avg_pearsons_test = np.zeros(tot_ds, dtype=np.float32)
                    avg_pearsons_train = np.zeros(tot_ds, dtype=np.float32)
                    for idx, ds in enumerate(self.src_datasets):
                        ds_df = test_eval[test_eval['from'] == ds]
                        ds_pearson = np.mean(ds_df['pearson'])
                        if np.isnan(ds_pearson):
                            ds_pearson = 0.0
                        avg_pearsons_test[idx] = ds_pearson
                        ds_df = train_eval[train_eval['from'] == ds]
                        ds_pearson = np.mean(ds_df['pearson'])
                        if np.isnan(ds_pearson):
                            ds_pearson = 0.0
                        avg_pearsons_train[idx] = ds_pearson
                    avg_pearsons = avg_pearsons_test
                    # this is to lookup correlations
                    safe_create(results, 'datasets_correlation',
                                data=avg_pearsons)

                # predict signature 3 for universe molecules
                with h5py.File(sign2_universe, "r") as features:
                    # reference prediction (based on no information)
                    nan_feat = np.full((1, features['x_test'].shape[1]),
                                       np.nan, dtype=np.float32)
                    nan_pred = siamese.predict(nan_feat)
                    # read input in chunks
                    for idx in tqdm(range(0, tot_inks, chunk_size),
                                    desc='Predicting'):
                        chunk = slice(idx, idx + chunk_size)
                        feat = features['x_test'][chunk]
                        # predict with final model
                        preds = siamese.predict(feat)
                        results['V'][chunk] = preds
                        # skip modelling confidence if not required
                        if not model_confidence:
                            continue
                        # save confidence natural scores
                        # compute estimated error from coverage
                        coverage = ~np.isnan(feat[:, 0::128])
                        results['exp_error'][chunk] = rf.predict(coverage)
                        # conformal prediction
                        ints, stds, cons = self.conformal_prediction(
                            siamese, feat, nan_pred=nan_pred)
                        results['intensity'][chunk] = ints
                        results['stddev'][chunk] = stds
                        results['consensus'][chunk] = cons
                        # distance from known predictions
                        distances = NN.search(preds, 50)
                        results['distance'][chunk] = np.mean(
                            distances, axis=1).flatten()

        # normalize consensus scores sampling distribution of known signatures
        if normalize_scores:
            self.normalize_scores()

        # use semi-supervised anomaly detection algorithm to predict novelty
        if predict_novelty:
            self.predict_novelty()

        self.background_distances("cosine")
        if validations:
            self.validate()
        # at the very end we learn how to get from A1 sign0 to sign4 directly
        # in order to enable SMILES to sign4 predictions
        if sign0 is not None:
            self.fit_sign0(sign0)
        self.mark_ready()

    def normalize_scores(self, chunk_size=10000):
        """Normalize confidence scores."""

        # get distribution for normalization
        distributions = self.confidence_distributions(self.sign2_self)
        std_dist, int_dist, err_dist, correlation = distributions
        weights = [abs(correlation['intensity']),
                   abs(correlation['stddev']),
                   abs(correlation['exp_error'])]
        # fit quantile transformers for normalization
        self.__log.info('Quatile transformation...')
        std_qtr = QuantileTransformer(n_quantiles=len(std_dist)).fit(std_dist)
        int_qtr = QuantileTransformer(n_quantiles=len(int_dist)).fit(int_dist)
        err_qtr = QuantileTransformer(n_quantiles=len(err_dist)).fit(err_dist)
        # add confidence scores and the normalized values
        with h5py.File(self.data_path, "r+") as results:
            safe_create(results, 'confidence_raw',
                        (self.tot_inks,), dtype=np.float32)
            for idx in tqdm(range(0, self.tot_inks, chunk_size)):
                chunk = slice(idx, idx + chunk_size)
                # get intensity and normalize wrt train distribution
                intensity = np.expand_dims(results['intensity'][chunk], axis=1)
                inten_norm = int_qtr.transform(intensity)
                results['intensity_norm'][chunk] = inten_norm.flatten()
                # get stddev and normalize wrt train distribution
                stddev = np.expand_dims(results['stddev'][chunk], axis=1)
                stddev_norm = std_qtr.transform(stddev)
                results['stddev_norm'][chunk] = stddev_norm.flatten()
                # get predicted error and normalize wrt train distribution
                error = np.expand_dims(results['exp_error'][chunk], axis=1)
                error_norm = err_qtr.transform(error)
                results['exp_error_norm'][chunk] = error_norm.flatten()
                # stddev, intensity and estimated error contribute
                # to the raw confidence score based on error correlation
                confidence_raw = np.average(
                    [inten_norm, (1 - stddev_norm), (1 - error_norm)],
                    weights=weights, axis=0)
                # range [0,1] the higher the confidence the lower the error
                results['confidence_raw'][chunk] = np.clip(
                    confidence_raw, 0.0, 1.0).flatten()
        self.__log.info('Confidence performance scaling...')
        # final scaling according to performance on unseen data
        ds_corr = self.get_h5_dataset('datasets_correlation')
        # load adanet_eval and its test data
        traintest_file = os.path.join(self.model_path, 'traintest.h5')
        predict_fn = self.get_predict_fn('adanet_eval')
        # get various 'difficulty' level correlations
        dist_pars = list()
        for corr_thr in [.7, .9, 1.0]:
            # mask highly correlated spaces
            corr_mask = ds_corr > corr_thr
            corr_spaces = np.array(self.src_datasets)[corr_mask].tolist()
            self.__log.debug('Masking %s' % str(corr_spaces))
            corr_idxs = np.argwhere(corr_mask).flatten().tolist()
            mask_fn = partial(mask_exclude, corr_idxs)
            # predict and compute component correlations (using unseen data)
            y_pred_test, y_true_test = AdaNet.predict_online(
                traintest_file, 'test', predict_fn, mask_fn)
            y_pred_val, y_true_val = AdaNet.predict_online(
                traintest_file, 'validation', predict_fn, mask_fn)
            y_pred = np.vstack([y_pred_test, y_pred_val])
            y_true = np.vstack([y_true_test, y_true_val])
            correlations = col_wise_correlation(y_pred, y_true)
            dist_pars.append(
                (corr_mask, np.mean(correlations), np.std(correlations)))
            self.__log.debug('Distribution N(%.2f,%.2f)' % (dist_pars[-1][1:]))
        # get mol indexes where to apply the different transformations
        full_coverage = DataSignature(self.sign2_coverage).get_h5_dataset(
            'x_test').astype(bool)
        confidence_raw = self.get_h5_dataset('confidence_raw')
        done = np.full(full_coverage.shape[0], False)
        confidence = list()
        for mask, mu, std in dist_pars:
            # get molecules without any of the spaces in the mask
            molmask = np.argwhere(
                ~np.any(full_coverage & mask, axis=1)).flatten()
            # exclude aready done molecules
            molmask = molmask[~done[molmask]]
            self.__log.debug('Scaling %s mols. to N(%.2f,%.2f)' %
                             (molmask.shape[0], mu, std))
            # get confidences
            dist = confidence_raw[molmask]
            # gaussianize the confidences
            scaled = norm.ppf((rankdata(dist) - 0.5) / len(dist), mu, std)
            # append to list for later sorting
            confidence.extend(zip(molmask, scaled))
            # these molecules are done, do include them in following
            done[molmask] = True
        confidence = np.array(sorted(confidence))
        with h5py.File(self.data_path, "r+") as results:
            if 'confidence' in results:
                del results['confidence']
            results['confidence'] = np.clip(
                confidence[:, 1], 0.0, 1.0).flatten()

    def confidence_distributions(self, sign2_self, max_sample=100000):
        """Get distributions for confidence scores normalization.

        Args:
            sign2_self(sign2): The signature that we are training for.
            max_sample(int): The maximum number of molecule to sample.
        Returns:
            stddev(np.array): Distribution of standard deviations.
            intensity(np.array): Distribution of intensity.
            exp_error(np.array): Distribution of predicted errors.
            correlation(dict): Error correlation coefficients.
        """
        self.__log.debug('Generating confidence scores distributions')
        # get current space inchikeys (limit to 20^4)
        dataset_inks = sign2_self.keys
        if len(dataset_inks) > max_sample:
            dataset_inks = np.random.choice(dataset_inks, max_sample,
                                            replace=False)
        # get the features to train the estimator on
        _, stddev = self.get_vectors(dataset_inks, dataset_name='stddev')
        _, intensity = self.get_vectors(dataset_inks, dataset_name='intensity')
        _, exp_error = self.get_vectors(
            dataset_inks, dataset_name='exp_error')
        # also get the predicted and actual sign2
        _, consensus = self.get_vectors(dataset_inks, dataset_name='consensus')
        _, predicted = self.get_vectors(dataset_inks)
        _, actual = sign2_self.get_vectors(dataset_inks)
        # calculate the error (what we want to predict)
        log_mse = np.log10(np.mean(((actual - predicted)**2), axis=1))
        log_mse_consensus = np.log10(
            np.average(((actual - consensus)**2), axis=1))
        # calculate correlations for scaling
        correlation = dict()
        corr_stddev_cons = pearsonr(stddev.flatten(), log_mse_consensus)[0]
        self.__log.debug('stddev score correlation to consensus error:' +
                         ' %.2f' % corr_stddev_cons)
        corr_stddev = pearsonr(stddev.flatten(), log_mse)[0]
        self.__log.debug('stddev score correlation to error:' +
                         ' %.2f' % corr_stddev)
        correlation['stddev'] = corr_stddev_cons
        corr_intensity_cons = pearsonr(
            intensity.flatten(), log_mse_consensus)[0]
        self.__log.debug('intensity score correlation to consensus error:' +
                         ' %.2f' % corr_intensity_cons)
        corr_intensity = pearsonr(intensity.flatten(), log_mse)[0]
        self.__log.debug('intensity score correlation to error:' +
                         ' %.2f' % corr_intensity)
        correlation['intensity'] = corr_intensity_cons
        corr_exp_error_cons = pearsonr(
            exp_error.flatten(), log_mse_consensus)[0]
        self.__log.debug('exp_error score correlation to consensus error:' +
                         ' %.2f' % corr_exp_error_cons)
        corr_exp_error = pearsonr(exp_error.flatten(), log_mse)[0]
        self.__log.debug('exp_error score correlation to error:' +
                         ' %.2f' % corr_exp_error)
        correlation['exp_error'] = corr_exp_error
        # save data in the confidence model
        error_file = os.path.join(self.model_path, 'error.h5')
        with h5py.File(error_file, "w") as hf:
            hf.create_dataset('keys',
                              data=np.array(dataset_inks,
                                            DataSignature.string_dtype()))
            hf.create_dataset('stddev', data=stddev, dtype=np.float32)
            hf.create_dataset('intensity', data=intensity, dtype=np.float32)
            hf.create_dataset('exp_error', data=exp_error, dtype=np.float32)
            hf.create_dataset('log_mse', data=log_mse, dtype=np.float32)
            hf.create_dataset('log_mse_consensus',
                              data=log_mse_consensus, dtype=np.float32)
        return stddev, intensity, exp_error, correlation

    def predict_novelty(self, retrain=False, update_sign4=True):
        """Model novelty score via LocalOutlierFactor (semi-supervised).

        Args:
            retrain(bool): Drop old model and train again. (default: False)
            update_sign4(bool): Write novelty scores in h5. (default: True)

        """
        novelty_path = os.path.join(self.model_path, 'novelty')
        if not os.path.isdir(novelty_path):
            os.mkdir(novelty_path)
        novelty_model = os.path.join(novelty_path, 'lof.pkl')
        s2_inks = self.sign2_self.keys
        model = None
        if not os.path.isfile(novelty_model) or retrain:
            self.__log.debug('Training novelty score predictor')
            # fit on molecules available in sign2
            _, predicted = self.get_vectors(s2_inks, dataset_name='V')
            t0 = time()
            model = LocalOutlierFactor(novelty=True, metric='cosine',
                                       n_jobs=self.params['error']['cpu'])
            model.fit(predicted)
            delta = time() - t0
            self.__log.debug('Training took: %s' % delta)
            # serialize for later
            pickle.dump(model, open(novelty_model, 'wb'))
        if update_sign4:
            self.__log.debug('Updating novelty scores')
            if model is None:
                model = pickle.load(open(novelty_model, 'rb'))
            # get scores for known molecules and pair with indexes
            s2_idxs = np.argwhere(np.isin(list(self.keys), s2_inks,
                                          assume_unique=True))
            s2_novelty = model.negative_outlier_factor_
            s2_outlier = [0] * s2_novelty.shape[0]
            assert(s2_idxs.shape[0] == s2_novelty.shape[0])
            # predict scores for other molecules and pair with indexes
            s3_inks = sorted(self.unique_keys - set(s2_inks))
            s3_idxs = np.argwhere(np.isin(list(self.keys), s3_inks,
                                          assume_unique=True))
            _, s3_pred_sign = self.get_vectors(s3_inks)
            s3_novelty = model.score_samples(s3_pred_sign)
            s3_outlier = model.predict(s3_pred_sign)
            assert(len(s3_inks) == s3_novelty.shape[0])
            ordered_scores = np.array(sorted(
                zip(s2_idxs.flatten(), s2_novelty, s2_outlier) +
                zip(s3_idxs.flatten(), s3_novelty, s3_outlier)))
            ordered_novelty = ordered_scores[:, 1]
            ordered_outlier = ordered_scores[:, 2]
            # novelty goes from 0 to -inf, we take the absolute to have
            # most novel molecules with score 1.
            abs_novelty = np.abs(np.expand_dims(ordered_novelty, 1))
            nov_qtr = QuantileTransformer(
                n_quantiles=100000).fit(abs_novelty[:100000])
            nov_qtr_path = os.path.join(novelty_path, 'qtr.pkl')
            pickle.dump(nov_qtr, open(nov_qtr_path, 'wb'))
            with h5py.File(self.data_path, "r+") as results:
                if 'novelty' in results:
                    del results['novelty']
                results['novelty'] = ordered_novelty
                if 'novelty_norm' in results:
                    del results['novelty_norm']
                results['novelty_norm'] = nov_qtr.transform(
                    abs_novelty).flatten()
                if 'outlier' in results:
                    del results['outlier']
                results['outlier'] = ordered_outlier

    def predict(self, src_file, dst_file, src_h5_ds='x_test',
                dst_h5_ds='V', model_path=None, chunk_size=1000):
        from chemicalchecker.tool.siamese import SiameseTriplets

        if model_path is None:
            model_path = os.path.join(self.model_path, 'siamese_test')
        self.__log.info('PREDICTING using model from: %s' % model_path)
        self.__log.info('INPUT from: %s' % src_file)
        self.__log.info('OUTPUT goes to: %s' % dst_file)
        siamese = SiameseTriplets(model_path)
        with h5py.File(src_file, "r") as features:
            with h5py.File(dst_file, "w") as preds:
                # create destination h5 dataset
                tot_inks = features[src_h5_ds].shape[0]
                preds_shape = (tot_inks, 128)
                preds.create_dataset(dst_h5_ds, preds_shape, dtype=np.float32)
                # predict in chunks
                for idx in tqdm(range(0, tot_inks, chunk_size), desc='PRED'):
                    chunk = slice(idx, idx + chunk_size)
                    feat = features[src_h5_ds][chunk]
                    preds[dst_h5_ds][chunk] = siamese.predict(feat)


def safe_create(h5file, *args, **kwargs):
    if args[0] not in h5file:
        h5file.create_dataset(*args, **kwargs)


def mask_keep(idxs, x1_data):
    # we will fill an array of NaN with values we want to keep
    x1_data_transf = np.zeros_like(x1_data, dtype=np.float32) * np.nan
    for idx in idxs:
        # copy column from original data
        col_slice = slice(idx * 128, (idx + 1) * 128)
        x1_data_transf[:, col_slice] = x1_data[:, col_slice]
    # keep rows containing at least one not-NaN value
    not_nan = np.isfinite(x1_data_transf).any(axis=1)
    x1_data_transf = x1_data_transf[not_nan]
    return x1_data_transf


def mask_exclude(idxs, x1_data):
    x1_data_transf = np.copy(x1_data)
    for idx in idxs:
        # set current space to nan
        col_slice = slice(idx * 128, (idx + 1) * 128)
        x1_data_transf[:, col_slice] = np.nan
    # drop rows that only contain NaNs
    not_nan = np.isfinite(x1_data_transf).any(axis=1)
    x1_data_transf = x1_data_transf[not_nan]
    return x1_data_transf


def col_wise_correlation(X, Y, scaled=False):
    if scaled:
        X = robust_scale(X)
        Y = robust_scale(Y)
    return row_wise_correlation(X.T, Y.T)


def row_wise_correlation(X, Y, scaled=False):
    if scaled:
        X = robust_scale(X)
        Y = robust_scale(Y)
    var1 = (X.T - np.mean(X, axis=1)).T
    var2 = (Y.T - np.mean(Y, axis=1)).T
    cov = np.mean(var1 * var2, axis=1)
    return cov / (np.std(X, axis=1) * np.std(Y, axis=1))


def linear_pow2(val, slope=0.33, intercept=4.):
    flat_log_size = int(np.floor(val * slope))
    return np.power(2, int(flat_log_size + intercept))


def inverse_linear_pow2(val, slope=100, intercept=4.):
    flat_log_size = int(np.floor(slope / val))
    return np.power(2, int(flat_log_size + intercept))


def layer_size_heuristic(samples, features, clip=(512, 2048)):
    log_size = np.log(samples * features)
    pow2 = linear_pow2(log_size, slope=0.3, intercept=5.)
    return np.clip(pow2, *clip)


def batch_size_heuristic(samples, features, clip=(32, 256)):
    log_size = np.log(samples * features)
    pow2 = linear_pow2(log_size, slope=0.5, intercept=-2.)
    return np.clip(pow2, *clip)


def epoch_per_iteration_heuristic(samples, features, clip=(16, 1024)):
    log_size = np.log(samples * features)
    pow2 = inverse_linear_pow2(log_size, slope=270, intercept=-8)
    return np.clip(pow2, *clip)


def subsampling_probs(sign2_coverage, dataset_idx):
    cov = DataSignature(sign2_coverage).get_h5_dataset('V')
    no_self = cov[(cov[:, dataset_idx] == 0).flatten()]
    # chemistry spaces have very few not selfs
    if no_self.shape[0] < 100000:
        no_self = np.vstack((no_self, cov[:100000]))
    # how many dataset per molecule?
    nrs, freq_nrs = np.unique(
        np.sum(no_self, axis=1).astype(int), return_counts=True)
    # frequency based probabilities
    p_nrs = freq_nrs / no_self.shape[0]
    # add minimum probability (corner cases where to use 1 or 2 datasets)
    min_p_nr = np.full(cov.shape[1], min(p_nrs), dtype=np.float32)
    for nr, p_nr in zip(nrs, p_nrs):
        min_p_nr[nr] = p_nr
    # but leave out too large nrs
    min_p_nr[max(nrs):] = 0.0
    # normalize (sum of probabilities must be one)
    min_p_nr = min_p_nr / np.sum(min_p_nr)
    # print(np.log10(min_p_nr + 1e-10).astype(int))
    # probabilities to keep a dataset?
    p_keep = np.sum(no_self, axis=0) / no_self.shape[0]
    return min_p_nr, p_keep


def subsample(tensor, sign_width=128,
              p_nr=np.array([1 / 25.] * 25),
              p_only_self=0.0,
              p_self=0.1,
              dataset_idx=[0],
              p_keep=np.array([1 / 25.] * 25),
              **kwargs):
    """Function to subsample stacked data."""
    # it is safe to make a local copy of the input matrix
    new_data = np.copy(tensor)
    # we will have a masking matrix at the end
    mask = np.zeros_like(new_data).astype(bool)
    p_keep[dataset_idx] = 0.0
    # if new_data.shape[1] % sign_width != 0:
    #    raise Exception('All signature should be of length %i.' % sign_width)
    for idx, row in enumerate(new_data):
        # the following assumes the stacked signature have a fixed width
        presence = ~np.isnan(row[0::sign_width])
        # case where we show only the space itself
        if np.random.rand() < p_only_self:
            presence_add = np.full_like(presence, False)
            presence_add[dataset_idx] = True
            mask[idx] = np.repeat(presence_add, sign_width)
            continue
        # datasets that I can select
        present_idxs = np.argwhere(presence).flatten()
        # how many dataset at most?
        max_add = present_idxs.shape[0]
        # normalize nr dataset probabilities
        p_nr_row = p_nr[:max_add] / np.sum(p_nr[:max_add])
        # how many dataset are we keeping?
        nr_keep = np.random.choice(np.arange(1, len(p_nr_row) + 1), p=p_nr_row)
        # normalize dataset keep probabilities
        p_keep_row = p_keep[presence] / np.sum(p_keep[presence])
        nr_keep = np.min([nr_keep, np.sum(p_keep_row > 0)])
        # which ones?
        to_add = np.random.choice(
            present_idxs, nr_keep, p=p_keep_row, replace=False)
        if np.random.rand() < p_self:
            to_add = np.append(to_add, dataset_idx)
        # dataset mask
        presence_add = np.zeros(presence.shape).astype(bool)
        presence_add[to_add] = True
        # from dataset mask to signature mask
        mask[idx] = np.repeat(presence_add, sign_width)
    # make masked dataset NaN
    new_data[~mask] = np.nan
    return new_data


def plot_subsample(ds='B1.001', p_self=.1, p_only_self=0., limit=10000):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from chemicalchecker import ChemicalChecker
    from chemicalchecker.core.signature_data import DataSignature
    from chemicalchecker.core.sign4 import subsample, subsampling_probs

    cc = ChemicalChecker()
    s4 = cc.get_signature("sign3", "full", ds)
    s4_X = os.path.join(s4.model_path, 'train.h5')
    data_X = DataSignature(s4_X)
    shape_X = data_X.info_h5['x']
    limit = min(shape_X[0], limit)
    X = data_X.get_h5_dataset('x', mask=np.arange(limit))
    sign_width = 128

    dataset_idx = np.argwhere(
        np.isin(list(cc.datasets_exemplary()), ds)).flatten()
    sign2_coverage = '/aloy/web_checker/current/full/all_sign2_coverage.h5'
    p_nr, p_keep = subsampling_probs(sign2_coverage, dataset_idx)

    X1 = subsample(X, dataset_idx=[dataset_idx], p_nr=p_nr, p_keep=p_keep,
                   p_self=p_self, p_only_self=p_only_self)

    presence_X = ~np.isnan(X[:, 0::sign_width])
    presence_X1 = ~np.isnan(X1[:, 0::sign_width])

    _, freq_nr_X = np.unique(
        np.sum(presence_X, axis=1).astype(int), return_counts=True)
    _, freq_nr_X1 = np.unique(
        np.sum(presence_X1, axis=1).astype(int), return_counts=True)
    freq_nr_noself = p_nr * limit

    plt.figure(figsize=(12, 12), dpi=100)
    plt.subplot(2, 1, 1)
    plt.title('which dataset')
    plt.bar(np.arange(25) - 0.2, np.sum(presence_X, axis=0),
            width=.2, alpha=.8, label='original')
    plt.bar(np.arange(25) + 0.2, np.sum(presence_X1, axis=0),
            width=.2, alpha=.8, label='subsampled')
    plt.xlim(-1, 25)
    plt.xticks(np.arange(25), [x[:2] for x in cc.datasets_exemplary()])

    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('how many dataset')
    plt.bar(np.arange(1, len(freq_nr_X) + 1) - 0.2,
            freq_nr_X, width=.2, alpha=.8, label='original')
    plt.bar(np.arange(1, len(freq_nr_X1) + 1),
            freq_nr_X1, width=.2, alpha=.8, label='subsampled')
    plt.bar(np.arange(1, len(freq_nr_noself) + 1) + 0.2,
            freq_nr_noself, width=.2, alpha=.8, label='not self')
    plt.legend()
    plt.xticks(np.arange(26))
    plt.xlim(0, 26)

    plt.tight_layout()
    plt.savefig('subsample_dataset_%s_pself_%.2f_ponly_%.2f.png' %
                (ds, p_self, p_only_self))
    plt.close()
