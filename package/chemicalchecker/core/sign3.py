"""Signature type 3.

Network embedding inferred similarity networks. Their added
value, compared to signatures type 2, is that they can be derived for
virtually *any* molecule in *any* dataset.
"""
import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial

from .signature_base import BaseSignature

from chemicalchecker.util.plot import Plot
from chemicalchecker.util import logged


@logged
class sign3(BaseSignature):
    """Signature type 2 class."""

    def __init__(self, signature_path, validation_path, dataset, **params):
        """Initialize the signature.

        Args:
            signature_path(str): The signature root directory.
            dataset(`Dataset`): `chemicalchecker.database.Dataset` object.
            params(): Parameters, expected keys are 'graph', 'node2vec', and
                'adanet'.
        """
        # Calling init on the base class to trigger file existence checks
        BaseSignature.__init__(self, signature_path,
                               validation_path, dataset, **params)
        self.validation_path = validation_path
        # generate needed paths
        self.data_path = os.path.join(self.signature_path, 'sign3.h5')
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
        default_adanet = {
            "augmentation": subsample,
            "initial_architecture": [9, 1]
        }
        self.params['adanet'] = params.get('adanet', default_adanet)

    def save_sign2_matrix(self, chemchecker, destination):
        """Save combined matrix of stacked signature 2.

        Args:
            chemchecker(ChemicalChecker): The CC instance where to fetch
                signatures.
            destination(str): Path to the H5 destination file.
        """
        # get current space on which we'll train (using reference set to
        # limit redundancy)
        my_sign2 = chemchecker.get_signature('sign2', 'full', self.dataset)
        ref_dimension = my_sign2.shape[1]
        feat_shape = (my_sign2.shape[0],
                      ref_dimension * len(self.meshed_datasets))
        with h5py.File(destination, 'w') as hf:
            hf.create_dataset('y', data=my_sign2[:], dtype=np.float32)
            hf.create_dataset('x', feat_shape, dtype=np.float32)
            for idx, ds in enumerate(self.meshed_datasets):
                sign2_ds = chemchecker.get_signature('sign2', 'full', ds)
                _, signs = sign2_ds.get_vectors(
                    my_sign2.keys, include_nan=True)
                col_slice = slice(ref_dimension * idx,
                                  ref_dimension * (idx + 1))
                hf['x_train'][:, col_slice] = signs
                available = np.isin(my_sign2.keys, sign2_ds.keys)
                self.__log.info('%s shared molecules between %s and %s',
                                sum(available), self.dataset, ds)
                del signs

    def _learn(self, chemchecker, reuse=True, suffix=None, evaluate=True,
               single_spaces_performances=True):
        """Learn the signature 3 model.

        chemchecker(ChemmChecker): The CC instance which allow fetching all
            signature 2.
        reuse(bool): Whether to reuse intermediate files (e.g. the aggregated
            signature 2 matrix).
        suffix(str): a suffix for the adanet model path (e.g.
            'sign3/models/adanet_<suffix>').
        evaluate(bool): Whether we are performing a train-test split and
            evaluating the performances (N.B. this is required for complete
            confidence scores)
        single_spaces_performances(bool): Whether to check performance of the
            trained network with incomplete input.
        """
        try:
            from chemicalchecker.tool.adanet import AdaNet, Traintest
        except ImportError as err:
            raise err
        # adanet
        self.__log.debug('AdaNet fit %s based on other sign2', self.dataset)
        # get params and set folder
        adanet_params = self.params['adanet']
        if suffix:
            adanet_path = os.path.join(self.model_path, 'adanet_%s' % suffix)
        else:
            adanet_path = os.path.join(self.model_path, 'adanet')
        if adanet_params:
            if 'model_dir' in adanet_params:
                adanet_path = adanet_params.pop('model_dir')
        if not reuse or not os.path.isdir(adanet_path):
            os.makedirs(adanet_path)
        # prepare train-test file
        sign2_matrix = os.path.join(self.model_path, 'train.h5')
        if not reuse or os.path.isfile(sign2_matrix):
            self.save_sign2_matrix(chemchecker, sign2_matrix)
        if evaluate:
            traintest_file = os.path.join(self.model_path, 'traintest.h5')
            if adanet_params:
                traintest_file = adanet_params.pop(
                    'traintest_file', traintest_file)
            if not reuse or not os.path.isfile(traintest_file):
                Traintest.split_h5(sign2_matrix, traintest_file)
        else:
            traintest_file = sign2_matrix
            if adanet_params:
                traintest_file = adanet_params.pop(
                    'traintest_file', traintest_file)
        # initialize adanet
        if adanet_params:
            ada = AdaNet(model_dir=adanet_path,
                         traintest_file=traintest_file,
                         **adanet_params)
        else:
            ada = AdaNet(model_dir=adanet_path, traintest_file=traintest_file)
        # learn
        self.__log.debug('AdaNet training on %s' % traintest_file)
        ada.train_and_evaluate(evaluate=evaluate)
        self.__log.debug('model saved to %s' % adanet_path)
        if evaluate:
            # compare performance with cross predictors
            if single_spaces_performances:
                singles = self.adanet_single_spaces(chemchecker, adanet_path,
                                                    traintest_file, suffix)
            else:
                singles = None
            # save AdaNet performances and plots
            sign2_plot = Plot(self.dataset, adanet_path, self.validation_path)
            ada.save_performances(adanet_path, sign2_plot, suffix, singles)

    def fit_sign0(self, chemchecker, ds='A1.001', sign0_traintest=None):
        """Train an AdaNet model to predict sign3 from sign0."""
        try:
            from chemicalchecker.tool.adanet import AdaNet, Traintest
        except ImportError as err:
            raise err

        # build input matrix if not provided
        if sign0_traintest is None:
            sign0_traintest = os.path.join(
                self.model_path, 'traintest_sign0_%s.h5' % ds)
        if not os.path.isfile(sign0_traintest):
            s0 = chemchecker.get_signature('sign0', 'full', ds)
            common_keys, features = s0.get_vectors(self.keys)
            _, labels = self.get_vectors(common_keys)
            # we also want to learn how to predict confidence scores
            mask = np.isin(self.keys, list(common_keys), assume_unique=True)
            stddev = self.get_h5_dataset('stddev_norm', mask)
            stddev = np.expand_dims(stddev, 1)
            intensity = self.get_h5_dataset('intensity_norm', mask)
            intensity = np.expand_dims(intensity, 1)
            confidence = self.get_h5_dataset('confidence', mask)
            confidence = np.expand_dims(confidence, 1)
            # so they become part of the supervised learning
            labels = np.hstack((labels, stddev, intensity, confidence))
            Traintest.create(features, labels, sign0_traintest)
        self.params['adanet'] = {
            'traintest_file': sign0_traintest,
            'augmentation': False}
        self._learn(chemchecker, suffix='sign0_%s_final_eval' %
                    ds, evaluate=True, single_spaces_performances=False)

    def predict_from_smiles(self, smiles, dest_dir, ds='A1.001'):
        """Given smiles generate sign0 and predict sign3.

        Args:
            smiles(list): A list of SMILES strings.
            dest_dir(str): the path where to save the predictions.
        Returns:
            pred_s3(sign3): The signature 3 created with predictions.
        """
        try:
            from chemicalchecker.tool.adanet import AdaNet
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError as err:
            raise err
        # load NN
        sign0_adanet_path = os.path.join(self.model_path,
                                         'adanet_sign0_%s_final_eval' % ds,
                                         'savedmodel')
        predict_fn = AdaNet.predict_fn(sign0_adanet_path)
        # create a sign3 in the destination dir
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        pred_s3 = sign3(dest_dir, dest_dir, self.dataset)
        with h5py.File(pred_s3.data_path, "w") as results:
            # initialize V (with NaN in case of failing rdkit) and smiles keys
            results.create_dataset('keys', data=np.array(smiles))
            results.create_dataset('V', (len(smiles), 128), dtype=np.float32)
            results.create_dataset(
                'stddev_norm', (len(smiles), ), dtype=np.float32)
            results.create_dataset(
                'intensity_norm', (len(smiles), ), dtype=np.float32)
            results.create_dataset(
                'confidence', (len(smiles), ), dtype=np.float32)
            # compute sign0
            nBits = 2048
            radius = 2
            for chunk in tqdm(list(pred_s3.chunker())):
                sign0s = list()
                failed = list()
                for idx, mol_smiles in enumerate(smiles[chunk]):
                    try:
                        mol = Chem.MolFromSmiles(mol_smiles)
                        if mol is None:
                            raise Exception("Cannot get molecule from smiles.")
                        info = {}
                        fp = AllChem.GetMorganFingerprintAsBitVect(
                            mol, radius, nBits=nBits, bitInfo=info)
                        bin_s0 = [fp.GetBit(i) for i in range(fp.GetNumBits())]
                        calc_s0 = np.array(bin_s0).astype(np.float32)
                    except Exception as err:
                        self.__log.warn("%s: %s", mol_smiles, str(err))
                        failed.append(idx)
                        calc_s0 = np.full((nBits, ),  np.nan)
                    finally:
                        sign0s.append(calc_s0)
                sign0s = np.vstack(sign0s)
                preds = predict_fn({'x': sign0s})['predictions']
                if failed:
                    preds[np.array(failed)] = np.full((128, ),  np.nan)
                results['V'][chunk] = preds[:, :128]
                results['stddev_norm'][chunk] = preds[:, 128]
                results['intensity_norm'][chunk] = preds[:, 129]
                results['confidence'][chunk] = preds[:, 130]
        return pred_s3

    @staticmethod
    def save_sign2_universe(chemchecker, destination, datasets=None):
        """Create a file with all signatures 2 for each molecule in the CC.

        Args:
            chemchecker(ChemicalChecker): The CC object where to fetch
                signature 2.
            destination(str): Path where the H5 is saved.
            datasets(list): List of datasets to join.
        """
        # get sorted universe inchikeys and CC signatures
        inchikeys = set()
        ds_sign = dict()
        if datasets is None:
            datasets = chemchecker.datasets
        for ds in datasets:
            sign = chemchecker.get_signature('sign2', 'full', ds)
            inchikeys.update(sign.unique_keys)
            ds_sign[ds] = sign
        inchikeys = sorted(list(inchikeys))
        tot_inks = len(inchikeys)
        tot_ds = len(ds_sign)
        # build input matrix if not provided
        if not os.path.isfile(destination):
            with h5py.File(destination, "w") as fh:
                fh.create_dataset('x_test',
                                  (tot_inks, 128 * tot_ds),
                                  dtype=np.float32)
                for idx, (ds, sign) in enumerate(ds_sign.items()):
                    _, vectors = sign.get_vectors(inchikeys, include_nan=True)
                    fh['x_test'][:, idx * 128:(idx + 1) * 128] = vectors
                    del vectors

    def fit(self, chemchecker, sign2_universe=None, model_confidence=True,
            save_support=True, save_correlations=True, update_preds=True,
            meshed_datasets=None):
        """Use the learned model to predict the signature 3.

        Args:
        chemchecker(ChemicalChecker): the CC instance for fetching signatures.
        sign2_universe(str): Path where to save the union of all signatures 2.
        model_confidence(bool): Whether to model confidence. That is  based on
            standard deviation of 10 samples predicted with dropout.
        save_support(bool): Whether to save the number of dataset used for
            a prediction.
        save_correlations(bool) Whether to save the correlation (average,
            tertile, max) for the given input dataset (based on evaluation).
        """
        try:
            from chemicalchecker.tool.adanet import AdaNet
        except ImportError as err:
            raise err

        if meshed_datasets is None:
            self.meshed_datasets = list(chemchecker.datasets)
        # check if have performance evaluations
        eval_stats = os.path.join(
            self.model_path, 'adanet_final_eval', 'stats.pkl')
        if not os.path.isfile(eval_stats):
            self._learn(chemchecker, suffix='final_eval', evaluate=True)

        # check if we have the final trained model
        final_adanet_path = os.path.join(self.model_path, 'adanet_final',
                                         'savedmodel')
        if not os.path.isdir(final_adanet_path):
            self._learn(chemchecker, suffix='final', evaluate=False)

        # get sorted universe inchikeys and CC signatures
        inchikeys = set()
        ds_sign = dict()
        for ds in self.meshed_datasets:
            sign = chemchecker.get_signature('sign2', 'full', ds)
            inchikeys.update(sign.unique_keys)
            ds_sign[ds] = sign
        inchikeys = sorted(list(inchikeys))
        tot_inks = len(inchikeys)
        tot_ds = len(ds_sign)

        # build input matrix if not provided
        if not os.path.isfile(sign2_universe):
            sign3.save_sign2_universe(
                chemchecker, sign2_universe, self.meshed_datasets)

        def safe_create(h5file, *args, **kwargs):
            if args[0] not in h5file:
                h5file.create_dataset(*args, **kwargs)

        # save universe sign3
        predict_fn = AdaNet.predict_fn(final_adanet_path)
        with h5py.File(self.data_path, "w") as results:
            # initialize V and keys datasets
            safe_create(results, 'V', (tot_inks, 128), dtype=np.float32)
            safe_create(results, 'keys', (tot_inks,), dtype='|S27')
            if model_confidence:
                # the actual confidence value will be stored here
                safe_create(results, 'confidence',
                            (tot_inks,), dtype=np.float32)
                # this is to store standard deviations
                safe_create(results, 'stddev', (tot_inks,), dtype=np.float32)
                safe_create(results, 'stddev_norm',
                            (tot_inks,), dtype=np.float32)
                # this is to store intensity
                safe_create(results, 'intensity',
                            (tot_inks,), dtype=np.float32)
                safe_create(results, 'intensity_norm',
                            (tot_inks,), dtype=np.float32)
                # consensus prediction
                safe_create(results, 'consensus',
                            (tot_inks, 128), dtype=np.float32)
            if save_support:
                # this will e number of available sign2 for given molecules
                safe_create(results, 'support', (tot_inks,), dtype=np.float32)
            if save_correlations:
                # read the correlations obtained evaluating on single spaces
                # the ration between test and train gives me an idea of how
                # well we can generalize
                df = pd.read_pickle(eval_stats)
                test_eval = df[(df.split != 'train') & (
                    df.algo == 'AdaNet_final_eval')]
                train_eval = df[(df.split == 'train') & (
                    df.algo == 'AdaNet_final_eval')]
                avg_pearsons = np.zeros(tot_ds, dtype=np.float32)
                avg_pearsons_test = np.zeros(tot_ds, dtype=np.float32)
                avg_pearsons_train = np.zeros(tot_ds, dtype=np.float32)
                for idx, ds in enumerate(self.meshed_datasets):
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
                safe_create(results, 'dataset_correlation', data=avg_pearsons)
                # Average/tertile/maximum Pearson correlations will be saved
                safe_create(results, 'pred_correlation', (tot_inks, 3),
                            dtype=np.float32)

            # predict signature 3 for universe molecules
            with h5py.File(sign2_universe, "r") as features:
                # reference no information prediction
                zero_feat = np.zeros(
                    (1, features['x_test'].shape[1]), dtype=np.float32)
                zero_pred = predict_fn({'x': zero_feat})['predictions']
                # read input in chunks
                for idx in tqdm(range(0, tot_inks, 1000)):
                    chunk = slice(idx, idx + 1000)
                    feat = features['x_test'][chunk]
                    # predict with final model
                    if update_preds:
                        results['V'][chunk] = AdaNet.predict(feat, predict_fn)
                    results['keys'][chunk] = inchikeys[chunk]
                    # compute support
                    if save_support:
                        support = np.sum(~np.isnan(feat[:, 0::128]), axis=1)
                        results['support'][chunk] = support
                    # lookup correlations
                    if save_correlations:
                        presence = ~np.isnan(feat[:, 0::128])
                        chunk_corr = np.zeros(
                            (presence.shape[0], 3), dtype=np.float32)
                        for row_id, row in enumerate(presence):
                            available_pearsons = avg_pearsons[row]
                            chunk_corr[row_id] = [
                                np.mean(available_pearsons),
                                np.percentile(
                                    available_pearsons, 100 * (2 / 3.)),
                                np.max(available_pearsons)
                            ]
                        results['pred_correlation'][chunk] = chunk_corr
                        del chunk_corr
                    # save stddevs needed for confidence model and prediction
                    if model_confidence and update_preds:
                        # draw prediction with sub-sampling (dropout)
                        samples = AdaNet.predict(feat, predict_fn,
                                                 subsample_x_only,
                                                 probs=True, samples=10)
                        # summarize the predictions as consensus
                        consensus = np.mean(samples, axis=2)
                        results['consensus'][chunk] = consensus
                        # summarize the standard deviation of components
                        stddevs = np.std(samples, axis=2)
                        # just save the average stddev over the components
                        results['stddev'][chunk] = np.mean(stddevs, axis=1)
                        # zeros input (no info) as intensity reference
                        centered = consensus - zero_pred
                        # measure the intensity (absolute sum of comps)
                        abs_sum = np.sum(np.abs(centered), axis=1)
                        results['intensity'][chunk] = abs_sum / 128.
        # train error estimator as save prediction
        std_dist, int_dist = self.train_error_estimator(ds_sign[self.dataset])
        with h5py.File(self.data_path, "r+") as results:
            safe_create(results, 'confidence', (tot_inks, 1), dtype=np.float32)
            for idx in tqdm(range(0, tot_inks, 1000)):
                chunk = slice(idx, idx + 1000)
                # get stddev and normalize wrt train distribution
                stddev = np.expand_dims(results['stddev'][chunk], axis=1)
                stddev_norm = np.count_nonzero(stddev > std_dist.T, axis=1)
                stddev_norm = stddev_norm / float(std_dist.shape[0])
                results['stddev_norm'][chunk] = stddev_norm
                # get intensity and normalize wrt train distribution
                intensity = np.expand_dims(results['intensity'][chunk], axis=1)
                inten_norm = np.count_nonzero(intensity > int_dist.T, axis=1)
                inten_norm = inten_norm / float(int_dist.shape[0])
                results['intensity_norm'][chunk] = inten_norm
                # confidence and intensity are equally important
                # high intensity imply low error, high stddev imply high error
                confidence = np.sqrt(inten_norm * (1 - stddev_norm))
                # the higher the better
                results['confidence'][chunk] = confidence

        self.validate()
        self.mark_ready()

    def train_error_estimator(self, sign2_self):
        """Train a error estimator.

        After some testing we realized that training a regressor does not
        accurately capture indefinite prediction (low intensity). We just
        save the distributions of stddevs and intensities.

        Args:
            sign2_self(sign2): The signature that we are training for.
        Returns:
            stddev(np.array): Distribution of standard deviations.
            intensity(np.array): Distribution of intensity.
        """
        # get current space inchikeys (limit to 20^4)
        dataset_inks = sign2_self.keys
        if len(dataset_inks) > 2 * 1e4:
            dataset_inks = np.random.choice(dataset_inks, int(2 * 1e4))
        # get the features to train the estimator on
        _, stddev = self.get_vectors(dataset_inks, dataset_name='stddev')
        _, intensity = self.get_vectors(dataset_inks, dataset_name='intensity')
        # also get the consensus and actual sign2
        _, consensus = self.get_vectors(dataset_inks, dataset_name='consensus')
        _, actual = sign2_self.get_vectors(dataset_inks)
        # calculate the error (what we want to predict)
        log_mse = np.log10(np.average(((actual - consensus)**2), axis=1))
        # save data in the confidence model
        error_file = os.path.join(self.model_path, 'error.h5')
        with h5py.File(error_file, "w") as hf:
            hf.create_dataset('keys', data=dataset_inks)
            hf.create_dataset('stddev', data=stddev, dtype=np.float32)
            hf.create_dataset('intensity', data=intensity, dtype=np.float32)
            hf.create_dataset('log_mse', data=log_mse, dtype=np.float32)
        return stddev, intensity

    def predict(self, chemchecker, output_file, sign2_molset=None,
                inchikeys=None, save_confidence=True, save_support=True,
                save_correlations=True):
        """Use the learned model to predict the signature 3.

        chemchecker(ChemicalChecker): the CC instance for fetching signatures.
        sign2_molset(str): Path where to save the union of all signatures 2.
        model_confidence(bool): Whether to model confidence. That is  based on
            standard deviation of 10 samples predicted with dropout.
        save_support(bool): Whether to save the number of dataset used for
            a prediction.
        save_correlations(bool) Whether to save the correlation (average,
            tertile, max) for the given input dataset (based on evaluation).
        """
        try:
            from chemicalchecker.tool.adanet import AdaNet
        except ImportError as err:
            raise err

        # get sorted universe inchikeys and CC signatures
        if inchikeys is None:
            inchikeys = set()
            ds_sign = dict()
            for ds in self.meshed_datasets:
                sign = chemchecker.get_signature('sign2', 'full', ds)
                inchikeys.update(sign.unique_keys)
                ds_sign[ds] = sign
            inchikeys = sorted(list(inchikeys))

        # build input matrix if not provided
        if not os.path.isfile(sign2_molset):
            with h5py.File(sign2_molset, "w") as fh:
                fh.create_dataset('x_test',
                                  (len(inchikeys), 128 * len(ds_sign)),
                                  dtype=np.float32)
                for idx, (ds, sign) in enumerate(ds_sign.items()):
                    vectors = sign.get_vectors(inchikeys, include_nan=True)
                    fh['x_test'][:, idx * 128:(idx + 1) * 128] = vectors
                    del vectors

        # load confidence model if needed
        if save_confidence:
            confidence_file = os.path.join(self.model_path, 'conf.h5')
            with h5py.File(confidence_file, "r") as confidence_model:
                stddev_dist = confidence_model['stddev_dist'][:]

        # save sign3 predictions
        final_adanet_path = os.path.join(self.model_path, 'adanet_final',
                                         'savedmodel')
        predict_fn = AdaNet.predict_fn(final_adanet_path)
        with h5py.File(output_file, "w") as results:
            # initialize V and keys datasets
            results.create_dataset(
                'V', (len(inchikeys), 128), dtype=np.float32)
            results.create_dataset('keys', (len(inchikeys),), dtype='|S27')
            if save_confidence:
                # the actual confidence value will be stored here
                results.create_dataset(
                    'confidence', (len(inchikeys),), dtype=np.float32)
            if save_support:
                # this will e number of available sign2 for given molecules
                results.create_dataset(
                    'support', (len(inchikeys),), dtype=np.float32)
            if save_correlations:
                # this is to lookup correlations
                with h5py.File(self.data_path, "r") as hf:
                    avg_pearsons = hf['dataset_correlation']
                # Average/tertile/maximum Pearson correlations will be saved
                results.create_dataset('pred_correlation', (len(inchikeys), 3),
                                       dtype=np.float32)

            # predict signature 3 for universe molecules
            with h5py.File(sign2_molset, "r") as features:
                # read input in chunks
                for idx in tqdm(range(0, len(inchikeys), 1000)):
                    chunk = slice(idx, idx + 1000)
                    feat = features['x_test'][chunk]
                    # predict with final model
                    results['V'][chunk] = AdaNet.predict(feat, predict_fn)
                    results['keys'][chunk] = inchikeys[chunk]
                    # compute support
                    if save_support:
                        support = np.sum(~np.isnan(feat[:, 0::128]), axis=1)
                        results['support'][chunk] = support
                    # lookup correlations
                    if save_correlations:
                        presence = ~np.isnan(feat[:, 0::128])
                        chunk_corr = np.zeros(
                            (presence.shape[0], 3), dtype=np.float32)
                        for row_id, row in enumerate(presence):
                            available_pearsons = avg_pearsons[row]
                            chunk_corr[row_id] = [
                                np.mean(available_pearsons),
                                np.percentile(
                                    available_pearsons, 2 * 100 / 3.),
                                np.max(available_pearsons)
                            ]
                        results['pred_correlation'][chunk] = chunk_corr
                        del chunk_corr
                    # save stddevs needed for confidence model and prediction
                    if save_confidence:
                        stddevs = AdaNet.predict(feat, predict_fn,
                                                 subsample_x_only,
                                                 probs=True, samples=10)
                        pred_stds = np.mean(stddevs, axis=1)
                        # how's is the stddev compared to our distribution?
                        conf_pred = np.count_nonzero(
                            pred_stds < stddev_dist, axis=1)
                        conf_pred = conf_pred / float(stddev_dist.shape[1])
                        results['confidence'][chunk] = conf_pred
                        del stddevs

    def adanet_single_spaces(self, chemchecker, adanet_path, traintest_file, suffix):
        """Prediction of adanet using single space signatures.

        We want to compare the performances of trained adanet to those of
        predictors based on single space.
        This is done filling the matrix with zeros in other spaces.
        """
        try:
            from chemicalchecker.tool.adanet import AdaNet, Traintest
        except ImportError as err:
            raise err

        def mask_keep(idxs, x_data, y_data):
            # we will fill an array of NaN with values we want to keep
            x_data_transf = np.zeros_like(x_data, dtype=np.float32) * np.nan
            for idx in idxs:
                # copy column from original data
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x_data_transf[:, col_slice] = x_data[:, col_slice]
            # keep rows containing at least one not-NaN value
            not_nan = np.isfinite(x_data_transf).any(axis=1)
            x_data_transf = x_data_transf[not_nan]
            y_data_transf = y_data[not_nan]
            return x_data_transf, y_data_transf

        def mask_exclude(idxs, x_data, y_data):
            x_data_transf = np.copy(x_data)
            for idx in idxs:
                # set current space to nan
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x_data_transf[:, col_slice] = np.nan
            # drop rows that only contain NaNs
            not_nan = np.isfinite(x_data_transf).any(axis=1)
            x_data_transf = x_data_transf[not_nan]
            y_data_transf = y_data[not_nan]
            return x_data_transf, y_data_transf

        def predict_and_save(name, idxs, save_dir, traintest_file, split,
                             predict_fn, mask_fn, adanet_path, total_size):
            # call predict
            self.__log.info("Predicting for: %s", name)
            y_pred, y_true = AdaNet.predict_online(
                traintest_file, split,
                predict_fn=predict_fn,
                mask_fn=partial(mask_fn, idxs))
            self.__log.info("%s Y: %s", name, y_pred.shape)
            if y_pred.shape[0] < 4:
                return
            file_true = os.path.join(
                adanet_path, "_".join(list(name) + [split, 'true']))
            np.save(file_true, y_true)
            y_true_shape = y_true.shape[0]
            del y_true
            file_pred = os.path.join(
                adanet_path, "_".join(list(name) + [split, 'pred']))
            np.save(file_pred, y_pred)
            del y_pred
            result = dict()
            result['true'] = file_true
            result['pred'] = file_pred
            result['coverage'] = y_true_shape / total_size
            result['time'] = 0.
            return result

        # get predict function (loads the neural network)
        self.__log.info("Loading AdaNet model")
        save_dir = os.path.join(adanet_path, 'savedmodel')
        predict_fn = AdaNet.predict_fn(save_dir)
        # get results for each split
        results = dict()
        all_dss = list(self.meshed_datasets)
        for split in ['train', 'test', 'validation']:
            traintest = Traintest(traintest_file, split)
            x_shape, y_shape = traintest.get_xy_shapes()
            total_size = float(y_shape[0])
            self.__log.info("%s X: %s Y: %s", split, x_shape, y_shape)
            self.__log.info("%s Y: %s", split, y_shape)

            # make prediction keeping only a single space
            for idx, ds in enumerate(all_dss):
                # algo name, prepare results
                name = ("AdaNet", ds)
                if suffix:
                    name = ("AdaNet_%s" % suffix, ds)
                if name not in results:
                    results[name] = dict()
                # predict and save
                results[name][split] = predict_and_save(name, [idx], save_dir,
                                                        traintest_file, split,
                                                        predict_fn, mask_keep,
                                                        adanet_path,
                                                        total_size)
            # make prediction excluding space to predict
            ds = self.dataset
            idx = all_dss.index(ds)
            # algo name, prepare results
            name = ("AdaNet", "not-%s" % ds)
            if suffix:
                name = ("AdaNet_%s" % suffix, "not-%s" % ds)
            if name not in results:
                results[name] = dict()
            # predict and save
            results[name][split] = predict_and_save(name, [idx], save_dir,
                                                    traintest_file, split,
                                                    predict_fn, mask_exclude,
                                                    adanet_path, total_size)
            # exclude level to predict
            dss = [d for d in all_dss if d.startswith(self.dataset[0])]
            idxs = [all_dss.index(d) for d in dss]
            # algo name, prepare results
            name = ("AdaNet", "not-%sX" % self.dataset[0])
            if suffix:
                name = ("AdaNet_%s" % suffix, "not-%sX" % self.dataset[0])
            if name not in results:
                results[name] = dict()
            # predict and save
            results[name][split] = predict_and_save(name, idxs, save_dir,
                                                    traintest_file, split,
                                                    predict_fn, mask_exclude,
                                                    adanet_path, total_size)
            # check special combinations
            dss = [d for d in all_dss if d.startswith('B')] + \
                [d for d in all_dss if d.startswith('C')]
            idxs = [all_dss.index(d) for d in dss]
            # algo name, prepare results
            name = ("AdaNet", "not-BX|CX")
            if suffix:
                name = ("AdaNet_%s" % suffix, "not-BX|CX")
            if name not in results:
                results[name] = dict()
            # predict and save
            results[name][split] = predict_and_save(name, idxs, save_dir,
                                                    traintest_file, split,
                                                    predict_fn, mask_exclude,
                                                    adanet_path, total_size)
        return results


def subsample_x_only(tensor, label=None):
    return subsample(tensor, label=None)[0]


def subsample(tensor, label):
    """Function to subsample stacked data."""
    # it is safe to make a local copy of the input matrix
    new_data = np.copy(tensor)
    # we will have a masking matrix at the end
    mask = np.zeros_like(new_data).astype(bool)
    for idx, row in enumerate(new_data):
        # the following assume the stacked signature to have a fixed width
        presence = ~np.isnan(row[0::128])
        # low probability of keeping the original sample
        if np.random.rand() > 0.95:
            presence_add = presence
        else:
            # present datasets
            present_idxs = np.argwhere(presence).flatten()
            # how many dataset in this subsampling?
            max_add = present_idxs.shape[0]
            n_to_add = np.random.choice(max_add) + 1
            # which ones?
            to_add = np.random.choice(
                present_idxs, n_to_add, replace=False)
            # dataset mask
            presence_add = np.zeros(presence.shape).astype(bool)
            presence_add[to_add] = True
        # from dataset mask to signature mask
        mask[idx] = np.repeat(presence_add, 128)
    # make masked dataset NaN
    new_data[~mask] = np.nan
    return new_data, label
