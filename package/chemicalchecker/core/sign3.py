"""Signature type 3.

Inferred network embedding of similarity networks. Their added
value, compared to signatures type 2, is that they can be derived for
virtually *any* molecule in *any* dataset.
"""
import os
import h5py
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from functools import partial
from scipy.stats import pearsonr, norm, rankdata

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import QuantileTransformer

from .signature_base import BaseSignature
from .signature_data import DataSignature

from chemicalchecker.util.plot import Plot
from chemicalchecker.util import logged
from chemicalchecker.util.splitter import Traintest


@logged
class sign3(BaseSignature, DataSignature):
    """Signature type 3 class."""

    def __init__(self, signature_path, dataset, **params):
        """Initialize the signature.

        Args:
            signature_path(str): The signature root directory.
            dataset(`Dataset`): `chemicalchecker.database.Dataset` object.
            params(): Parameters, expected keys are 'graph', 'node2vec', and
                'adanet'.
        """
        # Calling init on the base class to trigger file existence checks
        BaseSignature.__init__(self, signature_path,
                               dataset, **params)
        # generate needed paths
        self.data_path = os.path.join(self.signature_path, 'sign3.h5')
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
        default_adanet = {
            'adanet_iterations': 10,
            'shuffles': 5,
            'augmentation': subsample,
            'subnetwork_generator': 'StackDNNGenerator',
            'cpu': params.get('cpu', 4)
        }
        default_adanet.update(params.get('adanet', {}))
        self.params['adanet'] = default_adanet
        default_sign0 = {
            'adanet_iterations': 10,
            'augmentation': False,
            'subnetwork_generator': 'StackDNNGenerator',
            'cpu': params.get('cpu', 4)
        }
        default_sign0.update(params.get('sign0', {}))
        self.params['sign0'] = default_sign0
        default_err = {
            'adanet_iterations': 1,
            'augmentation': False,
            'layer_size': 25,
            'initial_architecture': [3, 2],
            'subnetwork_generator': 'ExtendDNNGenerator',
            'cpu': params.get('cpu', 4)
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
        sign3.__log.info("Generating signature 2 universe matrix.")
        inchikeys = set()
        for sign in sign2_list:
            inchikeys.update(sign.unique_keys)
        inchikeys = sorted(list(inchikeys))
        tot_inks = len(inchikeys)
        tot_ds = len(sign2_list)
        # build matrix stacking horizontally signature
        if os.path.isfile(destination):
            sign3.__log.warning("Skipping as destination %s already exists." %
                                destination)
            return
        with h5py.File(destination, "w") as fh:
            fh.create_dataset('x_test', (tot_inks, 128 * tot_ds),
                              dtype=np.float32)
            for idx, sign in enumerate(sign2_list):
                sign3.__log.info("Fetching from %s" % sign.data_path)
                # including NaN we have the correct number of molecules
                _, vectors = sign.get_vectors(inchikeys, include_nan=True)
                fh['x_test'][:, idx * 128:(idx + 1) * 128] = vectors
                del vectors

    @staticmethod
    def save_sign2_coverage(sign2_list, destination):
        """Create a file with all signatures 2 coverage of molecule in the CC.

        Args:
            sign2_list(list): List of signature 2 objects to learn from.
            destination(str): Path where the H5 is saved.
        """
        # get sorted universe inchikeys and CC signatures
        sign3.__log.info("Generating signature 2 coverage matrix.")
        inchikeys = set()
        for sign in sign2_list:
            inchikeys.update(sign.unique_keys)
        inchikeys = sorted(list(inchikeys))
        tot_inks = len(inchikeys)
        tot_ds = len(sign2_list)
        sign3.__log.info("Saving coverage for %s dataset and %s molecules." %
                         (tot_ds, tot_inks))
        # build matrix stacking horizontally signature
        if os.path.isfile(destination):
            sign3.__log.warning("Skipping as destination %s already exists." %
                                destination)
            return
        with h5py.File(destination, "w") as fh:
            fh.create_dataset('x_test', (tot_inks, tot_ds), dtype=np.float32)
            for idx, sign in enumerate(sign2_list):
                sign3.__log.info("Fetching from %s" % sign.data_path)
                # including NaN we have the correct number of molecules
                coverage = np.isin(inchikeys, sign.keys, assume_unique=True)
                sign3.__log.info("%s has %s Signature 2." %
                                 (sign.dataset, np.count_nonzero(coverage)))
                fh['x_test'][:, idx:(idx + 1)] = np.expand_dims(coverage, 1)

    def save_sign2_matrix(self, sign2_list, destination):
        """Save matrix of horizontally stacked signature 2.

        This is the matrix for training the signature 3. It is defined for all
        molecules or which we have a signature 2 in the current space, i.e. the
        Y. The X is the collections of signature 2 from other spaces
        horizontally stacked (and NaN filled).

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
            hf.create_dataset('y', data=self.sign2_self[
                              :], dtype=np.float32)
            hf.create_dataset('x', feat_shape, dtype=np.float32)
            # for each dataset fetch signatures for the molecules of current
            # space, if missing we add NaN.
            for idx, (ds, sign) in enumerate(zip(self.src_datasets, sign2_list)):
                _, signs = sign.get_vectors(
                    self.sign2_self.keys, include_nan=True)
                col_slice = slice(ref_dimension * idx,
                                  ref_dimension * (idx + 1))
                hf['x'][:, col_slice] = signs
                available = np.isin(self.sign2_self.keys, sign.keys)
                self.__log.info('%s shared molecules between %s and %s',
                                sum(available), self.dataset, ds)
                del signs

    def learn_sign2(self, adanet_params, reuse=True, suffix=None,
                    evaluate=True):
        """Learn the signature 3 model.

        This method is used twice. First to evaluate the performances of the
        AdaNet model. Second to train the final model on the full set of data.

        Args:
            adanet_params(dict): Dictionary with algorithm parameters.
            reuse(bool): Whether to reuse intermediate files (e.g. the
                aggregated signature 2 matrix).
            suffix(str): A suffix for the AdaNet model path (e.g.
                'sign3/models/adanet_<suffix>').
            evaluate(bool): Whether we are performing a train-test split and
                evaluating the performances (N.B. this is required for complete
                confidence scores)
        """
        try:
            from chemicalchecker.tool.adanet import AdaNet
        except ImportError as err:
            raise err
        # get params and set folder
        if suffix:
            adanet_path = os.path.join(self.model_path, 'adanet_%s' % suffix)
        else:
            adanet_path = os.path.join(self.model_path, 'adanet')
        if 'model_dir' in adanet_params:
            adanet_path = adanet_params.pop('model_dir')
        if not reuse or not os.path.isdir(adanet_path):
            os.makedirs(adanet_path)
        # generate input matrix
        sign2_matrix = os.path.join(self.model_path, 'train.h5')
        if not reuse or not os.path.isfile(sign2_matrix):
            self.save_sign2_matrix(self.sign2_list, sign2_matrix)
        # if evaluating, perform the train-test split
        if evaluate:
            traintest_file = os.path.join(self.model_path, 'traintest.h5')
            traintest_file = adanet_params.pop(
                'traintest_file', traintest_file)
            if not reuse or not os.path.isfile(traintest_file):
                Traintest.split_h5_blocks(sign2_matrix, traintest_file)
        else:
            traintest_file = sign2_matrix
            if adanet_params:
                traintest_file = adanet_params.pop(
                    'traintest_file', traintest_file)
        # parameter heuristics
        with h5py.File(sign2_matrix, 'r') as hf:
            mat_shape = hf['x'].shape
        if 'layer_size' not in adanet_params:
            adanet_params['layer_size'] = layer_size_heuristic(*mat_shape)
        if 'batch_size' not in adanet_params:
            adanet_params['batch_size'] = batch_size_heuristic(*mat_shape)
        if 'epoch_per_iteration' not in adanet_params:
            adanet_params['epoch_per_iteration'] = epoch_per_iteration_heuristic(
                *mat_shape)
        ada = AdaNet(model_dir=adanet_path,
                     traintest_file=traintest_file,
                     **adanet_params)
        self.__log.debug('AdaNet training on %s' % traintest_file)
        ada.train_and_evaluate(evaluate=evaluate)
        self.__log.debug('model saved to %s' % adanet_path)
        # when evaluating also save the performances
        if evaluate:
            singles = self.adanet_single_spaces(adanet_path, traintest_file,
                                                suffix)
            # save AdaNet performances and plots
            sign2_plot = Plot(self.dataset, adanet_path)
            ada.save_performances(adanet_path, sign2_plot, suffix, singles)

    def save_sign0_matrix(self, sign0, destination, include_confidence=True,
                          chunk_size=1000):
        """Save matrix of signature 0 and confidence values.

        Args:
            sign0(list): Signature 0 to learn from.
            destination(str): Path where to save the matrix (HDF5 file).
            include_confidence(bool): whether to include confidences.
        """
        self.__log.debug('Saving sign0 traintest to: %s' % destination)
        mask = np.isin(self.keys, sign0.keys, assume_unique=True)
        # the following work only if sign0 keys is a subset (or ==) of sign3
        assert(np.all(np.isin(sign0.keys, self.keys, assume_unique=True)))
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

    def learn_sign0(self, sign0, adanet_params, reuse=True, suffix=None,
                    evaluate=True, include_confidence=True):
        """Learn the signature 3 from sign0.

        This method is used twice. First to evaluate the performances of the
        AdaNet model. Second to train the final model on the full set of data.

        Args:
            sign0(list): Signature 0 object to learn from.
            adanet_params(dict): Dictionary with algorithm parameters.
            reuse(bool): Whether to reuse intermediate files (e.g. the
                aggregated signature 2 matrix).
            suffix(str): A suffix for the AdaNet model path (e.g.
                'sign3/models/adanet_<suffix>').
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
        if 'model_dir' in adanet_params:
            adanet_path = adanet_params.pop('model_dir')
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
            traintest_file = adanet_params.pop(
                'traintest_file', traintest_file)
            if not reuse or not os.path.isfile(traintest_file):
                Traintest.split_h5_blocks(sign0_matrix, traintest_file)
        else:
            traintest_file = sign0_matrix
            traintest_file = adanet_params.pop(
                'traintest_file', traintest_file)
        # initialize adanet and start learning
        ada = AdaNet(model_dir=adanet_path,
                     traintest_file=traintest_file,
                     **adanet_params)
        self.__log.debug('AdaNet training on %s' % traintest_file)
        ada.train_and_evaluate(evaluate=evaluate)
        self.__log.debug('model saved to %s' % adanet_path)
        # when evaluating also save the performances
        if evaluate:
            # save AdaNet performances and plots
            sign2_plot = Plot(self.dataset, adanet_path)
            ada.save_performances(adanet_path, sign2_plot, suffix)

    def save_error_matrix(self, sign3_train_file, predict_fn, destination,
                          max_total=1000000, chunk_size=100):
        """Save matrix of error in sign3 prediction.

        This is the matrix for training the signature 3 error estimator
        based on sign2 presence/absence.

        Args:
            sign3_train_file(list): The original train file with sign2s.
            predict_fn(func): Trained Adanet predict function.
            destination(str): Path where to save the matrix (HDF5 file).
            sampling(int): How much subsampling per molecule.
        """
        self.__log.debug('Saving error traintest to: %s' % destination)
        # get current space shape
        self_len = self.sign2_self.shape[0]
        # we calculate the sampling to fill a maximum total
        if self_len > max_total:
            # lower it can be is 1
            sampling = 1
        else:
            sampling = int(np.floor(max_total / float(self_len)))
        feat_shape = (self_len * sampling, len(self.src_datasets))
        # create a new file with x and y
        with h5py.File(destination, 'w') as hf_out:
            hf_out.create_dataset('x', feat_shape, dtype=np.float32)
            hf_out.create_dataset('y', (feat_shape[0], 1), dtype=np.float32)
            # perform sampling for each input and measure distance
            with h5py.File(sign3_train_file, 'r') as hf_in:
                offset = 0
                for i in tqdm(range(0, self_len, chunk_size)):
                    # source input is repeated sampling times
                    for r in range(sampling):
                        # last chunk has different size
                        csize = chunk_size
                        if i + csize > hf_in['x'].shape[0]:
                            csize = hf_in['x'].shape[0] - i
                        src_chunk = slice(i, i + csize)
                        # destination chunk is shifted by input length
                        dst_chunk = slice(offset + (r * csize),
                                          offset + csize + (r * csize))
                        in_x = hf_in['x'][src_chunk]
                        in_y = hf_in['y'][src_chunk]
                        sub_x, _ = subsample(in_x, in_y, prob_original=0.2)
                        coverage = ~np.isnan(sub_x[:, 0::128])
                        # save coverage as X
                        hf_out['x'][dst_chunk] = coverage
                        # run prediction on subsampled input
                        y_pred = predict_fn({'x': sub_x})['predictions']
                        # save log mean squared error as Y
                        mse = np.mean(((in_y - y_pred)**2), axis=1)
                        log_mse = np.log10(mse)
                        hf_out['y'][dst_chunk] = np.expand_dims(log_mse, 1)
                    offset = dst_chunk.stop
                assert(offset == feat_shape[0])

    def learn_error(self, predict_fn, adanet_params, reuse=True, suffix=None,
                    evaluate=True):
        """Learn the signature 3 prediction error.

        This method is used twice. First to evaluate the performances of the
        model. Second to train the final model on the full set of data.

        Args:
            sign2_list(list): List of signature 2 objects to learn from.
            reuse(bool): Whether to reuse intermediate files (e.g. the
                aggregated signature 2 matrix).
            suffix(str): A suffix for the AdaNet model path (e.g.
                'sign3/models/adanet_<suffix>').
            evaluate(bool): Whether we are performing a train-test split and
                evaluating the performances (N.B. this is required for complete
                confidence scores)
        """
        try:
            from chemicalchecker.tool.adanet import AdaNet
        except ImportError as err:
            raise err
        # get params and set folder
        if suffix:
            adanet_path = os.path.join(self.model_path, 'adanet_%s' % suffix)
        else:
            adanet_path = os.path.join(self.model_path, 'adanet')
        if adanet_params:
            if 'model_dir' in adanet_params:
                adanet_path = adanet_params.pop('model_dir')
        if not reuse or not os.path.isdir(adanet_path):
            os.makedirs(adanet_path)
        # generate input matrix
        sign3_train_file = os.path.join(self.model_path, 'train.h5')
        error_matrix = os.path.join(self.model_path, 'train_error.h5')
        if not reuse or not os.path.isfile(error_matrix):
            self.save_error_matrix(sign3_train_file, predict_fn, error_matrix)
        # if evaluating, perform the train-test split
        if evaluate:
            traintest_file = os.path.join(
                self.model_path, 'traintest_error.h5')
            if adanet_params:
                traintest_file = adanet_params.pop(
                    'traintest_file', traintest_file)
            if not reuse or not os.path.isfile(traintest_file):
                Traintest.split_h5_blocks(error_matrix, traintest_file)
        else:
            traintest_file = error_matrix
            if adanet_params:
                traintest_file = adanet_params.pop(
                    'traintest_file', traintest_file)
        # initialize adanet and start learning
        if adanet_params:
            # parameter heuristics
            ada = AdaNet(model_dir=adanet_path,
                         traintest_file=traintest_file,
                         **adanet_params)
        else:
            ada = AdaNet(model_dir=adanet_path, traintest_file=traintest_file)
        self.__log.debug('AdaNet training on %s' % traintest_file)
        ada.train_and_evaluate(evaluate=evaluate)
        self.__log.debug('model saved to %s' % adanet_path)
        # when evaluating also save the performances
        other_predictors = {
            ('LinearRegression', 'ALL'): LinearRegression(
                n_jobs=adanet_params['cpu']),
            ('RandomForest', 'ALL'): RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=50,
                n_jobs=adanet_params['cpu'])
        }
        if evaluate:
            others = self.train_other(
                other_predictors, adanet_path, traintest_file)
            # save AdaNet performances and plots
            sign2_plot = Plot(self.dataset, adanet_path)
            ada.save_performances(adanet_path, sign2_plot, suffix, others)
        else:
            self.train_other(
                other_predictors, adanet_path, traintest_file, train_only=True)

    def fit_sign0(self, sign0, include_confidence=True):
        """Train an AdaNet model to predict sign3 from sign0.

        This method is fitting a model that uses Morgan fingerprint as features
        to predict signature 3. In future other featurization approaches can be
        tested.

        Args:
            chemchecker(ChemicalChecker): The CC object used to fetch input
                signature 0.
            sign0_traintest(str): Path to the train file.
        """

        # check if performance evaluations need to be done
        s0_code = sign0.dataset
        eval_adanet_path = os.path.join(self.model_path,
                                        'adanet_sign0_%s_eval' % s0_code)
        eval_stats = os.path.join(eval_adanet_path, 'stats_eval.pkl')
        if not os.path.isfile(eval_stats):
            self.learn_sign0(sign0, self.params['sign0'],
                             suffix='sign0_%s_eval' % s0_code,
                             evaluate=True,
                             include_confidence=include_confidence)

        # check if we have the final trained model
        final_adanet_path = os.path.join(self.model_path,
                                         'adanet_sign0_%s_final' % s0_code,
                                         'savedmodel')
        if not os.path.isdir(final_adanet_path):
            self.learn_sign0(sign0, self.params['sign0'],
                             suffix='sign0_%s_final' % s0_code,
                             evaluate=False,
                             include_confidence=include_confidence)

    def get_predict_fn(self, model='adanet_sign0_A1.001_final'):
        try:
            from chemicalchecker.tool.adanet import AdaNet
        except ImportError as err:
            raise err
        modelpath = os.path.join(self.model_path, model, 'savedmodel')
        return AdaNet.predict_fn(modelpath)

    def predict_from_smiles(self, smiles, dest_file, chunk_size=1000,
                            predict_fn=None, accurate_novelty=False,
                            keys=None):
        """Given SMILES generate sign0 and predict sign3.

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
            results.create_dataset('V', (len(smiles), 128), dtype=np.float32)
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
            results.create_dataset("shape", data=(len(smiles), 128))
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
                    preds[np.array(failed)] = np.full((133, ),  np.nan)
                # save chunk to H5
                results['V'][chunk] = preds[:, :128]
                results['stddev_norm'][chunk] = preds[:, 128]
                results['intensity_norm'][chunk] = preds[:, 129]
                results['exp_error_norm'][chunk] = preds[:, 130]
                results['novelty_norm'][chunk] = preds[:, 131]
                results['confidence'][chunk] = preds[:, 132]
                if accurate_novelty:
                    novelty = novelty_model.score_samples(preds[:, :128])
                    abs_novelty = np.abs(np.expand_dims(novelty, 1))
                    results['novelty_norm'][chunk] = nov_qtr.transform(
                        abs_novelty).flatten()
        return pred_s3

    def fit(self, sign2_list, sign2_self, sign2_universe=None,
            sign2_coverage=None, sign0=None,
            model_confidence=True, save_correlations=True,
            predict_novelty=True, update_preds=True, normalize_scores=True,
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
                of the molecule used to train a sign2 independendent sign3
                predictor.
            model_confidence(bool): Whether to model confidence. That is based
                on standard deviation of prediction with dropout.
            save_correlations(bool) Whether to save the correlation (average,
                tertile, max) for the given input dataset (result of the
                evaluation).
            predict_novelty(bool) Whether to predict molecule novelty score.
            update_preds(bool): Whether to write or update the sign3.h5
            normalize_scores(bool): Whether to normalize confidence scores.
            validations(bool): Whether to perform validation.
            chunk_size(int): Chunk size when writing to sign3.h5
        """
        try:
            from chemicalchecker.tool.adanet import AdaNet
        except ImportError as err:
            raise err

        # define dataset that will be used
        self.src_datasets = [sign.dataset for sign in sign2_list]
        self.sign2_self = sign2_self
        self.sign2_list = sign2_list
        self.__log.debug('AdaNet fit %s based on %s', self.dataset,
                         str(self.src_datasets))
        # check if performance evaluations need to be done
        eval_adanet_path = os.path.join(self.model_path, 'adanet_eval')
        eval_stats = os.path.join(eval_adanet_path, 'stats_eval.pkl')
        if not os.path.isfile(eval_stats):
            self.learn_sign2(self.params['adanet'],
                             suffix='eval', evaluate=True)

        # check if we have the final trained model
        final_adanet_path = os.path.join(self.model_path, 'adanet_final')
        if not os.path.isdir(final_adanet_path):
            self.learn_sign2(self.params['adanet'],
                             suffix='final', evaluate=False)

        # part of confidence is the expected error
        if model_confidence:
            # generate prediction, measure error, fit regressor
            eval_err_path = os.path.join(self.model_path, 'adanet_error_eval')
            if not os.path.isdir(eval_err_path):
                predict_fn = self.get_predict_fn('adanet_eval')
                # step1 learn dataset availability to error predictor
                self.learn_error(predict_fn, self.params['error'],
                                 suffix='error_eval', evaluate=True)

            # final error predictor
            final_err_path = os.path.join(
                self.model_path, 'adanet_error_final')
            if not os.path.isdir(final_err_path):
                predict_fn = self.get_predict_fn('adanet_final')
                self.learn_error(predict_fn, self.params['error'],
                                 suffix='error_final', evaluate=False)
            self.__log.debug('Loading model for error prediction')
            rf = pickle.load(
                open(os.path.join(final_err_path, 'RandomForest.pkl')), 'rb')

        # get sorted universe inchikeys
        inchikeys = set()
        for ds, sign in zip(self.src_datasets, sign2_list):
            inchikeys.update(sign.unique_keys)
        inchikeys = sorted(list(inchikeys))
        tot_inks = len(inchikeys)
        self.tot_inks = tot_inks
        tot_ds = len(self.src_datasets)

        # build input matrix if not provided (should be since is shared)
        if sign2_universe is None:
            sign2_universe = os.path.join(self.model_path, 'all_sign2.h5')
        if not os.path.isfile(sign2_universe):
            sign3.save_sign2_universe(sign2_list, sign2_universe)
        if sign2_coverage is None:
            sign2_coverage = os.path.join(self.model_path,
                                          'all_sign2_coverage.h5')
        if not os.path.isfile(sign2_coverage):
            sign3.save_sign2_coverage(sign2_list, sign2_coverage)

        # save universe sign3
        if update_preds:
            predict_fn = self.get_predict_fn('adanet_final')
            with h5py.File(self.data_path, "r+") as results:
                # initialize V and keys datasets
                safe_create(results, 'V', (tot_inks, 128), dtype=np.float32)
                safe_create(results, 'keys',
                            data=np.array(inchikeys,
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
                    nan_pred = predict_fn({'x': nan_feat})['predictions']
                    # read input in chunks
                    for idx in tqdm(range(0, tot_inks, chunk_size)):
                        chunk = slice(idx, idx + chunk_size)
                        feat = features['x_test'][chunk]
                        # predict with final model
                        if not model_confidence:
                            results['V'][chunk] = AdaNet.predict(
                                feat, predict_fn)
                            continue
                        # save confidence natural scores
                        # compute estimated error from coverage
                        coverage = ~np.isnan(feat[:, 0::128])
                        results['exp_error'][chunk] = rf.predict(coverage)
                        # draw prediction with sub-sampling (dropout)
                        pred, samples = AdaNet.predict(feat, predict_fn,
                                                       subsample_x_only,
                                                       consensus=True,
                                                       samples=10)
                        results['V'][chunk] = pred
                        # summarize the predictions as consensus
                        consensus = np.mean(samples, axis=1)
                        results['consensus'][chunk] = consensus
                        # zeros input (no info) as intensity reference
                        centered = consensus - nan_pred
                        # measure the intensity (mean of absolute comps)
                        intensities = np.abs(centered)
                        results['intensity'][chunk] = np.mean(
                            intensities, axis=1).flatten()
                        # summarize the standard deviation of components
                        stddevs = np.std(samples, axis=1)
                        # just save the average stddev over the components
                        results['stddev'][chunk] = np.mean(
                            stddevs, axis=1).flatten()

        # normalize consensus scores sampling distribution of known signatures
        if normalize_scores:
            self.normalize_scores(sign2_coverage)

        # use semi-supervised anomaly detection algorithm to predict novelty
        if predict_novelty:
            self.predict_novelty()

        self.background_distances("cosine")
        if validations:
            self.validate()
        # at the very end we learn how to get from A1 sign0 to sign3 directly
        # in order to enable SMILES to sign3 predictions
        if sign0 is not None:
            self.fit_sign0(sign0)
        self.mark_ready()

    def normalize_scores(self, sign2_coverage, chunk_size=10000):
        """Normalize confidence scores."""
        try:
            from chemicalchecker.tool.adanet import AdaNet
        except ImportError as err:
            raise err

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
        full_coverage = DataSignature(sign2_coverage).get_h5_dataset(
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

    def predict_novelty(self, retrain=False, update_sign3=True):
        """Model novelty score via LocalOutlierFactor (semi-supervised).

        Args:
            retrain(bool): Drop old model and train again. (default: False)
            update_sign3(bool): Write novelty scores in h5. (default: True)

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
        if update_sign3:
            self.__log.debug('Updating novelty scores')
            if model is None:
                model = pickle.load(open(novelty_model, 'rb'))
            # get scores for known molecules and pair with indexes
            s2_idxs = np.argwhere(np.isin(self.keys, s2_inks,
                                          assume_unique=True))
            s2_novelty = model.negative_outlier_factor_
            s2_outlier = [0] * s2_novelty.shape[0]
            assert(s2_idxs.shape[0] == s2_novelty.shape[0])
            # predict scores for other molecules and pair with indexes
            s3_inks = sorted(self.unique_keys - set(s2_inks))
            s3_idxs = np.argwhere(np.isin(self.keys, s3_inks,
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

    def predict(self):
        # TODO decide default prediction mode.
        pass

    def train_other(self, predictors, save_path, traintest_file,
                    train_only=False):
        """Train other predictors for comparison."""

        def train_predict_save(name, model, x_true, y_true, split, save_path):
            result = dict()
            result['time'] = 0.
            # train and save
            if split is None or split == 'train':
                self.__log.info('Training model: %s' % name[0])
                t0 = time()
                model.fit(x_true, y_true)
                model_path = os.path.join(save_path, '%s.pkl' % name[0])
                pickle.dump(model, open(model_path, 'wb'))
                result['time'] = time() - t0
                self.__log.info('Training took: %s' % result['time'])
            # call predict
            self.__log.info("Predicting for: %s", name[0])
            y_pred = model.predict(x_true)
            y_pred = np.expand_dims(y_pred, 1)
            y_true = np.expand_dims(y_true, 1)
            self.__log.info("%s Y: %s", name[0], y_pred.shape)
            if y_pred.shape[0] < 4:
                return
            file_true = os.path.join(
                save_path, "_".join([name[0]] + [str(split), 'true']))
            np.save(file_true, y_true)
            file_pred = os.path.join(
                save_path, "_".join([name[0]] + [str(split), 'pred']))
            np.save(file_pred, y_pred)
            result['true'] = file_true
            result['pred'] = file_pred
            result['coverage'] = 1.0
            return result

        # get results for each split
        results = dict()
        if train_only:
            splits = [None]
        else:
            splits = ['train', 'test', 'validation']
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
        return results

    def adanet_single_spaces(self, adanet_path, traintest_file, suffix):
        """Prediction of adanet using single space signatures.

        We want to compare the performances of trained adanet to those of
        predictors based on single space. This is done filling the matrix
        with NaNs spaces not being evaluated. Also particular combinations
        can be assesses (e.g. excluding all spaces from level A).

        Args:
            adanet_path(str): Path to the AdaNet SavedModel.
            traintest_file(str): Path to the traintest file.
            suffix(str): Suffix string for the predictor name.
        """
        try:
            from chemicalchecker.tool.adanet import AdaNet
        except ImportError as err:
            raise err

        def predict_and_save(name, idxs, save_dir, traintest_file, split,
                             predict_fn, mask_fn, adanet_path, total_size):
            # call predict
            self.__log.info("Predicting for: %s", name)
            y_pred, y_true = AdaNet.predict_online(
                traintest_file, split,
                predict_fn=predict_fn,
                mask_fn=partial(mask_fn, idxs),
                limit=1000)
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
        all_dss = list(self.src_datasets)
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

    def grid_search_adanet(self, job_path, parameters, **kwargs):
        """Perform a grid search.

        parameters = {
            'boosting_iterations': [10, 25, 50],
            'adanet_lambda': [1e-3, 5 * 1e-3, 1e-2],
            'layer_size': [8, 128, 512, 1024]
        }
        """
        from chemicalchecker.util.hpc import HPC
        from sklearn.model_selection import ParameterGrid
        from chemicalchecker.util import Config

        # read config file
        cc_config = kwargs.get("cc_config", os.environ['CC_CONFIG'])
        cfg = Config(cc_config)

        # root grid search directory
        grid_search_path = kwargs.get("grid_search_path", 'adanet_grid_search')
        grid_search_root = os.path.join(self.model_path, grid_search_path)
        if not os.path.isdir(grid_search_root):
            os.makedirs(grid_search_root)

        # cpus
        cpu = kwargs.get("cpu", 1)

        # create job directory if not available
        if not os.path.isdir(job_path):
            os.mkdir(job_path)

        # elements to parallelize on are parameters combinations
        elements = list()
        for params in ParameterGrid(parameters):
            model_dir = '-'.join("%s_%s" % kv for kv in params.items())
            params.update(
                {'model_dir': os.path.join(grid_search_root, model_dir)})
            elements.append((self, params))

        # create script file
        cc_package_path = os.path.join(cfg.PATH.CC_REPO, 'package')
        script_lines = [
            "import sys, os",
            "import pickle",
            "sys.path.append('%s')" % cc_package_path,
            "from chemicalchecker.core import ChemicalChecker",
            "cc = ChemicalChecker()",
            "task_id = sys.argv[1]",  # <TASK_ID>
            "filename = sys.argv[2]",  # <FILE>
            "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
            "data = inputs[task_id]",  # elements for current job
            "for s3, params in data:",  # elements are indexes
            "    params['cpu'] = %s" % cpu,
            "    s3.learn_sign2(params)",
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'sign3_grid_search_adanet.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        params = {}
        params["num_jobs"] = len(elements)
        params["jobdir"] = job_path
        params["job_name"] = "CC_SIGN3_GRID_SEARCH_ADANET"
        params["elements"] = elements
        params["wait"] = False
        params["memory"] = 32
        # job command
        singularity_image = cfg.PATH.SINGULARITY_IMAGE
        command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={}" +\
            " singularity exec {} python {} <TASK_ID> <FILE>"
        command = command.format(
            os.path.join(cfg.PATH.CC_REPO, 'package'), cc_config,
            singularity_image, script_name)
        # submit jobs
        cluster = HPC.from_config(Config())
        cluster.submitMultiJob(command, **params)
        return cluster


def safe_create(h5file, *args, **kwargs):
    if args[0] not in h5file:
        h5file.create_dataset(*args, **kwargs)


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


def col_wise_correlation(X, Y):
    return row_wise_correlation(X.T, Y.T)


def row_wise_correlation(X, Y):
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


def subsample_x_only(tensor, label=None, prob_original=0.05):
    return subsample(tensor, label=None, prob_original=prob_original)[0]


def subsample(tensor, label, prob_original=0.05):
    """Function to subsample stacked data."""
    # it is safe to make a local copy of the input matrix
    new_data = np.copy(tensor)
    # we will have a masking matrix at the end
    mask = np.zeros_like(new_data).astype(bool)
    for idx, row in enumerate(new_data):
        # the following assume the stacked signature to have a fixed width
        presence = ~np.isnan(row[0::128])
        # low probability of keeping the original sample
        if np.random.rand() < prob_original:
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


def subsample_coverage(tensor, prob_original=0.05):
    """Function to subsample stacked data."""
    # it is safe to make a local copy of the input matrix
    new_data = np.copy(tensor)
    # we will have a masking matrix at the end
    mask = np.zeros_like(new_data).astype(bool)
    for idx, presence in enumerate(new_data):
        # low probability of keeping the original sample
        if np.random.rand() < prob_original:
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
        mask[idx] = presence_add
    return mask
