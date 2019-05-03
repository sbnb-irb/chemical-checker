"""Signature type 3.

Network embedding of observed *and* inferred similarity networks. Their added
value, compared to signatures type 2, is that they can be derived for
virtually *any* molecule in *any* dataset.
"""
import os
import h5py
import numpy as np
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
        self.params['graph'] = params.get('graph', None)
        self.params['node2vec'] = params.get('node2vec', None)
        default_adanet = {
            "augmentation": subsample,
            "initial_architecture": [9, 1]
        }
        self.params['adanet'] = params.get('adanet', default_adanet)

    def get_sign2_matrix(self, chemchecker, include_self=True):
        """Get combined matrix of stacked signature 2."""
        # get current space on which we'll train (using reference set to
        # limit redundancy)
        my_sign2 = chemchecker.get_signature(
            'sign2', 'full_map', self.dataset)
        # get other space signature 2 for molecule in current space (allow nan)
        other_spaces = list(chemchecker.datasets)
        if not include_self:
            other_spaces.remove(self.dataset)
        ref_dimension = my_sign2.shape[1]
        sign2_matrix = np.zeros(
            (my_sign2.shape[0], ref_dimension * len(other_spaces)),
            dtype=np.float32)
        for idx, ds in enumerate(other_spaces):
            sign2_ds = chemchecker.get_signature('sign2', 'full_map', ds)
            _, signs = sign2_ds.get_vectors(my_sign2.keys, include_nan=True)
            start_idx = ref_dimension * idx
            end_idx = ref_dimension * (idx + 1)
            sign2_matrix[:, start_idx:end_idx] = signs
            available = np.isin(my_sign2.keys, sign2_ds.keys)
            self.__log.info('%s shared molecules between %s and %s',
                            sum(available), self.dataset, ds)
        return sign2_matrix, my_sign2[:]

    def fit(self, chemchecker, reuse=True, suffix=None, evaluate=True):
        """Learn a model."""
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
        if evaluate:
            traintest_file = os.path.join(self.model_path, 'traintest.h5')
            if adanet_params:
                traintest_file = adanet_params.pop(
                    'traintest_file', traintest_file)
            if not reuse or not os.path.isfile(traintest_file):
                features, labels = self.get_sign2_matrix(chemchecker)
                Traintest.create(features, labels, traintest_file)
        else:
            traintest_file = os.path.join(self.model_path, 'train.h5')
            if adanet_params:
                traintest_file = adanet_params.pop(
                    'traintest_file', traintest_file)
            if not reuse or not os.path.isfile(traintest_file):
                features, labels = self.get_sign2_matrix(chemchecker)
                Traintest.create(features, labels, traintest_file,
                                 split_fractions=[1.0, .0, .0])
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
            singles = self.adanet_single_spaces(chemchecker, adanet_path,
                                                traintest_file, suffix)
            # save AdaNet performances and plots
            sign2_plot = Plot(self.dataset, adanet_path, self.validation_path)
            ada.save_performances(adanet_path, sign2_plot, suffix, singles)
        self.mark_ready()

    def predict(self, chemchecker, all_sign2, inchikeys=None, confidence=True):
        """Use the learned model to predict the signature."""
        try:
            from chemicalchecker.tool.adanet import AdaNet
        except ImportError as err:
            raise err

        # get signatures
        ds_sign = dict()
        for ds in chemchecker.datasets:
            ds_sign[ds] = chemchecker.get_signature('sign2',
                                                    'full_map', ds)
        # get universe molecules
        if not inchikeys:
            keys = list()
            for ds in chemchecker.datasets:
                keys.append(ds_sign[ds].unique_keys)
            inchikeys = sorted(set.union(*keys))
        # build input matrix if not provided
        if not os.path.isfile(all_sign2):
            with h5py.File(all_sign2, "w") as fh:
                fh.create_dataset('x_test',
                                  (len(inchikeys), 128 * len(ds_sign)),
                                  dtype=np.float32)
            with h5py.File(all_sign2, "a") as fh:
                for idx, ds in enumerate(chemchecker.datasets):
                    _, sign = ds_sign[ds].get_vectors(
                        inchikeys, include_nan=True)
                    fh['x_test'][:, idx * 128:(idx + 1) * 128] = sign
                    del sign

        # read the final model
        adanet_path = os.path.join(self.model_path, 'adanet_final',
                                   'savedmodel')
        predict_fn = AdaNet.predict_fn(adanet_path)
        # perform prediction
        with h5py.File(self.data_path, "w") as results:
            results.create_dataset(
                'V', (len(inchikeys), 128), dtype=np.float32)
            results.create_dataset('keys', (len(inchikeys),), dtype='|S27')
            results.create_dataset('uncertanty', (len(inchikeys),), dtype=np.float32)
            with h5py.File(all_sign2, "r") as features:
                for idx in tqdm(range(0, len(inchikeys), 10000)):
                    chunk = slice(idx, idx + 10000)
                    results['V'][chunk] = AdaNet.predict(adanet_path,
                                                         features['x_test'][
                                                             chunk],
                                                         predict_fn)
                    results['keys'][chunk] = inchikeys[chunk]
                    stddevs = AdaNet.predict(adanet_path, 
                                             features['x_test'][chunk],
                                             predict_fn, subsample_x_only,
                                             True)
                    results['uncertanty'][chunk] = np.std(stddevs, axis=1)

    @staticmethod
    def cross_fit(chemchecker, model_path, ds_from, ds_to, reuse=True):
        """Learn a predictor between sign_from and sign_to.

        Signatures should be `full` to maximize the intersaction, that's the
        training input.
        """
        try:
            from chemicalchecker.tool.adanet import AdaNet, Traintest
            from .sign2 import sign2
        except ImportError as err:
            raise err
        sign3.__log.debug('AdaNet cross fit signatures')
        # get params and set folder
        cross_pred_path = os.path.join(model_path, 'crosspred_%s' % ds_from)
        if not reuse or not os.path.isdir(cross_pred_path):
            os.makedirs(cross_pred_path)
        # create traintest file (simple intersection)
        ds_traintest_file = os.path.join(cross_pred_path, 'traintest.h5')
        sign_from = chemchecker.get_signature('sign2', 'full_map', ds_from)
        sign_to = chemchecker.get_signature('sign2', 'full_map', ds_to)
        if not reuse or not os.path.isfile(ds_traintest_file):
            _, X, Y = sign_from.get_intersection(sign_to)
            Traintest.create(X, Y, ds_traintest_file)
        # adanet
        ada_single_space = AdaNet(model_dir=cross_pred_path,
                                  traintest_file=ds_traintest_file)
        ada_single_space.train_and_evaluate()
        # add nearest neighbor predictor
        nearest_neighbor_pred = sign2.predict_nearest_neighbor(
            cross_pred_path, ds_traintest_file)
        extra_predictors = dict()
        extra_predictors['NearestNeighbor'] = nearest_neighbor_pred
        sign2_plot = Plot(ds_to, cross_pred_path, cross_pred_path)
        ada_single_space.save_performances(
            cross_pred_path, sign2_plot, extra_predictors)

    def average_cross_fit(self, chemchecker):
        """Prediction averaging single cross-predictors.

        We want to compare the performances of single cross-predictors (e.g.
        X: A1, y: E5) to the horizontally-stacked adanet prediction (e.g.
        X:A1-A2-...-E4, Y: E5).
        This is not totally fair as single preditors are trained on different
        train-test split so train molecules of single cross-predictors might be
        part of the test set used here giving them an advantage.
        """
        try:
            from chemicalchecker.tool.adanet import AdaNet, Traintest
            from .sign2 import sign2
            from .neig1 import neig1
            from scipy.stats import pearsonr
            from sklearn.metrics import r2_score, mean_squared_error
            from sklearn.metrics import explained_variance_score
            import pandas as pd
            import pickle
            from sklearn.linear_model import LinearRegression
        except ImportError as err:
            raise err

        def _stats_row(y_true, y_pred, algo, dataset):
            row = dict()
            row['algo'] = algo
            row['dataset'] = dataset
            row['r2'] = r2_score(y_true, y_pred)
            pps = [pearsonr(y_true[:, x], y_pred[:, x])[0]
                   for x in range(y_true.shape[1])]
            row['pearson_avg'] = np.mean(pps)
            row['pearson_std'] = np.std(pps)
            row['mse'] = mean_squared_error(y_true, y_pred)
            row['explained_variance'] = explained_variance_score(
                y_true, y_pred)
            return row

        df = pd.DataFrame(columns=[
            'dataset', 'r2', 'pearson_avg', 'pearson_std', 'algo', 'mse',
            'explained_variance', 'time', 'architecture', 'nr_variables',
            'nn_layers', 'layer_size', 'architecture_history'])

        # train-test split from horizontal-stack dataset
        traintest_file = os.path.join(self.model_path, 'traintest.h5')
        linreg = dict()
        # for each dataset partition
        for part in ['train', 'test', 'validation']:
            # copy matrix from traintest_file
            traintest = Traintest(traintest_file, part)
            traintest.open()
            x_data = traintest.get_all_x()
            results = dict()
            results['AdaNet'] = traintest.get_all_x()
            results['NearestNeighbor'] = traintest.get_all_x()
            results['LinearRegression'] = traintest.get_all_x()
            y_data = traintest.get_all_y()
            traintest.close()
            # iterate on spaces
            other_spaces = list(chemchecker.datasets)
            other_spaces.remove(self.dataset)

            # for each of the spaces generate individual prediction
            # with all methods (each with its own prediction matrix)
            for idx, ds in enumerate(other_spaces):
                cross_pred_path = os.path.join(
                    self.model_path, 'crosspred_%s' % ds)
                ds_traintest_file = os.path.join(
                    cross_pred_path, 'traintest.h5')
                # train (on ds_traintest_file) or get algo
                # copy traintest_file columns
                col_slice = slice(idx * 128, (idx + 1) * 128)
                ds_x_data = x_data[:, col_slice]
                # not nan idxs
                notnan_idxs = ~np.isnan(ds_x_data).any(axis=1)
                # predict with adanet and append to x matrix
                save_dir = os.path.join(cross_pred_path, 'savedmodel')
                ds_pred = AdaNet.predict(save_dir, ds_x_data[notnan_idxs])
                results['AdaNet'][notnan_idxs, col_slice] = ds_pred
                # nn predictions
                nn_path = os.path.join(cross_pred_path, "nearest_neighbor")
                sign2_dest = os.path.join(nn_path, "sign2")
                nn_sign2 = sign2(sign2_dest, sign2_dest, ds)
                neig1_dest = os.path.join(nn_path, "neig1")
                nn_neig1 = neig1(neig1_dest, neig1_dest, "NN.001")
                nn_idxs = nn_neig1.get_kth_nearest(ds_x_data[notnan_idxs], 1)
                tmp = list()
                for idx in nn_idxs:
                    tmp.append(nn_sign2[idx])
                results['NearestNeighbor'][
                    notnan_idxs, col_slice] = np.vstack(tmp)
                # linear regression
                if part == 'train':
                    tt = Traintest(ds_traintest_file, part)
                    tt.open()
                    linreg[ds] = LinearRegression().fit(
                        tt.get_all_x(), tt.get_all_y())
                    tt.close()
                results['LinearRegression'][notnan_idxs, col_slice] = \
                    linreg[ds].predict(ds_x_data[notnan_idxs])

            # the 'average' prediction is done per molecule reshaping the
            # full prediction matrix and getting the average along first axis
            for name, res in results.items():
                y_pred = np.zeros_like(y_data)
                for idx, row in enumerate(res):
                    mol = row.reshape((24, 128))
                    notnan_idxs = ~np.isnan(mol).any(axis=1)
                    y_pred[idx] = np.average(mol[notnan_idxs], axis=0)
                # save performances
                rows = _stats_row(y_data, y_pred, name, part)
                df.loc[len(df)] = pd.Series(rows)
                output_dir = os.path.join(self.model_path, 'crosspred_AVG')
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                output_pkl = os.path.join(output_dir, 'stats.pkl')
                with open(output_pkl, 'wb') as fh:
                    pickle.dump(df, fh)
                output_csv = os.path.join(output_dir, 'stats.csv')
                df.to_csv(output_csv)

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
                save_dir, traintest_file, split,
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
        all_dss = list(chemchecker.datasets)
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

    @staticmethod
    def test_params(cc_root, job_path, dataset, parameters, cpu=1):
        """Perform a grid search.

        parameters = {
            'augmentation': [False, subsample],
            'epoch_per_iteration': [1, 5, 25, 100],
            'adanet_iterations': [1, 3],
            'initial_architecture': [[], [9, 1], [4, 3, 2, 1]]
        }
        """
        from chemicalchecker.util.hpc import HPC
        from chemicalchecker.util import Config
        from sklearn.model_selection import ParameterGrid

        # create job directory if not available
        if not os.path.isdir(job_path):
            os.mkdir(job_path)
        # create script file
        cc_config = os.environ['CC_CONFIG']
        cfg = Config(cc_config)
        cc_package = os.path.join(cfg.PATH.CC_REPO, 'package')
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
            "for params in data:",  # elements are indexes
            "    ds = '%s'" % dataset,
            "    s3 = cc.get_signature('sign3', 'full_map', ds, adanet=params['init'])",
            "    s3.fit(cc, **params['fit'])",
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'sign3_test_params.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        elements = list()
        for params in ParameterGrid(parameters):
            str_params = list()
            for k, v in params.items():
                str_pair = "%s_%s" % (k, v)
                if hasattr(v, 'func_name'):
                    str_pair = "%s_%s" % (k, v.func_name)
                if type(v) == list:
                    str_pair = "%s_%s" % (k, ','.join([str(x) for x in v]))
                str_params.append(str_pair)
            suffix = '-'.join(str_params)
            elements.append({'init': params, 'fit': {'suffix': suffix}})

        params["num_jobs"] = len(elements)
        params["jobdir"] = job_path
        params["job_name"] = "CC_SIGN3_TEST_PARAMS"
        params["elements"] = elements
        params["wait"] = False
        params["memory"] = 16
        params["cpu"] = cpu
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC(Config())
        cluster.submitMultiJob(command, **params)
        return cluster
    """
    @staticmethod
    def fit_hpc(cc_root, job_path, elements, evaluate=True, suffix='final', cpu=1):
        from chemicalchecker.util.hpc import HPC
        from chemicalchecker.util import Config

        # create job directory if not available
        if not os.path.isdir(job_path):
            os.mkdir(job_path)
        if evaluate:
            suffix = '%s_eval' % suffix
        # create script file
        cc_config = os.environ['CC_CONFIG']
        cfg = Config(cc_config)
        cc_package = os.path.join(cfg.PATH.CC_REPO, 'package')
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
            "    s3 = cc.get_signature('sign3', 'full_map', ds)",
            "    s3.fit(cc, evaluate=%s, suffix='%s')" % (evaluate, suffix),
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'sign3_fit.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        params = dict()
        params["num_jobs"] = len(elements)
        params["jobdir"] = job_path
        params["job_name"] = "CC_SIGN3"
        params["elements"] = elements
        params["wait"] = False
        params["memory"] = 16
        params["cpu"] = cpu
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC(Config())
        cluster.submitMultiJob(command, **params)
        return cluster
    """

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

"""
import tensorflow as tf


def subsample_tf(tensor, label):
    #Function to subsample stacked data.
    # it is safe to make a local copy of the input matrix
    new_data = tf.identity(tensor)
    # we will have a masking matrix at the end
    mask = tf.zeros_like(new_data)
    mask = tf.cast(mask, tf.bool)
    # presence matrix (assume the stacked signature to have a fixed width)
    presence = tf.cast(~tf.is_nan(new_data[:, 0::128]), tf.int32)
    # how many dataset do we have for each row?
    max_ds = tf.cast(tf.math.reduce_sum(presence, axis=1), tf.int32)
    # how many dataset do we want to consider for each row?
    nr_ds = tf.random_uniform(max_ds.shape, minval=1,
                              maxval=max_ds + 1, dtype=tf.int32)
    # nr_ds_i goes from 1 to maximum dataset for each row
    # for each line remove 1s until we have same as nr_ds_i
    joined = tf.concat((presence, nr_ds), axis=1)

    def subsample_presence(joined):
        cond = lambda i: tf.equal(tf.math.reduce_sum(i[-1:]), i[-1])
        body = lambda i: tf.add(i, 1)
        return arg[0] + arg[1]
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

    # low probability of keeping the original sample
    if np.random.rand() > 0.95:
        presence_add = presence
    else:
        # make masked dataset NaN
    new_data[~mask] = np.nan
    return new_data, label
"""
