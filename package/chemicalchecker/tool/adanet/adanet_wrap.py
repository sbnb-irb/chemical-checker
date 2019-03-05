import os
import h5py
import shutil
import pickle
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
try:
    import adanet
    import tensorflow as tf
    import tensorflow.contrib.slim as slim
    from tensorflow.contrib import predictor
except ImportError:
    print("ADANET module not imported!")

from .dnn_stack_generator import StackDNNGenerator
from .dnn_extend_generator import ExtendDNNGenerator
from chemicalchecker.util import logged, profile


@logged
class Traintest(object):
    """Convenience batch reader from HDF5 files.

    This class allow creation and access to HDF5 train-test sets and expose
    the generator functions which tensorflow likes.
    """

    def __init__(self, hdf5_file, partition, nan_replacer=0.0):
        """Initialize the traintest object.

        We assume the file is containing diffrent partitions.
        e.g. "x_train", "y_train", "x_test", ...
        """
        self._file = hdf5_file
        self._f = None
        self.x_name = "x_%s" % partition
        self.y_name = "y_%s" % partition
        self.nan_replacer = nan_replacer

    def open(self):
        """Open the HDF5."""
        self._f = h5py.File(self._file, 'r')
        self.__log.info("HDF5 open %s", self._file)

    def close(self):
        """Close the HDF5."""
        try:
            self._f.close()
            self.__log.info("HDF5 close %s", self._file)
        except AttributeError:
            self.__log.error('HDF5 file is not open yet.')

    def get_xy(self, beg_idx, end_idx):
        """Get the batch."""
        features = self._f[self.x_name][beg_idx: end_idx]
        labels = self._f[self.y_name][beg_idx: end_idx]
        # handle NaNs
        features[np.where(np.isnan(features))] = self.nan_replacer
        return features, labels

    def get_x(self, beg_idx, end_idx):
        """Get the Xs in a range."""
        features = self._f[self.x_name][beg_idx: end_idx]
        # handle NaNs
        features[np.where(np.isnan(features))] = self.nan_replacer
        return features

    def get_all_x(self):
        """Get all the Xs."""
        features = self._f[self.x_name][:]
        # handle NaNs
        features[np.where(np.isnan(features))] = self.nan_replacer
        return features

    def get_all_y(self):
        """Get all the Ys."""
        labels = self._f[self.y_name][:]
        return labels

    @staticmethod
    def create_signature_file(sign_from, sign_to, out_filename):
        """Create the HDF5 file with both X and Y, train and test."""
        # get type1
        with h5py.File(sign_from, 'r') as fh:
            X = fh['V'][:]
            check_X = fh['keys'][:]
        X = np.asarray(X, dtype=np.float32)
        # get type2
        with h5py.File(sign_to, 'r') as fh:
            Y = fh['V'][:]
            check_Y = fh['keys'][:]
        assert(np.array_equal(check_X, check_Y))
        # train test validation splits
        Traintest.create(X, Y, out_filename)

    @staticmethod
    def create(X, Y, out_filename, augment=None):
        """Create the HDF5 file with both X and Y, train and test."""
        Traintest.__log.debug(
            "{:<20} shape: {:>10}".format("input X", X.shape))
        Traintest.__log.debug(
            "{:<20} shape: {:>10}".format("input Y", Y.shape))
        # train test validation splits
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42)

        if augment:
            if 'strategy' not in augment:
                raise Exception("Please specify a data augmentation strategy" +
                                "e.g.'noise','interpolation','probabilities'")
            if 'max_size' not in augment:
                augment['max_size'] = 1e7
            Traintest.__log.warn(
                "Augmenting data with %s strategy.", augment['strategy'])
            if augment['strategy'] == 'noise':
                # initial list
                p_x = list([x_train])
                p_y = list([y_train])
                # perturb train vectors
                for i in range(augment['max_size'] - len(x_train)):
                    perturbation = np.random.normal() * 1e-4
                    p_x.append(x_train + perturbation)
                    p_y.append(y_train + perturbation)
                # join list into numpy array
                x_train = np.concatenate(tuple(p_x))
                y_train = np.concatenate(tuple(p_y))
            elif augment['strategy'] == 'interpolation':
                # initial list
                p_x = list([x_train])
                p_y = list([y_train])
                # interpolate point to get new points
                mask = range(len(x_train))
                mask1 = range(len(x_train))
                for i in range(augment['max_size'] - len(x_train)):
                    np.random.shuffle(mask)
                    np.random.shuffle(mask1)
                    p_x.append((x_train + x_train[mask] + x_train[mask1]) / 3.)
                    p_y.append((y_train + y_train[mask] + y_train[mask1]) / 3.)
                # join list into numpy array
                x_train = np.concatenate(tuple(p_x))
                y_train = np.concatenate(tuple(p_y))
            elif augment['strategy'] == "probabilities" \
                    and x_train.shape[0] < augment['max_size']:
                # initial list
                p_x = list()
                p_y = list()
                if 'probabilities' not in augment:
                    raise Exception("Please specify probabilities.")
                # probabilities
                p_space, p_count = augment['probabilities']
                # repeat the subsampling until enough subsamples are drawn
                pbar = tqdm(total=augment['max_size'] - x_train.shape[0])
                while len(p_x) < augment['max_size'] - x_train.shape[0]:
                    # subsample vector
                    mol_idxs = range(len(x_train))
                    np.random.shuffle(mol_idxs)
                    for idx in mol_idxs:
                        # early termination criteria, when I have enought
                        if len(p_x) >= augment['max_size'] - x_train.shape[0]:
                            break
                        # get the presence array e.g. 110
                        presence = ~np.isnan(x_train[idx][0::128])
                        # debug_str = "* {:<3} ".format(idx) + \
                        #    ''.join(presence.astype(int).astype(str))
                        # Traintest.__log.debug(debug_str)
                        present_idxs = np.argwhere(presence).flatten()
                        # iterate on  individual starting point e.g. 100, 010
                        for initial in present_idxs:
                            presence_add = np.zeros(
                                presence.shape).astype(bool)
                            presence_add[initial] = True
                            # what's the max spaces I can had?
                            max_add = present_idxs.shape[0] - 1
                            my_p_count = {k: v for k,
                                          v in p_count.items() if k < max_add}
                            # normalize probabilities
                            prob_sum = sum(my_p_count.values())
                            my_p_count = [(k, v / prob_sum) for k,
                                          v in my_p_count.items()]
                            # check that we have options to subsample
                            if not my_p_count:
                                # Traintest.__log.debug("no subsampling.")
                                break
                            # pick how many to add
                            ns = [x[0] for x in my_p_count]
                            p_n = [x[1] for x in my_p_count]
                            to_add = np.random.choice(ns, 1, p_n)[0]
                            # pick which spaces
                            pick = list(present_idxs)
                            pick.remove(initial)
                            for i in range(to_add):
                                # normalize probabilities
                                p_s = list(p_space[pick] / sum(p_space[pick]))
                                added = np.random.choice(pick, 1, p_s)[0]
                                pick.remove(added)
                                presence_add[added] = True
                            # debug_str = "{:<3}".format(initial) + \
                            #    "{:<3}".format(to_add) + \
                            #    ''.join(presence_add.astype(int).astype(str))
                            # Traintest.__log.debug(debug_str)
                            # assert(sum(presence_add) == to_add + 1)
                            # check that generated array is a subset of
                            # starting
                            # assert(np.array_equal(
                            #    np.logical_or(presence, presence_add),
                            #    presence))
                            # convert array to mask
                            mask = np.hstack([[a] * 128 for a in presence_add])
                            # append new row
                            new_data = np.copy(x_train[idx])
                            new_data[~mask] = np.nan
                            p_x.append(new_data)
                            p_y.append(np.copy(y_train[idx]))
                            pbar.update(1)
                pbar.close()
                # join list into numpy array
                x_train = np.vstack((x_train, np.vstack(p_x)))
                y_train = np.vstack((y_train, np.vstack(p_y)))

        # create dataset
        with h5py.File(out_filename, "w") as fh:
            fh.create_dataset('x_train', data=x_train, dtype=np.float32)
            fh.create_dataset('y_train', data=y_train, dtype=np.float32)
            fh.create_dataset('x_test', data=x_test, dtype=np.float32)
            fh.create_dataset('y_test', data=y_test, dtype=np.float32)
            fh.create_dataset('x_validation', data=x_val, dtype=np.float32)
            fh.create_dataset('y_validation', data=y_val, dtype=np.float32)
            Traintest.__log.debug("{:<20} shape: {:>10}".format(
                "x_train", fh["x_train"].shape))
            Traintest.__log.debug("{:<20} shape: {:>10}".format(
                "y_train", fh["y_train"].shape))
            Traintest.__log.debug("{:<20} shape: {:>10}".format(
                "x_test", fh["x_test"].shape))
            Traintest.__log.debug("{:<20} shape: {:>10}".format(
                "y_test", fh["y_test"].shape))
            Traintest.__log.debug("{:<20} shape: {:>10}".format(
                "x_validation", fh["x_validation"].shape))
            Traintest.__log.debug("{:<20} shape: {:>10}".format(
                "y_validation", fh["y_validation"].shape))
        fh.close()
        Traintest.__log.info('Traintest saved to %s', out_filename)

    @staticmethod
    def generator_fn(file_name, partition, batch_size=None, only_x=False):
        """Return the generator function that we can query for batches."""
        reader = Traintest(file_name, partition)
        reader.open()
        x_shape = reader._f[reader.x_name].shape
        y_shape = reader._f[reader.y_name].shape
        if not batch_size:
            batch_size = x_shape[0]

        def example_generator_fn():
            epoch = 0
            beg_idx, end_idx = 0, batch_size
            total = reader._f[reader.x_name].shape[0]
            while True:
                if beg_idx >= total:
                    Traintest.__log.debug("EPOCH completed")
                    beg_idx = 0
                    epoch += 1
                    return
                if only_x:
                    yield reader.get_x(beg_idx, end_idx)
                else:
                    yield reader.get_xy(beg_idx, end_idx)
                beg_idx, end_idx = beg_idx + batch_size, end_idx + batch_size

        return x_shape, y_shape, example_generator_fn


@logged
class AdaNetWrapper(object):
    """Wrapper class adapted from scripted examples on AdaNet's github.

    https://github.com/tensorflow/adanet/blob/master/adanet/
    examples/tutorials/adanet_objective.ipynb
    """

    def __init__(self, traintest_file, **kwargs):
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.batch_size = int(kwargs.get("batch_size", 32))
        self.learn_mixture_weights = kwargs.get("learn_mixture_weights", True)
        self.adanet_lambda = kwargs.get("adanet_lambda", 0.001)
        self.random_seed = int(kwargs.get("random_seed", 42))
        self.model_dir = kwargs.get("model_dir", None)
        self.activation = kwargs.get("activation", tf.nn.relu)
        self.shuffles = int(kwargs.get("shuffles", 10))
        self.dropout_rate = float(kwargs.get("dropout_rate", 0.2))
        self.adanet_iterations = int(kwargs.get("adanet_iterations", 10))
        self.subnetwork_generator = eval(kwargs.get(
            "subnetwork_generator", "ExtendDNNGenerator"))
        # read input shape
        self.traintest_file = traintest_file
        with h5py.File(traintest_file, 'r') as hf:
            self.input_dimension = hf['x_train'].shape[1]
            self.label_dimension = hf['y_train'].shape[1]
            self.train_size = hf['x_train'].shape[0]
            self.total_size = hf['x_train'].shape[
                0] + hf['x_test'].shape[0] + hf['x_validation'].shape[0]
        # layer size heuristic
        heu_layer_size = AdaNetWrapper.layer_size_heuristic(
            self.total_size, self.input_dimension, self.label_dimension)
        self.layer_size = int(kwargs.get("layer_size", heu_layer_size))
        # make adanet iteration proportional to input size (with lower bound)
        # we want to guarantee one epoch per adanet iteration
        self.train_step = int(self.train_size // self.batch_size) + 1
        self.train_step = max(self.train_step, 1000)
        self.total_steps = self.train_step * self.adanet_iterations
        self.results = None
        self.estimator = None
        # log parameters
        self.__log.info("**** AdaNet Parameters: ***")
        self.__log.info("{:<22}: {:>12}".format(
            "train_size", self.train_size))
        self.__log.info("{:<22}: {:>12}".format(
            "input_dimension", self.input_dimension))
        self.__log.info("{:<22}: {:>12}".format(
            "label_dimension", self.label_dimension))
        self.__log.info("{:<22}: {:>12}".format("model_dir", self.model_dir))
        self.__log.info("{:<22}: {:>12}".format(
            "traintest_file", self.traintest_file))
        self.__log.info("{:<22}: {:>12}".format(
            "learning_rate", self.learning_rate))
        self.__log.info("{:<22}: {:>12}".format("batch_size", self.batch_size))
        self.__log.info("{:<22}: {:>12}".format(
            "learn_mixture_weights", self.learn_mixture_weights))
        self.__log.info("{:<22}: {:>12}".format(
            "adanet_lambda", self.adanet_lambda))
        self.__log.info("{:<22}: {:>12}".format(
            "adanet_iterations", self.adanet_iterations))
        self.__log.info("{:<22}: {:>12}".format(
            "random_seed", self.random_seed))
        self.__log.info("{:<22}: {:>12}".format("activation", self.activation))
        self.__log.info("{:<22}: {:>12}".format("layer_size", self.layer_size))
        self.__log.info("{:<22}: {:>12}".format("shuffles", self.shuffles))
        self.__log.info("{:<22}: {:>12}".format(
            "dropout_rate", self.dropout_rate))
        self.__log.info("{:<22}: {:>12}".format(
            "subnetwork_generator", self.subnetwork_generator))
        self.__log.info("{:<22}: {:>12}".format(
            "train_step", self.train_step))
        self.__log.info("{:<22}: {:>12}".format(
            "total_steps", self.total_steps))

    @staticmethod
    def layer_size_heuristic(nr_samples, nr_features, nr_out=128, s_fact=7.):
        heu_layer_size = (
            1 / s_fact) * (np.sqrt(nr_samples) / .3 + ((nr_features + nr_out) / 5.))
        heu_layer_size = np.power(2, np.ceil(np.log2(heu_layer_size)))
        heu_layer_size = np.maximum(heu_layer_size, 32)
        return heu_layer_size

    def train_and_evaluate(self):
        """Train and evaluate AdaNet."""

        """Define the `adanet.Estimator`."""
        self.estimator = adanet.Estimator(
            # We have a multiple-output regression problem,
            # so we'll use a regression head that optimizes for MSE.
            head=tf.contrib.estimator.regression_head(
                label_dimension=self.label_dimension,
                loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE),

            # Define the generator, which defines our search space of
            # subnetworks to train as candidates to add to the final AdaNet
            # model.
            subnetwork_generator=self.subnetwork_generator(
                optimizer=tf.train.RMSPropOptimizer(
                    learning_rate=self.learning_rate),
                learn_mixture_weights=self.learn_mixture_weights,
                layer_size=self.layer_size,
                dropout=self.dropout_rate,
                activation=self.activation,
                seed=self.random_seed),

            # Lambda is a the strength of complexity regularization. A larger
            # value will penalize more complex subnetworks.
            adanet_lambda=self.adanet_lambda,

            # The number of train steps per iteration.
            max_iteration_steps=self.train_step,

            # The evaluator will evaluate the model on the full training set to
            # compute the overall AdaNet loss (train loss + complexity
            # regularization) to select the best candidate to include in the
            # final AdaNet model.
            evaluator=adanet.Evaluator(
                input_fn=self.input_fn("train", training=False)),

            # Configuration for Estimators.
            config=tf.estimator.RunConfig(
                save_checkpoints_secs=1800,  # save checkpoints every half-our
                save_summary_steps=50000,
                tf_random_seed=self.random_seed,
                model_dir=self.model_dir),
            model_dir=self.model_dir
        )
        # Train and evaluate using using the tf.estimator tooling.
        train_spec = tf.estimator.TrainSpec(
            input_fn=self.input_fn("train", training=True),
            max_steps=self.total_steps)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=self.input_fn("test", training=False),
            steps=None,
            start_delay_secs=1,
            throttle_secs=1)
        # call train and evaluate collecting time stats
        t0 = time()
        self.results = tf.estimator.train_and_evaluate(
            self.estimator, train_spec, eval_spec)
        self.time = time() - t0
        # save persistent model
        self.save_dir = os.path.join(self.model_dir, 'savedmodel')
        self.__log.info("SAVING MODEL TO: %s", self.save_dir)
        tmp_dir = self.save_model(self.model_dir)
        if os.path.isdir(self.save_dir):
            shutil.rmtree(self.save_dir)
        shutil.move(tmp_dir, self.save_dir)
        # print final architechture
        AdaNetWrapper.print_model_architechture(self.save_dir)
        return self.estimator, self.results

    def architecture(self):
        """Extract the ensemble architecture from evaluation results."""
        if not self.results:
            return None
        try:
            architecture = self.results[0]["architecture/adanet/ensembles"]
            # The architecture is a serialized Summary proto for TensorBoard.
            summary_proto = tf.summary.Summary.FromString(architecture)
            return summary_proto.value[0].tensor.string_val[0]
        except Exception:
            return None

    def input_fn(self, partition, training):
        """Generate an input function for the Estimator."""
        def _input_fn():
            x_shape, y_shape, generator_fn = Traintest.generator_fn(
                self.traintest_file, partition, self.batch_size)

            dataset = tf.data.Dataset.from_generator(
                generator_fn,
                output_types=(tf.float32, tf.float32),
                output_shapes=((None, x_shape[1]),
                               (None, y_shape[1]))
            )
            # We call repeat after shuffling, rather than before,
            # to prevent separate epochs from blending together.
            if training:
                dataset = dataset.shuffle(
                    self.shuffles * self.batch_size,
                    seed=self.random_seed).repeat()
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
            return {'x': features}, labels

        return _input_fn

    @staticmethod
    def predict(model_dir, signature):
        """Predict on given testset."""
        predict_fn = predictor.from_saved_model(
            model_dir, signature_def_key='predict')
        pred = predict_fn({'x': signature[:]})
        return pred['predictions']

    def save_model(self, model_dir):
        def serving_input_fn():
            inputs = {
                "x": tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.input_dimension],
                                    name="x")
            }
            return tf.estimator.export.ServingInputReceiver(inputs, inputs)

        return self.estimator.export_saved_model(model_dir, serving_input_fn)

    @staticmethod
    def print_model_architechture(model_dir):
        """Print out the persistent NN structure."""
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], model_dir)
            model_vars = tf.trainable_variables()
            slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    @staticmethod
    def get_trainable_variables(model_dir):
        """Return the weigths of the persistent NN."""
        model_vars = list()
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], model_dir)
            for var in tf.trainable_variables():
                model_vars.append(var.eval())
        return model_vars

    def save_performances(self, output_dir, plot, extra_preditors=None):
        """Save stats and make plots."""
        # read input
        datasets = ['train', 'test', 'validation']
        x = dict()
        y = dict()
        for ds in datasets:
            # get dataset split
            traintest = Traintest(self.traintest_file, ds)
            traintest.open()
            x[ds] = traintest.get_all_x()
            y[ds] = traintest.get_all_y()
            traintest.close()
        # save in pandas
        df = pd.DataFrame(columns=[
            'dataset', 'r2', 'pearson_avg', 'pearson_std', 'algo', 'mse',
            'explained_variance', 'time', 'architecture', 'nr_variables',
            'nn_layers', 'layer_size', 'architecture_history'])

        def _stats_row(y_true, y_pred, algo, dataset):
            row = dict()
            row['algo'] = algo
            row['dataset'] = dataset
            row['layer_size'] = self.layer_size
            row['r2'] = r2_score(y_true, y_pred)
            pps = [pearsonr(y_true[:, x], y_pred[:, x])[0]
                   for x in range(y_true.shape[1])]
            row['pearson_avg'] = np.mean(pps)
            row['pearson_std'] = np.std(pps)
            row['mse'] = mean_squared_error(y_true, y_pred)
            row['explained_variance'] = explained_variance_score(
                y_true, y_pred)
            return row

        def _log_row(row):
            for k, v in row.items():
                if isinstance(v, float):
                    self.__log.debug("{:<24} {:>4.3f}".format(k, v))
                else:
                    self.__log.debug("{:<24} {}".format(k, v))

        # Performances for AdaNet
        rows = dict()
        for ds in datasets:
            self.__log.info("Performances for AdaNet on %s" % ds)
            y_pred = AdaNetWrapper.predict(self.save_dir, x[ds])
            rows[ds] = _stats_row(y[ds], y_pred, 'AdaNet', ds)
            rows[ds]['time'] = self.time
            rows[ds]['architecture_history'] = self.architecture()
            # log and save plot
            _log_row(rows[ds])
            plot.sign2_prediction_plot(y[ds], y_pred, "AdaNet_%s" % ds)

        # some additional shared stats
        # get nr of variables in final model
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], self.save_dir)
            model_vars = list()
            for var in tf.trainable_variables():
                model_vars.append(var.eval())
            nr_variables = np.sum([np.prod(v.shape) for v in model_vars])
            nn_layers = (len(model_vars) / 2) - 1
            architecture = [model_vars[i].shape[1]
                            for i in range(0, len(model_vars), 2)]

        # save rows
        for ds in datasets:
            rows[ds]["nr_variables"] = nr_variables
            rows[ds]["nn_layers"] = nn_layers
            rows[ds]["architecture"] = architecture
            df.loc[len(df)] = pd.Series(rows[ds])
        output_pkl = os.path.join(output_dir, 'stats.pkl')
        with open(output_pkl, 'wb') as fh:
            pickle.dump(df, fh)
        output_csv = os.path.join(output_dir, 'stats.csv')
        df.to_csv(output_csv)

        # compare to baseline Linear Regression
        linreg_start = time()
        linreg = LinearRegression().fit(x['train'], y['train'])
        linreg_stop = time()
        rows = dict()
        for ds in datasets:
            self.__log.info("Performances for LinearRegression on %s" % ds)
            y_pred = linreg.predict(x[ds])
            rows[ds] = _stats_row(y[ds], y_pred, 'LinearRegression', ds)
            rows[ds]['time'] = linreg_stop - linreg_start
            rows[ds]['architecture_history'] = '| linear |'
            rows[ds]['architecture'] = [y[ds].shape[1]]
            rows[ds]['layer_size'] = 0
            rows[ds]["nr_variables"] = y[ds].shape[1]
            rows[ds]["nn_layers"] = 0
            # log and save plot
            _log_row(rows[ds])
            plot.sign2_prediction_plot(
                y[ds], y_pred, "LinearRegression_%s" % ds)

        # save rows
        for ds in datasets:
            df.loc[len(df)] = pd.Series(rows[ds])
        output_pkl = os.path.join(output_dir, 'stats.pkl')
        with open(output_pkl, 'wb') as fh:
            pickle.dump(df, fh)
        output_csv = os.path.join(output_dir, 'stats.csv')
        df.to_csv(output_csv)

        # compare to other predictors
        if not extra_preditors:
            return

        for name, preds in extra_preditors.items():
            rows = dict()
            for ds in datasets:
                self.__log.info("Performances for %s on %s", name, ds)
                y_pred = preds[ds]
                rows[ds] = _stats_row(y[ds], y_pred, name, ds)
                rows[ds]['time'] = preds['time']
                rows[ds]['architecture_history'] = '| linear |'
                rows[ds]['architecture'] = [y[ds].shape[1]]
                rows[ds]['layer_size'] = 0
                rows[ds]["nr_variables"] = y[ds].shape[1]
                rows[ds]["nn_layers"] = 0
                # log and save plot
                _log_row(rows[ds])
                plot.sign2_prediction_plot(
                    y[ds], y_pred, "%s_%s" % (name, ds))

            # save rows
            for ds in datasets:
                df.loc[len(df)] = pd.Series(rows[ds])
            output_pkl = os.path.join(output_dir, 'stats.pkl')
            with open(output_pkl, 'wb') as fh:
                pickle.dump(df, fh)
            output_csv = os.path.join(output_dir, 'stats.csv')
            df.to_csv(output_csv)
