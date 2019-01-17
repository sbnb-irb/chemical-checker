import os
import h5py
import adanet
import numpy as np
import pandas as pd
from time import time
import tensorflow as tf
import pickle
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
import tensorflow.contrib.slim as slim

from .dnn_generator import SimpleDNNGenerator
from chemicalchecker.util import logged
from chemicalchecker.util import Plot


@logged
class Traintest(object):
    """Convenience batch reader from HDF5 files.

    This class allow creation and access to HDF5 train-test sets and expose
    the generator functions which tensorflow likes.
    """

    def __init__(self, hdf5_file, partition, batch_size):
        """Initialize the traintest object.

        We assume the file is containing train and test split i.e.
        x_train, y_train, x_test, y_test
        """
        self._file = hdf5_file
        self._f = None
        self.x_name = "x_%s" % partition
        self.y_name = "y_%s" % partition
        self.batch_size = batch_size

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
        return features, labels

    def get_x(self, beg_idx, end_idx):
        """Get the batch."""
        features = self._f[self.x_name][beg_idx: end_idx]
        return features

    @staticmethod
    def create(sign_from, sign_to, out_filename):
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
        Traintest.__log.info('shapes X %s  Y %s', X.shape, Y.shape)
        # train test split
        sp = ShuffleSplit(n_splits=1, train_size=0.8, random_state=0)
        train, test = list(sp.split(X))[0]
        # create dataset
        with h5py.File(out_filename, "w") as fh:
            fh.create_dataset('x_train', data=X[train], dtype=np.float32)
            fh.create_dataset('y_train', data=Y[train], dtype=np.float32)
            fh.create_dataset('x_test', data=X[test], dtype=np.float32)
            fh.create_dataset('y_test', data=Y[test], dtype=np.float32)
            Traintest.__log.debug("**** SHAPES ****")
            Traintest.__log.debug("x_train %s", fh["x_train"].shape)
            Traintest.__log.debug("y_train %s", fh["y_train"].shape)
            Traintest.__log.debug("x_test %s", fh["x_test"].shape)
            Traintest.__log.debug("y_test %s", fh["y_test"].shape)
            Traintest.__log.debug("**** SHAPES ****")
        fh.close()
        Traintest.__log.info('Traintest saved to %s', out_filename)

    @staticmethod
    def generator_fn(file_name, partition, batch_size=None, only_x=False):
        """Return the generator function that we can query for batches."""
        reader = Traintest(file_name, partition, batch_size)
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

    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.train_step = int(kwargs.get("train_step", 1000000))
        self.batch_size = int(kwargs.get("batch_size", 32))
        self.learn_mixture_weights = kwargs.get("learn_mixture_weights", True)
        self.adanet_lambda = kwargs.get("adanet_lambda", 0.001)
        self.boosting_iterations = int(kwargs.get("boosting_iterations", 64))
        self.random_seed = int(kwargs.get("random_seed", 42))
        self.model_dir = kwargs.get("model_dir", None)
        self.activation = kwargs.get("activation", tf.nn.relu)
        self.layer_size = int(kwargs.get("layer_size", 8))
        self.shuffles = int(kwargs.get("shuffles", 10))
        self.results = None
        self.estimator = None
        self.__log.info("**** AdaNet Parameters: ***")
        self.__log.info("{:<22}: {:>12}".format("model_dir", self.model_dir))
        self.__log.info("{:<22}: {:>12}".format(
            "learning_rate", self.learning_rate))
        self.__log.info("{:<22}: {:>12}".format("train_step", self.train_step))
        self.__log.info("{:<22}: {:>12}".format("batch_size", self.batch_size))
        self.__log.info("{:<22}: {:>12}".format(
            "learn_mixture_weights", self.learn_mixture_weights))
        self.__log.info("{:<22}: {:>12}".format(
            "adanet_lambda", self.adanet_lambda))
        self.__log.info("{:<22}: {:>12}".format(
            "boosting_iterations", self.boosting_iterations))
        self.__log.info("{:<22}: {:>12}".format(
            "random_seed", self.random_seed))
        self.__log.info("{:<22}: {:>12}".format("activation", self.activation))
        self.__log.info("{:<22}: {:>12}".format("layer_size", self.layer_size))
        self.__log.info("{:<22}: {:>12}".format("shuffles", self.shuffles))

    def train_and_evaluate(self, traintest_file):
        self.starttime = time()

        self.traintest_file = traintest_file
        with h5py.File(traintest_file, 'r') as hf:
            self.input_dimension = hf['x_test'].shape[1]
            self.label_dimension = hf['y_test'].shape[1]
            self.train_size = hf['x_train'].shape[0]
        self.train_step = self.boosting_iterations * self.train_size // self.batch_size

        """Train an `adanet.Estimator`."""
        self.estimator = adanet.Estimator(
            # We have amultiple-output regression problem,
            # so we'll use a regression head that optimizes for MSE.
            head=tf.contrib.estimator.regression_head(
                label_dimension=self.label_dimension,
                loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE),

            # Define the generator, which defines our search space of
            # subnetworks to train as candidates to add to the final AdaNet
            # model.
            subnetwork_generator=SimpleDNNGenerator(
                optimizer=tf.train.RMSPropOptimizer(
                    learning_rate=self.learning_rate),
                learn_mixture_weights=self.learn_mixture_weights,
                layer_size=self.layer_size,
                seed=self.random_seed),

            # Lambda is a the strength of complexity regularization. A larger
            # value will penalize more complex subnetworks.
            adanet_lambda=self.adanet_lambda,

            # The number of train steps per iteration.
            max_iteration_steps=self.train_step // self.boosting_iterations,

            # The evaluator will evaluate the model on the full training set to
            # compute the overall AdaNet loss (train loss + complexity
            # regularization) to select the best candidate to include in the
            # final AdaNet model.
            evaluator=adanet.Evaluator(
                input_fn=self.input_fn("train", training=False)),

            # Configuration for Estimators.
            config=tf.estimator.RunConfig(
                save_checkpoints_steps=50000,
                save_summary_steps=50000,
                tf_random_seed=self.random_seed),

            model_dir=self.model_dir
        )

        # Train and evaluate using using the tf.estimator tooling.
        train_spec = tf.estimator.TrainSpec(
            input_fn=self.input_fn("train", training=True),
            max_steps=self.train_step)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=self.input_fn("test", training=False),
            steps=None)

        self.results = tf.estimator.train_and_evaluate(
            self.estimator, train_spec, eval_spec)

        self.save_dir = self.save_model(self.model_dir)
        self.__log.info("SAVING MODEL TO: %s", self.save_dir)
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

    def test_input_fn(self, partition):
        """Generate a test input function for the Estimator."""
        def _input_fn():
            x_shape, y_shape, generator_fn = Traintest.generator_fn(
                self.traintest_file, partition, self.batch_size, only_x=True)

            dataset = tf.data.Dataset.from_generator(
                generator_fn,
                output_types=tf.float32,
                output_shapes=(None, x_shape[1])
            )
            iterator = dataset.make_one_shot_iterator()
            features = iterator.get_next()
            return {'x': features}
        return _input_fn

    def predict(self, partition):
        """Predict on given testset."""
        predict_results = self.estimator.predict(
            input_fn=self.test_input_fn(partition),
            yield_single_examples=False)
        return predict_results

    def save_model(self, model_dir):
        def serving_input_fn():
            serialized_tf_example = tf.placeholder(
                dtype=tf.string, shape=[None], name='input_tensors')
            receiver_tensors = {"predictor_inputs": serialized_tf_example}
            feature_spec = {"x": tf.FixedLenFeature(
                [self.input_dimension], tf.float32)}
            features = tf.parse_example(serialized_tf_example, feature_spec)
            return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

        return self.estimator.export_saved_model(model_dir, serving_input_fn)

    @staticmethod
    def print_model_architechture(model_dir):
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], model_dir)
            graph = tf.get_default_graph()
            model_vars = tf.trainable_variables()
            slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    def save_performances(self, output_dir, plot):
        self.time = time() - self.starttime

        df = pd.DataFrame(columns=[
            'dataset', 'r2', 'pearson_avg', 'pearson_std', 'algo', 'mse',
            'explained_variance', 'time', 'architecture', 'nr_variables'])
        with h5py.File(self.traintest_file, 'r') as hf:
            y_train = hf['y_train'][:]
            y_test = hf['y_test'][:]

        # Performances for AdaNet on TRAIN
        self.__log.info("Performances for AdaNet on TRAIN")
        y_train_pred = [y['predictions'] for y in self.predict("train")]
        y_train_pred = np.concatenate(y_train_pred)
        row_train = dict()
        row_train['algo'] = 'AdaNet'
        row_train['dataset'] = 'train'
        row_train['r2'] = r2_score(y_train, y_train_pred)
        pps = [pearsonr(y_train[:, x], y_train_pred[:, x])[0]
               for x in range(self.label_dimension)]
        row_train['pearson_avg'] = np.mean(pps)
        row_train['pearson_std'] = np.std(pps)
        row_train['mse'] = mean_squared_error(y_train, y_train_pred)
        row_train['explained_variance'] = explained_variance_score(
            y_train, y_train_pred)
        row_train['time'] = self.time
        row_train['architecture'] = self.architecture()
        for k, v in row_train.items():
            if isinstance(v, float):
                self.__log.debug("{:<24} {:>4.3f}".format(k, v))
            else:
                self.__log.debug("{:<24} {}".format(k, v))
        # save plot
        plot.sign2_plot(y_train, y_train_pred, "AdaNet_TRAIN")

        # Performances for AdaNet on TEST
        self.__log.info("Performances for AdaNet on TEST")
        y_test_pred = [y['predictions'] for y in self.predict("test")]
        y_test_pred = np.concatenate(y_test_pred)
        row_test = dict()
        row_test['algo'] = 'AdaNet'
        row_test['dataset'] = 'test'
        row_test['r2'] = r2_score(y_test, y_test_pred)
        pps = [pearsonr(y_test[:, x], y_test_pred[:, x])[0]
               for x in range(self.label_dimension)]
        row_test['pearson_avg'] = np.mean(pps)
        row_test['pearson_std'] = np.std(pps)
        row_test['mse'] = mean_squared_error(y_test, y_test_pred)
        row_test['explained_variance'] = explained_variance_score(
            y_test, y_test_pred)
        row_test['time'] = self.time
        row_test['architecture'] = self.architecture()
        for k, v in row_test.items():
            if isinstance(v, float):
                self.__log.debug("{:<24} {:>4.3f}".format(k, v))
            else:
                self.__log.debug("{:<24} {}".format(k, v))
        # save plot
        plot.sign2_plot(y_test, y_test_pred, "AdaNet_TEST")

        # get nr of variables in final model
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], self.save_dir)
            tf.get_default_graph()
            nr_variables = np.sum([np.prod(v.get_shape().as_list())
                                   for v in tf.trainable_variables()])

        # save rows
        row_test["nr_variables"] = nr_variables
        row_train["nr_variables"] = nr_variables
        df.loc[len(df)] = pd.Series(row_test)
        df.loc[len(df)] = pd.Series(row_train)
        output_pkl = os.path.join(output_dir, 'stats.pkl')
        with open(output_pkl, 'wb') as fh:
            pickle.dump(df, fh)
        output_csv = os.path.join(output_dir, 'stats.csv')
        df.to_csv(output_csv)

        # compare to simple Linear Regression on TRAIN
        self.__log.info("Performances for LinearRegression on TRAIN")
        with h5py.File(self.traintest_file, 'r') as hf:
            x_train = hf['x_train'][:]
        t0 = time()
        linreg = LinearRegression().fit(x_train, y_train)
        t1 = time()
        y_train_pred = linreg.predict(x_train)
        row_train = dict()
        row_train['algo'] = 'LinearRegression'
        row_train['dataset'] = 'train'
        row_train['r2'] = r2_score(y_train, y_train_pred)
        pps = [pearsonr(y_train[:, x], y_train_pred[:, x])[0]
               for x in range(self.label_dimension)]
        row_train['pearson_avg'] = np.mean(pps)
        row_train['pearson_std'] = np.std(pps)
        row_train['mse'] = mean_squared_error(y_train, y_train_pred)
        row_train['explained_variance'] = explained_variance_score(
            y_train, y_train_pred)
        row_train['time'] = t1 - t0
        row_train['architecture'] = '| linear |'
        row_train["nr_variables"] = self.label_dimension
        for k, v in row_train.items():
            if isinstance(v, float):
                self.__log.debug("{:<24} {:>4.3f}".format(k, v))
            else:
                self.__log.debug("{:<24} {}".format(k, v))
        # save plot
        plot.sign2_plot(y_train, y_train_pred, "LinearRegression_TRAIN")

        # compare to simple Linear Regression on TEST
        self.__log.info("Performances for LinearRegression on TEST")
        with h5py.File(self.traintest_file, 'r') as hf:
            x_test = hf['x_test'][:]
        y_test_pred = linreg.predict(x_test)
        row_test = dict()
        row_test['algo'] = 'LinearRegression'
        row_test['dataset'] = 'test'
        row_test['r2'] = r2_score(y_test, y_test_pred)
        pps = [pearsonr(y_test[:, x], y_test_pred[:, x])[0]
               for x in range(self.label_dimension)]
        row_test['pearson_avg'] = np.mean(pps)
        row_test['pearson_std'] = np.std(pps)
        row_test['mse'] = mean_squared_error(y_test, y_test_pred)
        row_test['explained_variance'] = explained_variance_score(
            y_test, y_test_pred)
        row_test['time'] = t1 - t0
        row_test['architecture'] = '| linear |'
        row_test["nr_variables"] = self.label_dimension
        for k, v in row_test.items():
            if isinstance(v, float):
                self.__log.debug("{:<24} {:>4.3f}".format(k, v))
            else:
                self.__log.debug("{:<24} {}".format(k, v))
        # save plot
        plot.sign2_plot(y_test, y_test_pred, "LinearRegression_TEST")

        # save rows
        df.loc[len(df)] = pd.Series(row_test)
        df.loc[len(df)] = pd.Series(row_train)
        with open(output_pkl, 'wb') as fh:
            pickle.dump(df, fh)
        df.to_csv(output_csv)
