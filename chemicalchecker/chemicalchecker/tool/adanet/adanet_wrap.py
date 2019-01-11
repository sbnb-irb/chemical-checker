import h5py
import adanet
import numpy as np
import tensorflow as tf
from sklearn.model_selection import ShuffleSplit

from .dnn_generator import SimpleDNNGenerator
from chemicalchecker.util import logged


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
        self.__log.debug("**** SHAPES ****")
        self.__log.debug("x_train %s", self._f["x_train"].shape)
        self.__log.debug("y_train %s", self._f["y_train"].shape)
        self.__log.debug("x_test %s", self._f["x_test"].shape)
        self.__log.debug("y_test %s", self._f["y_test"].shape)
        self.__log.debug("**** SHAPES ****")

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
        self.boosting_iterations = int(kwargs.get("boosting_iterations", 100))
        self.random_seed = int(kwargs.get("random_seed", 42))
        self.model_dir = kwargs.get("model_dir", None)
        self.activation = kwargs.get("activation", tf.nn.relu)
        self.layer_size = int(kwargs.get("layer_size", 8))
        self.shuffles = int(kwargs.get("shuffles", 10))
        self.results = None
        self.estimator = None

    def train_and_evaluate(self, traintest_file):

        self.traintest_file = traintest_file
        with h5py.File(traintest_file, 'r') as hf:
            self.label_dimension = hf['y_test'].shape[1]

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


"""
import h5py
from vec4bio.algorithms import Adanet

traintest_file = './train_test/E1.TRAIN_TEST.h5'
ad = Adanet(model_dir='./models/E1/', train_step=100)
estimator, results = ad.train_and_evaluate(traintest_file)

with h5py.File(traintest_file, 'r') as hf:
    x_test = hf['x_test'][:]
    y_test = hf['y_test'][:]

y_pred = ad.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
print "R2", r2_score(y_test, y_pred)
print "MSE", mean_squared_error(y_test, y_pred)
"""
