import os
import h5py
import shutil
import pickle
import numpy as np
import pandas as pd
from time import time
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LinearRegression
try:
    import tensorflow as tf
    import tensorflow.contrib.slim as slim
    from tensorflow.contrib import predictor
except ImportError:
    raise ImportError("requires tensorflow " +
                      "https://www.tensorflow.org/")
try:
    import adanet
except ImportError:
    raise ImportError("requires adanet " +
                      "https://github.com/tensorflow/adanet")

from .dnn_stack_generator import StackDNNGenerator
from .dnn_extend_generator import ExtendDNNGenerator

from chemicalchecker.util import logged


@logged
class Traintest(object):
    """Convenience batch reader from HDF5 files.

    This class allow creation and access to HDF5 train-test sets and expose
    the generator functions which tensorflow likes.
    """

    def __init__(self, hdf5_file, split, replace_nan=None):
        """Initialize the traintest object.

        We assume the file is containing diffrent splits.
        e.g. "x_train", "y_train", "x_test", ...
        """
        self._file = hdf5_file
        self._f = None
        self.replace_nan = replace_nan
        if split is None:
            self.x_name = "x"
            self.y_name = "y"
        else:
            self.x_name = "x_%s" % split
            self.y_name = "y_%s" % split
            available_splits = self.get_split_names()
            if split not in available_splits:
                raise Exception("Split '%s' not found in %s!" %
                                (split, str(available_splits)))

    def get_xy_shapes(self):
        """Return the shpaes of X an Y."""
        self.open()
        x_shape = self._f[self.x_name].shape
        y_shape = self._f[self.y_name].shape
        self.close()
        return x_shape, y_shape

    def get_split_names(self):
        """Return the name of the splits."""
        self.open()
        if "split_names" in self._f:
            split_names = [a.decode() for a in self._f["split_names"]]
        else:
            split_names = ['train', 'test', 'validation']
        self.close()
        return split_names

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
        """Get a batch of X and Y."""
        features = self._f[self.x_name][beg_idx: end_idx]
        # handle NaNs
        if self.replace_nan is not None:
            features[np.where(np.isnan(features))] = self.replace_nan
        labels = self._f[self.y_name][beg_idx: end_idx]
        return features, labels

    def get_x(self, beg_idx, end_idx):
        """Get a batch of X."""
        features = self._f[self.x_name][beg_idx: end_idx]
        # handle NaNs
        if self.replace_nan is not None:
            features[np.where(np.isnan(features))] = self.replace_nan
        return features

    def get_all_x(self):
        """Get full X."""
        features = self._f[self.x_name][:]
        # handle NaNs
        if self.replace_nan is not None:
            features[np.where(np.isnan(features))] = self.replace_nan
        return features

    def get_all_x_columns(self, columns):
        """Get all the X.

        Args:
            colums(tuple(int,int)): start, stop indexes.
        """
        features = self._f[self.x_name][:, slice(*columns)]
        # handle NaNs
        if self.replace_nan is not None:
            features[np.where(np.isnan(features))] = self.replace_nan
        return features

    def get_all_y(self):
        """Get full Y."""
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
    def get_split_indeces(rows, fractions):
        """Get random indeces for different splits."""
        if not sum(fractions) == 1.0:
            raise Exception("Split fractions should sum to 1.0")
        # shuffle indeces
        idxs = list(range(rows))
        np.random.shuffle(idxs)
        # from frequs to indices
        splits = np.cumsum(fractions)
        splits = splits[:-1]
        splits *= len(idxs)
        splits = splits.round().astype(np.int)
        return np.split(idxs, splits)

    @staticmethod
    def create(X, Y, out_file, split_names=['train', 'test', 'validation'],
               split_fractions=[.8, .1, .1], x_dtype=np.float32,
               y_dtype=np.float32, chunk_size=10000):
        """Create the HDF5 file with validation splits for both X and Y.

        Args:
            X(numpy.ndarray): features to train from.
            Y(numpy.ndarray): labels to predict.
            out_file(str): path of the h5 file to write.
            split_names(list(str)): names for the split of data.
            split_fractions(list(float)): fraction of data in each split.
            x_dtype(type): numpy data type for X.
            y_dtype(type): numpy data type for Y (np.float32 for regression,
                np.int32 for classification.
        """
        # Force number of dimension to 2 (reshape Y)
        if Y.ndim == 1:
            Traintest.__log.debug("We need Y as a column vector, reshaping.")
            Y = np.reshape(Y, (len(Y), 1))
        Traintest.__log.debug(
            "{:<20} shape: {:>10}".format("input X", X.shape))
        Traintest.__log.debug(
            "{:<20} shape: {:>10}".format("input Y", Y.shape))
        # train test validation splits
        if len(split_names) != len(split_fractions):
            raise Exception("Split names and fraction should be same amount.")
        split_names = [s.encode() for s in split_names]
        split_idxs = Traintest.get_split_indeces(
            Y.shape[0], split_fractions)

        # create dataset
        Traintest.__log.info('Traintest saving to %s', out_file)
        with h5py.File(out_file, "w") as fh:
            fh.create_dataset('split_names', data=split_names)
            fh.create_dataset('split_fractions', data=split_fractions)
            for name, idxs in zip(split_names, split_idxs):
                ds_name = "x_%s" % name
                fh.create_dataset(ds_name, (len(idxs), X.shape[1]),
                                  dtype=x_dtype)
                for i in range(0, len(idxs), chunk_size):
                    chunk = slice(i, i + chunk_size)
                    fh[ds_name][chunk] = X[idxs[chunk]]
                Traintest.__log.debug("Written: {:<20} shape: {:>10}".format(
                    ds_name, fh[ds_name].shape))
                ds_name = "y_%s" % name
                fh.create_dataset(ds_name, (len(idxs), Y.shape[1]),
                                  dtype=y_dtype)
                for i in range(0, len(idxs), chunk_size):
                    chunk = slice(i, i + chunk_size)
                    fh[ds_name][chunk] = Y[idxs[chunk]]
                Traintest.__log.debug("Written: {:<20} shape: {:>10}".format(
                    ds_name, fh[ds_name].shape))
        Traintest.__log.info('Traintest saved to %s', out_file)

    @staticmethod
    def split_h5(in_file, out_file,
                 split_names=['train', 'test', 'validation'],
                 split_fractions=[.8, .1, .1], chunk_size=1000):
        """Create the HDF5 file with validation splits from an input file.

        Args:
            in_file(str): path of the h5 file to read from.
            out_file(str): path of the h5 file to write.
            split_names(list(str)): names for the split of data.
            split_fractions(list(float)): fraction of data in each split.
        """
        with h5py.File(in_file, 'r') as hf_in:
            # log input datasets and shapes
            for k in hf_in.keys():
                Traintest.__log.debug(
                    "{:<20} shape: {:>10}".format(k, str(hf_in[k].shape)))
                rows = hf_in[k].shape[0]

            # train test validation splits
            if len(split_names) != len(split_fractions):
                raise Exception(
                    "Split names and fraction should be same amount.")
            split_names = [s.decode() for s in split_names]
            split_idxs = Traintest.get_split_indeces(rows, split_fractions)

            Traintest.__log.info('Traintest saving to %s', out_file)
            with h5py.File(out_file, "w") as hf_out:
                # create fixed datasets
                hf_out.create_dataset(
                    'split_names', data=np.array(split_names))
                hf_out.create_dataset(
                    'split_fractions', data=np.array(split_fractions))

                for name, idxs in zip(split_names, split_idxs):
                    # for each original dataset
                    for k in hf_in.keys():
                        # create all splits
                        ds_name = "%s_%s" % (k, name.decode())
                        hf_out.create_dataset(ds_name,
                                              (len(idxs), hf_in[k].shape[1]),
                                              dtype=hf_in[k].dtype)
                        # fill-in by chunks
                        for i in range(0, len(idxs), chunk_size):
                            chunk = slice(i, i + chunk_size)
                            sorted_idxs = sorted(list(idxs[chunk]))
                            hf_out[ds_name][chunk] = hf_in[k][sorted_idxs]
                        Traintest.__log.debug(
                            "Written: {:<20} shape: {:>10}".format(
                                ds_name, str(hf_out[ds_name].shape)))
        Traintest.__log.info('Traintest saved to %s', out_file)

    @staticmethod
    def generator_fn(file_name, split, batch_size=None, only_x=False):
        """Return the generator function that we can query for batches."""
        reader = Traintest(file_name, split)
        reader.open()
        # read shapes
        x_shape = reader._f[reader.x_name].shape
        y_shape = reader._f[reader.y_name].shape
        # read data types
        x_dtype = reader._f[reader.x_name].dtype
        y_dtype = reader._f[reader.y_name].dtype
        # no batch size -> return everything
        if not batch_size:
            batch_size = x_shape[0]

        def example_generator_fn():
            # generator function yielding data
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

        return (x_shape, y_shape), (x_dtype, y_dtype), example_generator_fn


@logged
class AdaNetWrapper(object):
    """Wrapper class adapted from scripted examples on AdaNet's github.

    https://github.com/tensorflow/adanet/blob/master/adanet/
    examples/tutorials/adanet_objective.ipynb
    """

    def __init__(self, traintest_file, **kwargs):
        # read parameters
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.batch_size = int(kwargs.get("batch_size", 32))
        self.learn_mixture_weights = kwargs.get("learn_mixture_weights", True)
        self.adanet_lambda = kwargs.get("adanet_lambda", 0.001)
        self.random_seed = int(kwargs.get("random_seed", 42))
        self.model_dir = kwargs.get("model_dir", None)
        self.activation = kwargs.get("activation", tf.nn.relu)
        self.shuffles = int(kwargs.get("shuffles", 10))
        self.dropout_rate = float(kwargs.get("dropout_rate", 0.2))
        self.augmentation = kwargs.get("augmentation", False)
        self.min_train_step = kwargs.get("min_train_step", 1000)
        self.nan_mask_value = kwargs.get("nan_mask_value", 0.0)
        self.subnetwork_generator = eval(kwargs.get(
            "subnetwork_generator", "ExtendDNNGenerator"))
        self.initial_architecture = kwargs.get("initial_architecture", [])
        self.cpu = kwargs.get("cpu", 4)
        # read input shape
        self.traintest_file = traintest_file
        with h5py.File(traintest_file, 'r') as hf:
            x_ds = 'x'
            y_ds = 'y'
            if 'x_train' in hf.keys():
                x_ds = 'x_train'
                y_ds = 'y_train'
            self.input_dimension = hf[x_ds].shape[1]
            if len(hf[y_ds].shape) == 1:
                self.label_dimension = 1
            else:
                self.label_dimension = hf[y_ds].shape[1]
            self.train_size = hf[x_ds].shape[0]
            self.total_size = 0
            for split in [i for i in hf.keys() if i.startswith('x')]:
                self.total_size += hf[split].shape[0]
            # derive number of classes from train data
            self.n_classes = np.unique(hf[y_ds][:100000]).shape[0]
        # override number of classes if specified
        self.n_classes = kwargs.get("n_classes", self.n_classes)
        # layer size heuristic
        heu_layer_size = AdaNetWrapper.layer_size_heuristic(
            self.total_size, self.input_dimension, self.label_dimension)
        self.layer_size = int(kwargs.get("layer_size", heu_layer_size))
        # make adanet iteration proportional to input size (with lower bound)
        adanet_it, epoch_it = AdaNetWrapper.iteration_epoch_heuristic(
            self.total_size)
        self.epoch_per_iteration = int(
            kwargs.get("epoch_per_iteration", epoch_it))
        self.adanet_iterations = int(
            kwargs.get("adanet_iterations", adanet_it))
        # howevere we want to guarantee one epoch per adanet iteration
        self.train_step = int(np.ceil(self.train_size / self.batch_size *
                                      float(self.epoch_per_iteration)))
        if self.train_step < self.min_train_step:
            self.epoch_per_iteration = int(
                np.ceil(float(self.min_train_step) * self.batch_size /
                        self.train_size))
            self.__log.warn("Given input size (%s) would result in few train" +
                            " steps, increasing epoch per iterations to %s",
                            self.train_size, self.epoch_per_iteration)
            self.train_step = int(np.ceil(self.train_size / self.batch_size *
                                          float(self.epoch_per_iteration)))

        self.train_step = max(self.train_step, self.min_train_step)
        self.total_steps = self.train_step * self.adanet_iterations
        self.results = None
        self.estimator = None
        # check the prediction task at hand
        self.prediction_task = kwargs.get("prediction_task", "regression")
        if self.prediction_task == "regression":
            self._estimator_head = tf.contrib.estimator.regression_head(
                label_dimension=self.label_dimension)
        elif self.prediction_task == "classification":
            self._estimator_head = \
                tf.contrib.estimator.binary_classification_head()
            if self.n_classes > 2:
                self._estimator_head = tf.contrib.estimator.multi_class_head(
                    n_classes=self.n_classes)
        else:
            raise Exception("Prediction task '%s' not recognized.",
                            self.prediction_task)
        # tensorflow session_config
        self.session_config = tf.ConfigProto(
            intra_op_parallelism_threads=self.cpu,
            inter_op_parallelism_threads=self.cpu,
            allow_soft_placement=True,
            device_count={'CPU': self.cpu})

        # log parameters
        self.__log.info("**** AdaNet Parameters: ***")
        self.__log.info("{:<22}: {:>12}".format(
            "prediction_task", self.prediction_task))
        if "classification" in self.prediction_task:
            self.__log.info("{:<22}: {:>12}".format(
                "n_classes", self.n_classes))
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
        self.__log.info("{:<22}: {:>12}".format(
            "activation", str(self.activation)))
        self.__log.info("{:<22}: {:>12}".format("layer_size", self.layer_size))
        self.__log.info("{:<22}: {:>12}".format("shuffles", self.shuffles))
        self.__log.info("{:<22}: {:>12}".format(
            "dropout_rate", self.dropout_rate))
        self.__log.info("{:<22}: {:>12}".format(
            "subnetwork_generator", str(self.subnetwork_generator)))
        self.__log.info("{:<22}: {:>12}".format(
            "train_step", self.train_step))
        self.__log.info("{:<22}: {:>12}".format(
            "total_steps", self.total_steps))
        self.__log.info("{:<22}: {:>12}".format(
            "augmentation", str(self.augmentation)))
        self.__log.info("{:<22}: {:>12}".format(
            "epoch_per_iteration", str(self.epoch_per_iteration)))
        self.__log.info("{:<22}: {:>12}".format(
            "nan_mask_value", str(self.nan_mask_value)))
        self.__log.info("{:<22}: {:>12}".format(
            "initial_architecture", str(self.initial_architecture)))
        self.__log.info("{:<22}: {:>12}".format("cpu", str(self.cpu)))
        self.__log.info("**** AdaNet Parameters: ***")

    @staticmethod
    def layer_size_heuristic(nr_samples, nr_features, nr_out=128, s_fact=7.):
        heu_layer_size = (
            1 / s_fact) * (np.sqrt(nr_samples) / .3 + ((nr_features + nr_out) / 5.))
        heu_layer_size = np.power(2, np.ceil(np.log2(heu_layer_size)))
        heu_layer_size = np.maximum(heu_layer_size, 32)
        return heu_layer_size

    @staticmethod
    def iteration_epoch_heuristic(nr_samples, min_it=3, max_it=10, min_ep=30,
                                  max_ep=300, sigmoid_midpoint=100000,
                                  steepness=1):
        # logistic function of nr of samples
        adanet_it = np.int32(min_it + (max_it - min_it) /
                             (1 + np.exp(steepness * (nr_samples - sigmoid_midpoint))))
        epoch_it = np.int32(min_ep + (max_ep - min_ep) /
                            (1 + np.exp(steepness * (nr_samples - sigmoid_midpoint))))
        return adanet_it, epoch_it

    def train_and_evaluate(self, evaluate=True):
        """Train and evaluate AdaNet."""
        # Define the `adanet.Evaluator`
        if evaluate:
            self.evaluator = adanet.Evaluator(
                input_fn=self.input_fn("train", training=False))
        else:
            self.evaluator = adanet.Evaluator(
                input_fn=self.input_fn(None, training=False))
        # Define the `adanet.Estimator`
        self.estimator = adanet.Estimator(
            # We'll use a regression head defined during initialization.
            head=self._estimator_head,

            # Define the generator, which defines our search space of
            # subnetworks to train as candidates to add to the final AdaNet
            # model.
            subnetwork_generator=self.subnetwork_generator(
                optimizer=tf.train.RMSPropOptimizer(
                    learning_rate=self.learning_rate),
                input_shape=self.input_dimension,
                nan_mask_value=self.nan_mask_value,
                learn_mixture_weights=self.learn_mixture_weights,
                layer_size=self.layer_size,
                dropout=self.dropout_rate,
                activation=self.activation,
                seed=self.random_seed,
                initial_architecture=self.initial_architecture),

            # Lambda is a the strength of complexity regularization. A larger
            # value will penalize more complex subnetworks.
            adanet_lambda=self.adanet_lambda,

            # The number of train steps per iteration.
            max_iteration_steps=self.train_step,

            # The evaluator will evaluate the model on the full training set to
            # compute the overall AdaNet loss (train loss + complexity
            # regularization) to select the best candidate to include in the
            # final AdaNet model.
            evaluator=self.evaluator,

            # Configuration for Estimators.
            config=tf.estimator.RunConfig(
                save_checkpoints_secs=18000,  # save checkpoints every 5 hours
                save_summary_steps=50000,
                tf_random_seed=self.random_seed,
                model_dir=self.model_dir,
                session_config=self.session_config),
            model_dir=self.model_dir
        )
        # Train and evaluate using using the tf.estimator tooling.
        if evaluate:
            train_spec = tf.estimator.TrainSpec(
                input_fn=self.input_fn("train", training=True,
                                       augmentation=self.augmentation),
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
        else:
            # call train and train only collecting time stats
            t0 = time()
            self.results = self.estimator.train(
                input_fn=self.input_fn(None, training=True,
                                       augmentation=self.augmentation),
                max_steps=self.total_steps)
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

    def input_fn(self, split, training, augmentation=False):
        """Generate an input function for the Estimator.

        Args:
            split(str): the split to use within the traintest file.
            training(bool): whether we are training or evaluating.
            augmentation(func): a function to aument data, False if no
                aumentation is desired.
        """
        def _input_fn():
            # get shapes, dtypes, and generator function
            (x_shape, y_shape), dtypes, generator_fn = Traintest.generator_fn(
                self.traintest_file, split, self.batch_size)
            # create dataset object
            dataset = tf.data.Dataset.from_generator(
                generator_fn,
                output_types=dtypes,
                output_shapes=(tf.TensorShape([None, x_shape[1]]),
                               tf.TensorShape([None, y_shape[1]]))
            )
            # We call repeat after shuffling, rather than before,
            # to prevent separate epochs from blending together.
            if training:
                dataset = dataset.shuffle(
                    self.shuffles * self.batch_size,
                    seed=self.random_seed).repeat()
                if augmentation:
                    dataset = dataset.map(lambda x, y: tuple(
                        tf.py_function(augmentation, [x, y],
                                       [x.dtype, y.dtype])))
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
            return {'x': features}, labels

        return _input_fn

    @staticmethod
    def predict(features, predict_fn=None, mask_fn=None, probs=False,
                samples=10, model_dir=None, zero_centered=False):
        """Load model and return predictions.

        Args:
            model_dir(str): path where to save the model.
            features(matrix): a numpy matrix of Xs.
            predict_fn(func): the predict function returned by `predict_fn`.
            probs(bool): if this is a classifier return the probabilities.
            zero_centered(bool): all 0s as input result in 0s output
                (regression only).
        """
        if predict_fn is None:
            predict_fn = predictor.from_saved_model(
                model_dir, signature_def_key='predict')

        if mask_fn is None:
            # TODO if no subsampling is provided we can apply some noise
            def mask_fn(data):
                return data
        pred = predict_fn({'x': features[:]})
        if 'predictions' in pred:
            if zero_centered:
                zero_feat = np.zeros((1, features.shape[1]), dtype=np.float32)
                zero_pred = predict_fn({'x': zero_feat})['predictions']
            if probs:
                pred_shape = pred['predictions'].shape
                # axis are 0=molecules, 1=components, 2=samples
                results = np.ndarray((pred_shape[0], pred_shape[1], samples))
                for idx in range(samples):
                    mask_pred = predict_fn({'x': mask_fn(features[:])})
                    results[:, :, idx] = mask_pred['predictions']
                if zero_centered:
                    return results - np.expand_dims(zero_pred, axis=2)
                else:
                    return results
            else:
                if zero_centered:
                    return pred['predictions'] - zero_pred
                else:
                    return pred['predictions']
        else:
            if probs:
                return pred['probabilities']
            else:
                return pred['class_ids']

    @staticmethod
    def predict_fn(model_dir):
        """Load model and return the predict function.

        Args:
            model_dir(str): path where to save the model.
        """
        predict_fn = predictor.from_saved_model(
            model_dir, signature_def_key='predict')
        return predict_fn

    @staticmethod
    def predict_online(h5_file, split, predict_fn=None,
                       mask_fn=None, batch_size=10000, limit=10000,
                       probs=False, n_classes=None, model_dir=None):
        """Predict on given testset without killing the memory.

        Args:
            model_dir(str): path where to save the model.
            h5_file(str): path to h5 file compatible with `Traintest`.
            split(str): the split to use within the h5_file.
            predict_fn(func): the predict function returned by `predict_fn`.
            mask_fn(func): a function masking part of the input.
            batch_size(int): batch size for `Traintest` file.
            limit(int): maximum number of predictions.
            probs(bool): if this is a classifier return the probabilities.
        """
        if predict_fn is None:
            predict_fn = predictor.from_saved_model(
                model_dir, signature_def_key='predict')
        shapes, dtypes, fn = Traintest.generator_fn(
            h5_file, split, batch_size, only_x=False)
        x_shape, y_shape = shapes
        x_dtype, y_dtype = dtypes
        # tha max size of the return prediction is at most same size as input
        y_pred = np.full(y_shape, np.nan, dtype=x_dtype)
        if probs:
            if n_classes is None:
                raise Exception("Specify number of classes.")
            y_pred = np.full((y_shape[0], n_classes), np.nan, dtype=x_dtype)
        y_true = np.full(y_shape, np.nan, dtype=y_dtype)
        last_idx = 0
        if y_shape[0] < limit:
            limit = y_shape[0]
        if mask_fn is None:
            def mask_fn(x, y):
                return x, y
        for x_data, y_data in fn():
            x_m, y_m = mask_fn(x_data, y_data)
            if x_m.shape[0] == 0:
                continue
            y_m_pred = AdaNetWrapper.predict(x_m, predict_fn, probs=probs)
            y_true[last_idx:last_idx + len(y_m)] = y_m
            y_pred[last_idx:last_idx + len(y_m)] = y_m_pred
            last_idx += len(y_m)
            if last_idx >= limit:
                break
        # we might not reach the limit
        if last_idx < limit:
            limit = last_idx
        return y_pred[:limit], y_true[:limit]

    def save_model(self, model_dir):
        """Print out the NAS network architechture structure.

        Args:
            model_dir(str): path where to save the model.
        """
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
        """Print out the NAS network architechture structure.

        Args:
            model_dir(str): path where of the saved model.
        """
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], model_dir)
            model_vars = tf.trainable_variables()
            slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    @staticmethod
    def get_trainable_variables(model_dir):
        """Return the weigths of the trained neural network.

        Args:
            model_dir(str): path where of the saved model.
        """
        model_vars = list()
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], model_dir)
            for var in tf.trainable_variables():
                model_vars.append(var.eval())
        return model_vars

    def save_performances(self, output_dir, plot,
                          suffix=None, extra_predictors=None, do_plot=False):
        """Save stats and make plots."""
        # read input
        splits = ['train', 'test', 'validation']
        # save in pandas
        df = pd.DataFrame(columns=[
            'dataset', 'split', 'component', 'r2', 'pearson', 'algo', 'mse',
            'explained_variance', 'time', 'architecture', 'nr_variables',
            'nn_layers', 'layer_size', 'architecture_history', 'from',
            'dataset_size', 'coverage'])

        def _stats_row(y_true, y_pred, algo, split, dataset):
            rows = list()
            for comp in range(y_true.shape[1]):
                row = dict()
                row['algo'] = algo
                row['split'] = split
                row['dataset'] = split
                row['dataset_size'] = y_true.shape[0]
                row['component'] = comp
                comp_res = y_true[:, comp].flatten(), y_pred[:, comp].flatten()
                row['r2'] = r2_score(*comp_res)
                row['pearson'] = pearsonr(*comp_res)[0]
                row['mse'] = mean_squared_error(*comp_res)
                row['explained_variance'] = explained_variance_score(*comp_res)
                row['from'] = dataset
                # self.__log.debug("comp: %s p: %.2f", comp, row['pearson'])
                rows.append(row)
            return rows

        def _update_row(rows, key, value):
            for row in rows:
                row[key] = value
            return rows

        def _log_row(row):
            for k, v in row.items():
                if isinstance(v, float):
                    self.__log.debug("{:<24} {:>4.3f}".format(k, v))
                else:
                    self.__log.debug("{:<24} {}".format(k, v))

        # Performances for AdaNet
        rows = dict()
        # load network
        predict_fn = AdaNetWrapper.predict_fn(self.save_dir)
        for split in splits:
            self.__log.info("Performances for AdaNet on %s" % split)
            y_pred, y_true = AdaNetWrapper.predict_online(
                self.traintest_file, split, predict_fn=predict_fn)
            if suffix:
                name = "AdaNet_%s" % suffix
            else:
                name = 'AdaNet'
            rows[split] = _stats_row(y_true, y_pred, name, split, "ALL")
            rows[split] = _update_row(rows[split], "time", self.time)
            rows[split] = _update_row(
                rows[split], "architecture_history", self.architecture())
            # log and save plot
            # _log_row(rows[split])
            if do_plot:
                plot.sign2_prediction_plot(y_true, y_pred, "AdaNet_%s" % split)

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
        for split in splits:
            rows[split] = _update_row(
                rows[split], "nr_variables", nr_variables)
            rows[split] = _update_row(rows[split], "nn_layers", nn_layers)
            rows[split] = _update_row(
                rows[split], "architecture", architecture)
            rows[split] = _update_row(rows[split], "coverage", 1.0)
            df = df.append(pd.DataFrame(rows[split]), ignore_index=True)
        output_pkl = os.path.join(output_dir, 'stats.pkl')
        with open(output_pkl, 'wb') as fh:
            pickle.dump(df, fh)
        output_csv = os.path.join(output_dir, 'stats.csv')
        df.to_csv(output_csv)

        '''
        # TODO use out-of-core linear regression
        # compare to baseline Linear Regression
        linreg_start = time()
        linreg = LinearRegression().fit(x['train'], y['train'])
        linreg_stop = time()
        rows = dict()
        for split in splits:
            self.__log.info("Performances for LinearRegression on %s" % split)
            y_pred = linreg.predict(x[split])
            rows[split] = _stats_row(
                y[split], y_pred, 'LinearRegression', split, "ALL")
            rows[split] = _update_row(
                rows[split], "time", linreg_stop - linreg_start)
            rows[split] = _update_row(
                rows[split], "architecture_history", '| linear |')
            rows[split] = _update_row(
                rows[split], "architecture", [y[split].shape[1]])
            rows[split] = _update_row(rows[split], "layer_size", 0)
            rows[split] = _update_row(
                rows[split], "nr_variables", [y[split].shape[1]])
            rows[split] = _update_row(rows[split], "nn_layers", 0)
            rows[split] = _update_row(rows[split], "coverage", 1.0)
            # log and save plot
            #_log_row(rows[split])if do_plot:
            #    plot.sign2_prediction_plot(
            #    y[split], y_pred, "LinearRegression_%s" % split)

        # save rows
        for split in splits:
            df = df.append(pd.DataFrame(rows[split]), ignore_index=True)
        output_pkl = os.path.join(output_dir, 'stats.pkl')
        with open(output_pkl, 'wb') as fh:
            pickle.dump(df, fh)
        output_csv = os.path.join(output_dir, 'stats.csv')
        df.to_csv(output_csv)
        '''

        # compare to other predictors
        if not extra_predictors:
            return

        for name in sorted(extra_predictors):
            preds = extra_predictors[name]
            rows = dict()
            for split in splits:
                if split not in preds:
                    self.__log.info("Skipping %s on %s", name, split)
                    continue
                if preds[split] is None:
                    self.__log.info("Skipping %s on %s", name, split)
                    continue
                self.__log.info("Performances for %s on %s", name, split)
                algo, dataset = name
                y_pred = np.load(preds[split]['pred'] + ".npy")
                y_true = np.load(preds[split]['true'] + ".npy")
                runtime = preds[split]['time']
                coverage = preds[split]['coverage']
                rows[split] = _stats_row(y_true, y_pred, algo, split, dataset)
                rows[split] = _update_row(rows[split], "coverage", coverage)
                rows[split] = _update_row(rows[split], "time", runtime)
                rows[split] = _update_row(
                    rows[split], "architecture_history", '| linear |')
                rows[split] = _update_row(
                    rows[split], "architecture", [y_true.shape[1]])
                rows[split] = _update_row(rows[split], "layer_size", 0)
                rows[split] = _update_row(
                    rows[split], "nr_variables", [y_true.shape[1]])
                rows[split] = _update_row(rows[split], "nn_layers", 0)
                # log and save plot
                # _log_row(rows[split])
                if do_plot:
                    plot.sign2_prediction_plot(
                        y_true, y_pred, "_".join(list(name) + [split]))

            # save rows
            for split in rows:
                df = df.append(pd.DataFrame(rows[split]), ignore_index=True)
            output_pkl = os.path.join(output_dir, 'stats.pkl')
            with open(output_pkl, 'wb') as fh:
                pickle.dump(df, fh)
            output_csv = os.path.join(output_dir, 'stats.csv')
            df.to_csv(output_csv)
