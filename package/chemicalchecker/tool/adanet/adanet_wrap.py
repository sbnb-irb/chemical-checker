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
try:
    import tensorflow.compat.v1 as tf
    import tensorflow as tf2
    #import tensorflow.contrib.slim as slim
    #from tensorflow.contrib import predictor
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
from chemicalchecker.util.splitter import Traintest


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
        self.activation = kwargs.get("activation", tf.nn.tanh)
        self.shuffles = int(kwargs.get("shuffles", 10))
        self.dropout_rate = float(kwargs.get("dropout_rate", 0.2))
        self.augmentation = kwargs.get("augmentation", False)
        self.nan_mask_value = kwargs.get("nan_mask_value", 0.0)
        self.subnetwork_generator = eval(kwargs.get(
            "subnetwork_generator", "StackDNNGenerator"))
        self.extension_step = kwargs.get("extension_step", 1)
        self.initial_architecture = kwargs.get("initial_architecture", [1])
        self.cpu = kwargs.get("cpu", 4)
        # read input shape
        self.traintest_file = traintest_file
        with h5py.File(traintest_file, 'r') as hf:
            x_ds = 'x'
            y_ds = 'y'

            decoded_keys=[k.decode() if type(k) is bytes else k for k in hf.keys()]  # NS convert the bytes into strings
            if 'x_train' in decoded_keys:
                x_ds = 'x_train'
                y_ds = 'y_train'
            self.input_dimension = hf[x_ds].shape[1]
            if len(hf[y_ds].shape) == 1:
                self.label_dimension = 1
            else:
                self.label_dimension = hf[y_ds].shape[1]
            self.train_size = hf[x_ds].shape[0]
            self.total_size = 0
            for split in [i for i in decoded_keys if i.startswith('x')]:
                self.total_size += hf[split].shape[0]
            # derive number of classes from train data
            self.n_classes = np.unique(hf[y_ds][:100000]).shape[0]
        # override number of classes if specified
        self.n_classes = kwargs.get("n_classes", self.n_classes)
        # layer size
        self.layer_size = int(kwargs.get("layer_size", 1024))
        # make adanet iteration proportional to input size (with lower bound)
        self.epoch_per_iteration = int(kwargs.get("epoch_per_iteration", 8))
        self.adanet_iterations = int(kwargs.get("adanet_iterations", 10))
        # however we want to guarantee one epoch per adanet iteration
        self.train_step = int(np.ceil(self.train_size / self.batch_size *
                                      float(self.epoch_per_iteration)))
        self.total_steps = self.train_step * self.adanet_iterations
        self.results = None
        self.estimator = None
        # check the prediction task at hand
        self.prediction_task = kwargs.get("prediction_task", "regression")
        if self.prediction_task == "regression":
            self._estimator_head = tf.estimator.RegressionHead(
                label_dimension=self.label_dimension)
        elif self.prediction_task == "classification":
            self._estimator_head = \
                tf.estimator.BinaryClassHead()
            if self.n_classes > 2:
                self._estimator_head = tf.estimator.MultiClassHead(
                    n_classes=self.n_classes)
        else:
            raise Exception("Prediction task '%s' not recognized.",
                            self.prediction_task)


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
        self.__log.info("{:<22}: {:>12}".format(
            "extension_step", str(self.extension_step)))
        self.__log.info("{:<22}: {:>12}".format("cpu", str(self.cpu)))
        self.__log.info("**** AdaNet Parameters: ***")

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
                initial_architecture=self.initial_architecture,
                extension_step=self.extension_step),

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
                model_dir=self.model_dir),
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
            arch_txt = summary_proto.value[0].tensor.string_val[0]
            return [x.strip().split('_')[0] for x in arch_txt.split('|')[1:-1]]
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
                self.traintest_file, split, self.batch_size, return_on_epoch=True)
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
                                       [x.dtype, y.dtype])),
                        num_parallel_calls=self.cpu)
            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            features, labels = iterator.get_next()
            return {'x': features}, labels

        return _input_fn

    @staticmethod
    def predict(features, predict_fn=None, mask_fn=None, probs=False,
                samples=10, model_dir=None, consensus=False):
        """Load model and return predictions.

        Args:
            model_dir(str): path where to save the model.
            features(matrix): a numpy matrix of Xs.
            predict_fn(func): the predict function returned by `predict_fn`.
            probs(bool): if this is a classifier return the probabilities.
            consensus(bool): return also a sampling for consensus calculation.
                (regression only).
        """
        if predict_fn is None:
            imported = tf2.saved_model.load(model_dir)
            predict_fn = imported.signatures["predict"]
            #predict_fn = predictor.from_saved_model(
            #    model_dir, signature_def_key='predict')

        if mask_fn is None:
            # TODO if no subsampling is provided we can apply some noise
            def mask_fn(data):
                return data
        pred = predict_fn(tf2.convert_to_tensor(features[:], dtype=tf2.float32))
        if 'predictions' in pred:
            if consensus:
                pred_shape = pred['predictions'].shape
                # axis are 0=molecules, 1=samples, 2=components
                repeat = features[:].repeat(samples, axis=0)
                sampling = predict_fn(tf2.convert_to_tensor(mask_fn(repeat), dtype=tf2.float32))['predictions']
                sampling = sampling.reshape(
                    pred_shape[0], samples, pred_shape[1])
                return pred['predictions'], sampling
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
        imported = tf2.saved_model.load(model_dir)
        predict_fn = imported.signatures["predict"]
        #predict_fn = predictor.from_saved_model(
        #    model_dir, signature_def_key='predict')
        return predict_fn

    @staticmethod
    def predict_online(h5_file, split, predict_fn=None,
                       mask_fn=None, batch_size=1000, limit=None,
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
            imported = tf2.saved_model.load(model_dir)
            predict_fn = imported.signatures["predict"]
            #predict_fn = predictor.from_saved_model(
            #    model_dir, signature_def_key='predict')
        shapes, dtypes, fn = Traintest.generator_fn(
            h5_file, split, batch_size, only_x=False, return_on_epoch=True)
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
        if limit is None:
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
            #slim.model_analyzer.analyze_vars(model_vars, print_info=True)

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
                          suffix=None, extra_predictors=None, do_plot=True):
        """Save stats and make plots."""
        # read input
        splits = ['train', 'test', 'validation']
        # save in pandas
        df = pd.DataFrame(columns=[
            'dataset', 'split', 'component', 'r2', 'pearson', 'algo', 'mse',
            'explained_variance', 'time', 'architecture',
            'nr_variables', 'nn_layers', 'layer_size',
            'from', 'dataset_size', 'coverage'])

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
            self.__log.info("Performances %s\t%s\t%.2f\t%s" % (
                algo, dataset, np.mean([r['pearson'] for r in rows]), split))
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

        # output files
        if suffix is None:
            output_pkl = os.path.join(output_dir, 'stats.pkl')
            output_csv = os.path.join(output_dir, 'stats.csv')
            algo_name = 'AdaNet'
        else:
            output_pkl = os.path.join(output_dir, 'stats_%s.pkl' % suffix)
            output_csv = os.path.join(output_dir, 'stats_%s.csv' % suffix)
            algo_name = "AdaNet_%s" % suffix
        # Performances for AdaNet
        rows = dict()
        predict_fn = AdaNetWrapper.predict_fn(self.save_dir)
        for split in splits:
            y_pred, y_true = AdaNetWrapper.predict_online(
                self.traintest_file, split, predict_fn=predict_fn, limit=1000)
            rows[split] = _stats_row(y_true, y_pred, algo_name, split, "ALL")
            rows[split] = _update_row(rows[split], "time", self.time)
            rows[split] = _update_row(
                rows[split], "architecture", self.architecture())
            rows[split] = _update_row(
                rows[split], "layer_size", self.layer_size)
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

        # save rows
        for split in splits:
            rows[split] = _update_row(
                rows[split], "nr_variables", nr_variables)
            rows[split] = _update_row(rows[split], "nn_layers", nn_layers)
            rows[split] = _update_row(
                rows[split], "architecture", self.architecture())
            rows[split] = _update_row(rows[split], "coverage", 1.0)
            df = pd.concat([ df, pd.DataFrame(rows[split], columns=df.columns) ], ignore_index=True)
        with open(output_pkl, 'wb') as fh:
            pickle.dump(df, fh)
        df.to_csv(output_csv)

        # other predictors to compare?
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
                algo, dataset = name
                y_pred = np.load(preds[split]['pred'] + ".npy")
                y_true = np.load(preds[split]['true'] + ".npy")
                runtime = preds[split]['time']
                coverage = preds[split]['coverage']
                rows[split] = _stats_row(y_true, y_pred, algo, split, dataset)
                rows[split] = _update_row(rows[split], "coverage", coverage)
                rows[split] = _update_row(rows[split], "time", runtime)
                # log and save plot
                # _log_row(rows[split])
                if do_plot:
                    plot.sign2_prediction_plot(
                        y_true, y_pred, "_".join(list(name) + [split]))
            # save rows
            for split in rows:
                df = pd.concat([ df, pd.DataFrame(rows[split], columns=df.columns) ], ignore_index=True)
            with open(output_pkl, 'wb') as fh:
                pickle.dump(df, fh)
            df.to_csv(output_csv)
