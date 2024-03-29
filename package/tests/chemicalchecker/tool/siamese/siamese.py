import os
import pickle
import numpy as np
from time import time
from functools import partial

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers import Input, Dropout, Lambda, Dense

from chemicalchecker.util import logged
#from chemicalchecker.util.splitter import NeighborPairTraintest


@logged
class Siamese(object):
    """Siamese class.

    This class implements a simple siamese neural network based on Keras that
    allows metric learning.
    """

    def __init__(self, model_dir, traintest_file=None, evaluate=False, **kwargs):
        """Initialize the Siamese class.

        Args:
            model_dir(str): Directorty where models will be stored.
            traintest_file(str): Path to the traintest file.
            evaluate(bool): Whether to run evaluation.
        """
        from chemicalchecker.core.signature_data import DataSignature
        # read parameters
        self.epochs = int(kwargs.get("epochs", 10))
        self.batch_size = int(kwargs.get("batch_size", 100))
        self.learning_rate = float(kwargs.get("learning_rate", 1e-3))
        self.replace_nan = float(kwargs.get("replace_nan", 0.0))
        self.dropout = float(kwargs.get("dropout", 0.2))
        self.suffix = str(kwargs.get("suffix", 'eval'))
        self.split = str(kwargs.get("split", 'train'))
        self.layers = kwargs.get("layers", [128])
        self.augment_fn = kwargs.get("augment_fn", None)
        self.augment_kwargs = kwargs.get("augment_kwargs", None)
        self.augment_scale = int(kwargs.get("augment_scale", 1))

        # internal variables
        self.name = '%s_%s' % (self.__class__.__name__.lower(), self.suffix)
        self.time = 0
        self.output_dim = None
        self.model_dir = os.path.abspath(model_dir)
        self.model_file = os.path.join(self.model_dir, "%s.h5" % self.name)
        self.model = None
        self.evaluate = evaluate

        # check output path
        if not os.path.exists(model_dir):
            self.__log.warning("Creating model directory: %s", self.model_dir)
            os.mkdir(self.model_dir)

        # check if a scaler is available
        scaler_file = os.path.join(self.model_dir, 'scaler.pkl')
        if os.path.isfile(scaler_file):
            scaler = pickle.load(open(scaler_file, 'rb'))
            self.set_predict_scaler(scaler)
        else:
            self.__log.warn('No scaler available. %s' % scaler_file)

        # check input path
        self.traintest_file = traintest_file
        if self.traintest_file is not None:
            self.traintest_file = os.path.abspath(traintest_file)
            if not os.path.exists(traintest_file):
                raise Exception('Input data file does not exists!')

            # initialize train generator
            self.sharedx = DataSignature(traintest_file).get_h5_dataset('x')
            tr_shape_type_gen = NeighborPairTraintest.generator_fn(
                self.traintest_file,
                'train_train',
                batch_size=int(self.batch_size / self.augment_scale),
                replace_nan=self.replace_nan,
                sharedx=self.sharedx,
                augment_fn=self.augment_fn,
                augment_kwargs=self.augment_kwargs,
                augment_scale=self.augment_scale)
            self.tr_shapes = tr_shape_type_gen[0]
            self.tr_gen = tr_shape_type_gen[2]()
            self.steps_per_epoch = np.ceil(
                self.tr_shapes[0][0] / self.batch_size)
            self.output_dim = tr_shape_type_gen[0][1][1]

        # initialize validation/test generator
        if evaluate:
            val_shape_type_gen = NeighborPairTraintest.generator_fn(
                self.traintest_file,
                'test_test',
                batch_size=self.batch_size,
                replace_nan=self.replace_nan,
                sharedx=self.sharedx,
                shuffle=False)
            self.val_shapes = val_shape_type_gen[0]
            self.val_gen = val_shape_type_gen[2]()
            self.validation_steps = np.ceil(
                self.val_shapes[0][0] / self.batch_size)
        else:
            self.val_shapes = None
            self.val_gen = None
            self.validation_steps = None

        # log parameters
        self.__log.info("**** %s Parameters: ***" % self.__class__.__name__)
        self.__log.info("{:<22}: {:>12}".format("model_dir", self.model_dir))
        if self.traintest_file is not None:
            self.__log.info("{:<22}: {:>12}".format(
                "traintest_file", self.traintest_file))
            tmp = NeighborPairTraintest(self.traintest_file, 'train_train')
            self.__log.info("{:<22}: {:>12}".format(
                'train_train', str(tmp.get_py_shapes())))
            if evaluate:
                tmp = NeighborPairTraintest(self.traintest_file, 'train_test')
                self.__log.info("{:<22}: {:>12}".format(
                    'train_test', str(tmp.get_py_shapes())))
                tmp = NeighborPairTraintest(self.traintest_file, 'test_test')
                self.__log.info("{:<22}: {:>12}".format(
                    'test_test', str(tmp.get_py_shapes())))
        self.__log.info("{:<22}: {:>12}".format(
            "learning_rate", self.learning_rate))
        self.__log.info("{:<22}: {:>12}".format(
            "epochs", self.epochs))
        self.__log.info("{:<22}: {:>12}".format(
            "output_dim", self.output_dim))
        self.__log.info("{:<22}: {:>12}".format(
            "batch_size", self.batch_size))
        self.__log.info("{:<22}: {:>12}".format(
            "layers", str(self.layers)))
        self.__log.info("{:<22}: {:>12}".format(
            "dropout", str(self.dropout)))
        self.__log.info("{:<22}: {:>12}".format(
            "augment_fn", str(self.augment_fn)))
        self.__log.info("{:<22}: {:>12}".format(
            "augment_scale", self.augment_scale))
        self.__log.info("{:<22}: {:>12}".format(
            "augment_kwargs", str(self.augment_kwargs)))
        self.__log.info("**** %s Parameters: ***" % self.__class__.__name__)

    def build_model(self, input_shape, load=False):
        """Compile Keras model

        input_shape(tuple): X dimensions (only nr feat is needed)
        load(bool): Whether to load the pretrained model.
        """
        def euclidean_distance(vects):
            x, y = vects
            sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
            return K.sqrt(K.maximum(sum_square, K.epsilon()))

        def dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)

        # we have two inputs
        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        # each goes to a network with the same architechture
        model_layers = list()
        # first layer
        model_layers.append(
            Dense(self.layers[0], activation='tanh', input_shape=input_shape))
        if self.dropout is not None:
            model_layers.append(Dropout(self.dropout))
        # other layers
        for layer in self.layers[1:-1]:
            model_layers.append(Dense(layer, activation='tanh'))
            if self.dropout is not None:
                model_layers.append(Dropout(self.dropout))
        # last layer
        model_layers.append(
            Dense(self.layers[-1], activation='tanh'))

        basenet = Sequential(model_layers)
        basenet.summary()

        encoded_a = basenet(input_a)
        encoded_b = basenet(input_b)

        # layer to merge two encoded inputs with distance between them
        distance = Lambda(euclidean_distance, output_shape=dist_output_shape)
        # call this layer on list of two input tensors.
        prediction = distance([encoded_a, encoded_b])
        model = Model([input_a, input_b], prediction)

        # define monitored metrics
        def accuracy(y_true, y_pred, threshold=0.5):
            y_pred = K.cast(y_pred < threshold, y_pred.dtype)
            return K.mean(K.equal(y_true, y_pred))

        metrics = [
            accuracy
        ]

        def contrastive_loss(y_true, y_pred):
            '''Contrastive loss from Hadsell-et-al.'06
            http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
            '''
            margin = 1
            square_pred = K.square(y_pred)
            margin_square = K.square(K.maximum(margin - y_pred, 0))
            return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

        # compile and print summary
        model.compile(
            optimizer=keras.optimizers.RMSprop(lr=self.learning_rate),
            loss=contrastive_loss,
            metrics=metrics)
        model.summary()

        # if pre-trained model is specified, load its weights
        self.model = model
        if load:
            self.model.load_weights(self.model_file)
        # this will be the encoder/transformer
        self.transformer = self.model.layers[2]

    def fit(self, monitor='val_accuracy'):
        """Fit the model.

        monitor(str): variable to monitor for early stopping.
        """
        # builf model
        input_shape = (self.tr_shapes[0][1],)
        self.build_model(input_shape)

        # prepare callbacks
        callbacks = list()

        def mask_keep(idxs, x1_data, x2_data, y_data):
            # we will fill an array of NaN with values we want to keep
            x1_data_transf = np.zeros_like(x1_data, dtype=np.float32) * np.nan
            for idx in idxs:
                # copy column from original data
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x1_data_transf[:, col_slice] = x1_data[:, col_slice]
            x2_data_transf = np.zeros_like(x2_data, dtype=np.float32) * np.nan
            for idx in idxs:
                # copy column from original data
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x2_data_transf[:, col_slice] = x2_data[:, col_slice]
            # keep rows containing at least one not-NaN value
            not_nan1 = np.isfinite(x1_data_transf).any(axis=1)
            not_nan2 = np.isfinite(x2_data_transf).any(axis=1)
            not_nan = np.logical_and(not_nan1, not_nan2)
            x1_data_transf = x1_data_transf[not_nan]
            x2_data_transf = x2_data_transf[not_nan]
            y_data_transf = y_data[not_nan]
            return x1_data_transf, x2_data_transf, y_data_transf

        def mask_exclude(idxs, x1_data, x2_data, y_data):
            x1_data_transf = np.copy(x1_data)
            for idx in idxs:
                # set current space to nan
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x1_data_transf[:, col_slice] = np.nan
            x2_data_transf = np.copy(x2_data)
            for idx in idxs:
                # set current space to nan
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x2_data_transf[:, col_slice] = np.nan
            # drop rows that only contain NaNs
            not_nan1 = np.isfinite(x1_data_transf).any(axis=1)
            not_nan2 = np.isfinite(x2_data_transf).any(axis=1)
            not_nan = np.logical_and(not_nan1, not_nan2)
            x1_data_transf = x1_data_transf[not_nan]
            x2_data_transf = x2_data_transf[not_nan]
            y_data_transf = y_data[not_nan]
            return x1_data_transf, x2_data_transf, y_data_transf

        # additional validation sets
        space_idx = self.augment_kwargs['dataset_idx']
        mask_fns = {
            'ALL': None,
            'NOT-SELF': partial(mask_exclude, space_idx),
            'ONLY-SELF': partial(mask_keep, space_idx),
        }
        validation_sets = list()
        if self.evaluate:
            vsets = ['train_test', 'test_test']
            for split in vsets:
                for set_name, mask_fn in mask_fns.items():
                    name = '_'.join([split, set_name])
                    shapes, dtypes, gen = NeighborPairTraintest.generator_fn(
                        self.traintest_file, split,
                        batch_size=self.batch_size,
                        replace_nan=self.replace_nan,
                        mask_fn=mask_fn,
                        sharedx=self.sharedx,
                        shuffle=False)
                    validation_sets.append((gen, shapes, name))
            additional_vals = AdditionalValidationSets(
                validation_sets, self.model, batch_size=self.batch_size)
            callbacks.append(additional_vals)

        patience = 10
        early_stopping = EarlyStopping(
            monitor=monitor,
            verbose=1,
            patience=patience,
            mode='max',
            restore_best_weights=True)
        if monitor or not self.evaluate:
            callbacks.append(early_stopping)

        # call fit and save model
        t0 = time()
        self.history = self.model.fit_generator(
            generator=self.tr_gen,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            callbacks=callbacks,
            validation_data=self.val_gen,
            validation_steps=self.validation_steps)
        self.time = time() - t0
        self.model.save(self.model_file)
        if self.evaluate:
            self.history.history.update(additional_vals.history)

        # check early stopping
        if early_stopping.stopped_epoch != 0:
            self.last_epoch = early_stopping.stopped_epoch - patience
        else:
            self.last_epoch = self.epochs

        # save and plot history
        history_file = os.path.join(
            self.model_dir, "%s_history.pkl" % self.name)
        pickle.dump(self.history.history, open(history_file, 'wb'))
        plot_file = os.path.join(self.model_dir, "%s.png" % self.name)
        self._plot_history(self.history.history, vsets, plot_file)

    def set_predict_scaler(self, scaler):
        self.scaler = scaler

    def predict(self, input_mat):
        """Do predictions.

        prediction_file(str): Path to input file containing Xs.
        split(str): which split to predict.
        batch_size(int): batch size for prediction.
        """
        # load model if not alredy there
        if self.model is None:
            self.build_model((input_mat.shape[1],), load=True)
        no_nans = np.nan_to_num(input_mat)
        if hasattr(self, 'scaler'):
            scaled = self.scaler.fit_transform(no_nans)
        else:
            scaled = no_nans
        return self.transformer.predict(scaled)

    def _plot_history(self, history, vsets, destination):
        """Plot history.

        history(dict): history result from Keras fit method.
        destination(str): path to output file.
        """
        import matplotlib.pyplot as plt

        metrics = list({k.split('_')[-1] for k in history})

        rows = len(metrics)
        cols = len(vsets)

        plt.figure(figsize=(cols * 5, rows * 5), dpi=100)

        c = 1
        for metric in sorted(metrics):
            for vset in vsets:
                plt.subplot(rows, cols, c)
                plt.title(metric.capitalize())
                plt.plot(history[metric], label="Train", lw=2, ls='--')
                plt.plot(history['val_' + metric], label="Val", lw=2, ls='--')
                vset_met = [k for k in history if vset in k and metric in k]
                for valset in vset_met:
                    plt.plot(history[valset], label=valset, lw=2)
                plt.ylim(0, 1)
                plt.legend()
                c += 1

        plt.tight_layout()

        if destination is not None:
            plt.savefig(destination)
        plt.close('all')


class AdditionalValidationSets(Callback):

    def __init__(self, validation_sets, model, verbose=1, batch_size=None):
        """
        validation_sets(list): list of 3-tuples (val_data, val_targets,
        val_set_name) or 4-tuples (val_data, val_targets, sample_weights,
        val_set_name).
        verbose(int): verbosity mode, 1 or 0.
        batch_size(int): batch size to be used when evaluating on the
        additional datasets.
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size
        self.model = model

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for val_gen, val_shapes, val_set_name in self.validation_sets:
            results = self.model.evaluate_generator(
                val_gen(),
                steps=np.ceil(val_shapes[0][0] / self.batch_size),
                verbose=self.verbose)

            for i, result in enumerate(results):
                name = '_'.join([val_set_name, self.model.metrics_names[i]])
                self.history.setdefault(name, []).append(result)
