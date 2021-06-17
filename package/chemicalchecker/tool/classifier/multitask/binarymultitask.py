import os
import pickle
import numpy as np
from time import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout

from chemicalchecker.util import logged
from chemicalchecker.util.splitter import Traintest


@logged
class BinaryMultitask(object):
    """BinaryMultitask class.

    This class implements a simple neural network based on Keras that allows
    performing multitask binary classification.
    """

    def __init__(self, model_dir, traintest_file=None, evaluate=False, **kwargs):
        """Initialize the BinaryMultitask class.

        Args:
            model_dir(str): Directorty where models will be stored.
            traintest_file(str): Path to the traintest file.
            evaluate(bool): Whether to run evaluation.
        """

        # read parameters
        self.epochs = int(kwargs.get("epochs", 10))
        self.batch_size = int(kwargs.get("batch_size", 100))
        self.learning_rate = float(kwargs.get("learning_rate", 1e-3))
        self.replace_nan = float(kwargs.get("replace_nan", 0.0))
        self.dropout = float(kwargs.get("dropout", 0.2))
        self.suffix = str(kwargs.get("suffix", 'eval'))
        self.split = str(kwargs.get("split", 'train'))
        self.layers = kwargs.get("layers", [128])
        self.pretrained_model = kwargs.get("pretrained_model", None)
        self.sample_weights = kwargs.get("sample_weights", False)

        # internal variables
        self.name = '%s_%s' % (self.__class__.__name__.lower(), self.suffix)
        self.time = 0
        self.evaluate = evaluate
        self.output_dim = None
        self.model_dir = os.path.abspath(model_dir)
        self.model_file = os.path.join(self.model_dir, "%s.h5" % self.name)
        self.model = None

        # check output path
        if not os.path.exists(model_dir):
            self.__log.warning("Creating model directory: %s", self.model_dir)
            os.mkdir(self.model_dir)

        # check input path
        self.traintest_file = traintest_file
        if self.traintest_file is not None:
            self.traintest_file = os.path.abspath(traintest_file)
            if not os.path.exists(traintest_file):
                raise Exception('Input data file does not exists!')

        # initialize train generator
        tr_shape_type_gen = Traintest.generator_fn(
            self.traintest_file,
            self.split,
            batch_size=self.batch_size,
            sample_weights=self.sample_weights)
        self.tr_shapes = tr_shape_type_gen[0]
        self.tr_gen = tr_shape_type_gen[2]()
        self.steps_per_epoch = np.ceil(self.tr_shapes[0][0] / self.batch_size)
        self.output_dim = tr_shape_type_gen[0][1][1]

        # initialize validation/test generator
        if evaluate:
            val_shape_type_gen = Traintest.generator_fn(
                self.traintest_file,
                'test',
                batch_size=self.batch_size,
                sample_weights=self.sample_weights,
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
            self.__log.info("{:<22}: {:>12}".format(
                'train', str(self.tr_shapes)))
            if evaluate:
                self.__log.info("{:<22}: {:>12}".format(
                    'test', str(self.val_shapes)))
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
            "pretrained_model", str(self.pretrained_model)))
        self.__log.info("**** %s Parameters: ***" % self.__class__.__name__)

    def build_model(self, input_shape, load=False):
        """Compile Keras model

        input_shape(tuple): X dimensions (only nr feat is needed)
        load(bool): Whether to load the pretrained model.
        """
        model_layers = list()
        # first layer
        model_layers.append(
            Dense(self.layers[0], activation='relu', input_shape=input_shape))
        if self.dropout is not None:
            model_layers.append(Dropout(self.dropout))
        # other layers
        for layer in self.layers[1:]:
            model_layers.append(Dense(layer, activation='relu'))
            if self.dropout is not None:
                model_layers.append(Dropout(self.dropout))
        # last layer
        model_layers.append(
            Dense(self.output_dim, activation='sigmoid'))
        model = keras.Sequential(model_layers)

        # define monitored metrics
        def mcc(y_true, y_pred):
            '''Flatten and compute Matthew's Corr. Coef.'''
            y_pred = tf.keras.backend.flatten(y_pred)
            y_true = tf.keras.backend.flatten(y_true)
            y_pred_pos = K.round(K.clip(y_pred, 0, 1))
            y_pred_neg = 1 - y_pred_pos
            y_pos = K.round(K.clip(y_true, 0, 1))
            y_neg = 1 - y_pos
            tp = K.sum(y_pos * y_pred_pos)
            tn = K.sum(y_neg * y_pred_neg)
            fp = K.sum(y_neg * y_pred_pos)
            fn = K.sum(y_pos * y_pred_neg)
            numerator = (tp * tn - fp * fn)
            denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            return numerator / (denominator + K.epsilon())

        metrics = [
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.AUC(name='auroc'),
            keras.metrics.AUC(curve='PR', name='auprc'),
            keras.metrics.TopKCategoricalAccuracy(k=10, name='top10'),
            mcc,
        ]

        # compile and print summary
        model.compile(
            optimizer=keras.optimizers.Adam(lr=self.learning_rate),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics)
        model.summary()

        # if pre-trained model is specified, load its weights
        self.model = model
        if self.pretrained_model is not None:
            self.model.load_weights(self.pretrained_model)
        if load:
            self.model.load_weights(self.model_file)

    def fit(self, monitor='val_top10', class_weight=None):
        """Fit the model.

        monitor(str): variable to monitor for early stopping.
        class_weight(list): list of weights for each class (Y dimension).
        """
        # builf model
        input_shape = (self.tr_shapes[0][1],)
        self.build_model(input_shape)

        # prepare callbacks
        patience = 4
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            verbose=1,
            patience=patience,
            mode='max',
            restore_best_weights=True)
        if monitor is None:
            callbacks = None
        else:
            callbacks = [early_stopping]

        # call fit and save model
        t0 = time()
        self.history = self.model.fit_generator(
            generator=self.tr_gen,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            callbacks=callbacks,
            validation_data=self.val_gen,
            class_weight=class_weight,
            validation_steps=self.validation_steps)
        self.time = time() - t0
        self.model.save(self.model_file)

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
        self._plot_history(self.history.history, plot_file)

    def predict(self, prediction_file, split='train', batch_size=100):
        """Do predictions.

        prediction_file(str): Path to input file containing Xs.
        split(str): which split to predict.
        batch_size(int): batch size for prediction.
        """
        # get input shape
        ptt = Traintest(prediction_file, split)
        x_shape = ptt.get_x_shapes()
        # load model if not alredy there
        if self.model is None:
            self.build_model((x_shape[1],), load=True)
        # predict with generator
        shapes, dtypes, gen = Traintest.generator_fn(
            prediction_file, split,
            batch_size=batch_size,
            only_x=True,
            shuffle=False)
        res = self.model.predict_generator(
            gen(), steps=np.ceil(shapes[0] / batch_size))
        return res[:shapes[0]]

    def _plot_history(self, history, destination):
        """Plot history.

        history(dict): history result from Keras fit method.
        destination(str): path to output file.
        """
        import matplotlib.pyplot as plt

        metrics = [k for k in history if not k.startswith('val_')]

        rows = np.ceil(len(metrics) / 2.)
        cols = 2

        colors = ['orangered', 'royalblue', 'forestgreen', 'darkmagenta',
                  'darkorange', 'olive'] * 10

        plt.figure(figsize=(cols * 5, rows * 5), dpi=100)

        for idx, metric in enumerate(metrics):
            plt.subplot(rows, cols, idx + 1)
            plt.title(metric.capitalize())
            plt.plot(history[metric],
                     label="Train", lw=2, ls='--', color=colors[idx])
            if self.evaluate:
                plt.plot(history["val_%s" % metric],
                         label="Val", lw=2, color=colors[idx])
            plt.ylim(0, 1)

        plt.tight_layout()

        if destination is not None:
            plt.savefig(destination)
        plt.close('all')