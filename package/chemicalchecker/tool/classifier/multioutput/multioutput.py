from __future__ import absolute_import
from __future__ import print_function

import os
import json
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from functools import partial
from scipy.stats import rankdata
from keras.backend import int_shape
import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout

from chemicalchecker.util import logged
from chemicalchecker.util.splitter import Traintest


@logged
class Multioutput(object):
    """Multioutput class"""

    def __init__(self, model_dir, traintest_file=None, evaluate=False, **kwargs):
        """Initialize the Multioutput class

        Args:
            model_dir(str): Directorty where models will be stored.
            batch_size(int): The batch size for the NN (default=128)
            epochs(int): The number of epochs (default: 200)
        """

        self.epochs = int(kwargs.get("epochs", 10))
        self.batch_size = int(kwargs.get("batch_size", 100))
        self.learning_rate = float(kwargs.get("learning_rate", 1e-3))
        self.replace_nan = float(kwargs.get("replace_nan", 0.0))
        self.dropout = float(kwargs.get("dropout", 0.2))
        self.suffix = str(kwargs.get("suffix", 'eval'))
        self.split = str(kwargs.get("split", 'train'))
        self.layers = kwargs.get("layers", [128])
        self.pretrained_model = kwargs.get("pretrained_model", None)
        self.name = 'multioutput_%s' % self.suffix
        self.time = 0
        self.output_dim = None

        self.model_dir = os.path.abspath(model_dir)
        if not os.path.exists(model_dir):
            self.__log.warning(
                "Specified models directory does not exist: %s",
                self.model_dir)
            os.mkdir(self.model_dir)

        self.traintest_file = traintest_file
        if self.traintest_file is not None:
            self.traintest_file = os.path.abspath(traintest_file)
            if not os.path.exists(traintest_file):
                raise Exception('Data path not exists!')

        tr_shape_type_gen = Traintest.generator_fn(
            self.traintest_file,
            self.split,
            batch_size=self.batch_size)
        self.tr_shapes = tr_shape_type_gen[0]
        self.tr_gen = tr_shape_type_gen[2]
        self.output_dim = tr_shape_type_gen[0][1][1]

        if evaluate:
            val_shape_type_gen = Traintest.generator_fn(
                self.traintest_file,
                'test',
                batch_size=self.batch_size)
        else:
            val_shape_type_gen = Traintest.generator_fn(
                self.traintest_file,
                self.split,
                batch_size=self.batch_size)
        self.val_shapes = val_shape_type_gen[0]
        self.val_gen = val_shape_type_gen[2]

        self.model_file = os.path.join(self.model_dir, "%s.h5" % self.name)
        self.model = None

        self.__log.info("**** Multioutput Parameters: ***")
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
        self.__log.info("{:<22}: {:>12}".format("batch_size", self.batch_size))

        self.__log.info("{:<22}: {:>12}".format(
            "layers", str(self.layers)))
        self.__log.info("{:<22}: {:>12}".format(
            "dropout", str(self.dropout)))
        self.__log.info("{:<22}: {:>12}".format(
            "pretrained_model", str(self.pretrained_model)))

        self.__log.info("**** Imbalanced Parameters: ***")

    def build_model(self, input_shape=None, load=False):
        metrics = [
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.TopKCategoricalAccuracy(k=10, name='top10')
        ]

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

        model.compile(
            optimizer=keras.optimizers.Adam(lr=self.learning_rate),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=metrics)

        model.summary()

        self.model = model
        if self.pretrained_model is not None:
            self.model.load_weights(self.pretrained_model)
        if load:
            self.model.load_weights(self.model_file)

    def fit(self):
        input_shape = (self.tr_shapes[0][1],)
        self.build_model(input_shape)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            verbose=1,
            patience=4,
            mode='max',
            restore_best_weights=True)

        t0 = time()
        self.history = self.model.fit_generator(
            generator=self.tr_gen(),
            steps_per_epoch=np.ceil(self.tr_shapes[0][0] / self.batch_size),
            epochs=self.epochs,
            callbacks=[early_stopping],
            validation_data=self.val_gen(),
            validation_steps=np.ceil(self.val_shapes[0][0] / self.batch_size),
            shuffle=True)

        self.model.save(self.model_file)
        self.time = time() - t0

        '''
        hist_log = os.path.join(self.model_dir,'history_%s.json' % self.name)
        with open(hist_log, 'w') as file:
            json.dump(self.history.history, file)
        '''

        plot_file = os.path.join(self.model_dir, "%s.png" % self.name)
        self._plot_history(self.history, plot_file)

    def evaluate(self, eval_set, splits=['train_test', 'test_test'],
                 mask_fn=None):
        def specific_eval(split, b_size=self.batch_size, mask_fn=None):
            shapes, dtypes, gen = Traintest.generator_fn(
                self.traintest_file, split,
                batch_size=b_size,
                mask_fn=mask_fn)

            y_loss, y_acc = self.model.evaluate_generator(
                gen(), steps=shapes[0][0] // b_size,
                max_queue_size=1, verbose=1)

            self.__log.debug("Accuracy %s %s: %f" % (eval_set, split, y_acc))
            return {'accuracy': y_acc}

        input_shape = (self.tr_shapes[0][1],)

        self.build_model(input_shape, load=True)

        results = dict()
        for split in splits:
            results[split] = specific_eval(split, mask_fn=mask_fn)

        return results

    def _predict(self, input_mat):
        if self.model is None:
            self.build_model((input_mat.shape[1],), load=True)
        no_nans = np.nan_to_num(input_mat)
        return self.model.predict(no_nans)

    def predict(self, prediction_file, split='train', batch_size=1000):
        ptt = Traintest(prediction_file, split)
        x_shape = ptt.get_x_shapes()
        if self.model is None:
            self.build_model((x_shape[1],), load=True)
        shapes, dtypes, gen = Traintest.generator_fn(
            prediction_file, split,
            batch_size=batch_size, only_x=True)
        res = self.model.predict_generator(gen(),
                                           steps=np.ceil(shapes[0] / 1000))
        return res[:shapes[0]]

    def _plot_history(self, history, destination=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4), dpi=600)

        plt.subplot(1, 4, 1)
        plt.title('Loss: categorical_crossentropy')
        plt.plot(history.history["loss"],
                 label="Train", lw=1, ls='--', color="red")
        plt.plot(history.history["val_loss"],
                 label="Val", lw=1, color="red")
        plt.ylim(0, 1)

        plt.subplot(1, 4, 2)
        plt.title('Accuracy')
        plt.plot(history.history["accuracy"],
                 label="Train", lw=1, ls='--', color="blue")
        plt.plot(history.history["val_accuracy"],
                 label="Val", lw=1, color="blue")
        plt.ylim(0, 1)

        plt.subplot(1, 4, 3)
        plt.title('AUC')
        plt.plot(history.history["auc"],
                 label="Train", lw=1, ls='--', color="green")
        plt.plot(history.history["val_auc"],
                 label="Val", lw=1, color="green")
        plt.ylim(0, 1)

        plt.subplot(1, 4, 4)
        plt.title('TOP10')
        plt.plot(history.history["top10"],
                 label="Train", lw=1, ls='--', color="orange")
        plt.plot(history.history["val_top10"],
                 label="Val", lw=1, color="orange")
        plt.ylim(0, 1)

        plt.tight_layout()

        if destination is not None:
            plt.savefig(destination)
        plt.close('all')

    def _plot_eval(self, df):
        pass
