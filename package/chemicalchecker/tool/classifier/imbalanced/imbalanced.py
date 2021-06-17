from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from functools import partial
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout

from chemicalchecker.util import logged
from chemicalchecker.util.splitter import PairTraintest


@logged
class Imbalanced(object):
    """Imbalanced class"""

    def __init__(self, model_dir, traintest_file=None, evaluate=False, **kwargs):
        """Initialize the Imbalanced class

        Args:
            model_dir(str): Directorty where models will be stored.
            batch_size(int): The batch size for the NN (default=128)
            epochs(int): The number of epochs (default: 200)
        """

        self.epochs = int(kwargs.get("epochs", 10))
        self.batch_size = int(kwargs.get("batch_size", 1000))
        self.learning_rate = float(kwargs.get("learning_rate", 1e-3))
        self.replace_nan = float(kwargs.get("replace_nan", 0.0))
        self.dropout = float(kwargs.get("dropout", 0.2))
        self.suffix = str(kwargs.get("suffix", 'eval'))
        self.layers = kwargs.get("layers", [128])
        self.name = 'imbalanced_%s' % self.suffix
        self.time = 0
        self.class_weight = None
        self.output_bias = None

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

            # class weigths
            tmp = PairTraintest(self.traintest_file, 'train')
            tmp.open()
            pos, neg = tmp.get_pos_neg()
            tmp.close()
            total = pos + neg
            weight_for_0 = (1 / neg) * (total) / 2.0
            weight_for_1 = (1 / pos) * (total) / 2.0
            class_weight = {0: weight_for_0, 1: weight_for_1}
            self.class_weight = class_weight
            self.output_bias = np.log([pos / neg])

            tr_shape_type_gen = PairTraintest.generator_fn(
                self.traintest_file,
                'train',
                batch_size=self.batch_size)
            self.tr_shapes = tr_shape_type_gen[0]
            self.tr_gen = tr_shape_type_gen[2]

            if evaluate:
                val_shape_type_gen = PairTraintest.generator_fn(
                    self.traintest_file,
                    'test',
                    batch_size=self.batch_size)
            else:
                val_shape_type_gen = tr_shape_type_gen
            self.val_shapes = val_shape_type_gen[0]
            self.val_gen = val_shape_type_gen[2]

        self.model_file = os.path.join(self.model_dir, "imbalanced.h5")
        self.model = None

        self.__log.info("**** Imbalanced Parameters: ***")
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
        self.__log.info("{:<22}: {:>12}".format("batch_size", self.batch_size))
        self.__log.info("{:<22}: {:>12}".format(
            "class_weight", str(self.class_weight)))
        self.__log.info("{:<22}: {:>12}".format(
            "output_bias", str(self.output_bias)))
        self.__log.info("{:<22}: {:>12}".format(
            "layers", str(self.layers)))
        self.__log.info("{:<22}: {:>12}".format(
            "dropout", str(self.dropout)))

        self.__log.info("**** Imbalanced Parameters: ***")

    def build_model(self, input_shape=None, load=False):
        def mcc(y_true, y_pred):
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
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.AUC(name='auc'),
            mcc,
        ]

        if self.output_bias is not None:
            output_bias = tf.keras.initializers.Constant(self.output_bias)
        else:
            output_bias = tf.keras.initializers.Constant(0.0)

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
            Dense(1, activation='sigmoid', bias_initializer=output_bias))

        model = keras.Sequential(model_layers)

        model.compile(
            optimizer=keras.optimizers.Adam(lr=self.learning_rate),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics)

        model.summary()

        self.model = model
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
            class_weight=self.class_weight,
            shuffle=True)

        self.model.save(self.model_file)
        self.time = time() - t0

        plot_file = os.path.join(self.model_dir, "%s.png" % self.name)
        self._plot_history(self.history, plot_file)

    def save_performances(self, path, suffix,
                          splits=['train', 'test']):

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

        all_acc = self.evaluate('ALL')

        space_idx = self.augment_kwargs['dataset_idx']
        excl_fn = partial(mask_exclude, space_idx)
        excl_acc = self.evaluate('NOT-SELF', mask_fn=excl_fn)

        keep_fn = partial(mask_keep, space_idx)
        keep_acc = self.evaluate('ONLY-SELF', mask_fn=keep_fn)

        perf_file = os.path.join(path, "siamese_%s.pkl" % suffix)
        df = pd.DataFrame(columns=['algo', 'split', 'time', 'epochs',
                                   'batch_size', 'learning_rate',
                                   'replace_nan', 'augment_fn',
                                   'augment_scale', 'augment_kwargs',
                                   'eval_set', 'accuracy'])
        row = {
            'algo': self.name,
            'time': self.time,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'replace_nan': self.replace_nan,
            'augment_fn': str(self.augment_fn),
            'augment_kwargs': str(self.augment_kwargs),
            'augment_scale': self.augment_scale,
        }
        for split in splits:
            row.update({'eval_set': 'ALL', 'split': split,
                        'accuracy': all_acc[split]['accuracy']})
            df.loc[len(df)] = pd.Series(row)
            row.update({'eval_set': 'NOT-SELF', 'split': split,
                        'accuracy': excl_acc[split]['accuracy']})
            df.loc[len(df)] = pd.Series(row)
            row.update({'eval_set': 'ONLY-SELF', 'split': split,
                        'accuracy': keep_acc[split]['accuracy']})
            df.loc[len(df)] = pd.Series(row)
        df.to_pickle(perf_file)
        df.to_csv(perf_file.replace('.pkl', '.csv'), index=False)
        self._plot_eval(df)

    def evaluate(self, eval_set, splits=['train_test', 'test_test'],
                 mask_fn=None):
        def specific_eval(split, b_size=self.batch_size, mask_fn=None):
            shapes, dtypes, gen = PairTraintest.generator_fn(
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
        ptt = PairTraintest(prediction_file, split)
        x1_shape, x2_shape, y_shape = ptt.get_xy_shapes()
        if self.model is None:
            self.build_model((x1_shape[1] + x2_shape[1],), load=True)
        shapes, dtypes, gen = PairTraintest.generator_fn(
            prediction_file, split,
            batch_size=batch_size, only_x=True)
        res = self.model.predict_generator(gen(),
                                           steps=np.ceil(shapes[0][0] / 1000))
        return res[:shapes[0][0]]

    def _plot_history(self, history, destination=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4), dpi=600)

        plt.subplot(1, 4, 1)
        plt.title('Loss: BinaryCrossEntropy')
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
        plt.title('MCC')
        plt.plot(history.history["mcc"],
                 label="Train", lw=1, ls='--', color="orange")
        plt.plot(history.history["val_mcc"],
                 label="Val", lw=1, color="orange")
        plt.ylim(0, 1)

        plt.tight_layout()

        if destination is not None:
            plt.savefig(destination)

    def _plot_eval(self, df):
        pass
