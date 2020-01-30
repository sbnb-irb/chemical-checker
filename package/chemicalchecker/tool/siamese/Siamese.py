from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from time import time
from functools import partial
from keras import backend as K
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, Dropout, Lambda
from keras.callbacks import EarlyStopping

from chemicalchecker.util import logged
from chemicalchecker.util.splitter import NeighborPairTraintest


@logged
class Siamese(object):
    """Siamese class"""

    def __init__(self, model_dir, traintest_file=None, evaluate=False, **kwargs):
        """Initialize the AutoEncoder class

        Args:
            model_dir(str): Directorty where models will be stored.
            batch_size(int): The batch size for the NN (default=128)
            epochs(int): The number of epochs (default: 200)
        """

        self.epochs = int(kwargs.get("epochs", 15))
        self.batch_size = int(kwargs.get("batch_size", 100))
        self.learning_rate = float(kwargs.get("learning_rate", 0.001))
        self.replace_nan = float(kwargs.get("replace_nan", 0.0))
        self.augment_fn = kwargs.get("augment_fn", None)
        self.augment_kwargs = kwargs.get("augment_kwargs", None)
        self.augment_scale = int(kwargs.get("augment_scale", 1))
        self.suffix = str(kwargs.get("suffix", 'eval'))
        self.name = 'siamese_%s' % self.suffix
        self.time = 0

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

            tr_shape_type_gen = NeighborPairTraintest.generator_fn(
                self.traintest_file,
                'train_train',
                batch_size=int(self.batch_size / self.augment_scale),
                replace_nan=self.replace_nan,
                augment_fn=self.augment_fn,
                augment_kwargs=self.augment_kwargs,
                augment_scale=self.augment_scale)
            self.tr_shapes = tr_shape_type_gen[0]
            self.tr_gen = tr_shape_type_gen[2]

            if evaluate:
                val_shape_type_gen = NeighborPairTraintest.generator_fn(
                    self.traintest_file,
                    'test_test',
                    batch_size=self.batch_size,
                    replace_nan=self.replace_nan)
            else:
                val_shape_type_gen = NeighborPairTraintest.generator_fn(
                    self.traintest_file,
                    'train_train',
                    batch_size=self.batch_size,
                    replace_nan=self.replace_nan)
            self.val_shapes = val_shape_type_gen[0]
            self.val_gen = val_shape_type_gen[2]

        self.siamese_model_file = os.path.join(self.model_dir, "siamese.h5")
        self.siamese = None
        self.transformer = None
        self.last_epoch = 0

        self.__log.info("**** Siamese Parameters: ***")
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
        self.__log.info("{:<22}: {:>12}".format("batch_size", self.batch_size))

        self.__log.info("{:<22}: {:>12}".format(
            "augment_fn", str(self.augment_fn)))
        self.__log.info("{:<22}: {:>12}".format(
            "augment_scale", self.augment_scale))
        self.__log.info("{:<22}: {:>12}".format(
            "augment_kwargs", str(self.augment_kwargs)))
        self.__log.info("{:<22}: {:>12}".format(
            "replace_nan", str(self.replace_nan)))
        self.__log.info("**** Siamese Parameters: ***")

    def build_model(self, input_shape=None, load=False):
        def euclidean_distance(vects):
            x, y = vects
            sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
            return K.sqrt(K.maximum(sum_square, K.epsilon()))

        def eucl_dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)

        def contrastive_loss(y_true, y_pred):
            '''Contrastive loss from Hadsell-et-al.'06
            http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
            '''
            margin = 1
            square_pred = K.square(y_pred)
            margin_square = K.square(K.maximum(margin - y_pred, 0))
            return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

        def create_base_network(input_shape):
            '''Create network architecture'''
            input = Input(shape=input_shape)
            x = Dense(1024, activation='relu')(input)  # 1024
            x = Dropout(0.1)(x)
            x = Dense(128, activation='relu')(x)
            return Model(input, x)

        def accuracy(y_true, y_pred):
            '''Compute classification accuracy with a fixed threshold on distances.
            '''
            return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

        base_network = create_base_network(input_shape)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)(
            [processed_a, processed_b])

        model = Model([input_a, input_b], distance)

        model.summary()

        rms = RMSprop()
        model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
        self.siamese = model
        if load:
            self.siamese.load_weights(self.siamese_model_file)

    def fit(self, final=False):
        input_shape = (self.tr_shapes[0][1],)
        self.build_model(input_shape)

        if not final:
            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                verbose=1,
                patience=5,
                mode='max',
                restore_best_weights=True)

            t0 = time()
            self.history = self.siamese.fit_generator(
                generator=self.tr_gen(),
                steps_per_epoch=np.ceil(self.tr_shapes[0][0] / self.batch_size),
                epochs=self.epochs,
                callbacks=[early_stopping],
                validation_data=self.val_gen(),
                validation_steps=np.ceil(self.val_shapes[0][0] / self.batch_size),
                shuffle=True,
                verbose=2)

            self.last_epoch = early_stopping.stopped_epoch

        else:
            t0 = time()
            self.history = self.siamese.fit_generator(
                generator=self.tr_gen(),
                steps_per_epoch=np.ceil(self.tr_shapes[0][0] / self.batch_size),
                epochs=self.epochs,
                validation_data=self.val_gen(),
                validation_steps=np.ceil(self.val_shapes[0][0] / self.batch_size),
                shuffle=True,
                verbose=2)

        self.siamese.save(self.siamese_model_file)
        self.time = time() - t0

        plot_file = os.path.join(self.model_dir, "%s.png" % self.name)
        self._plot_history(self.history, plot_file)

    def save_performances(self, path, suffix,
                          splits=['train_test', 'test_test']):

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

        plot_file = os.path.join(self.model_dir, "full_%s.png" % self.name)
        self._plot_eval(df, destination=plot_file)

    def evaluate(self, eval_set, splits=['train_test', 'test_test'],
                 mask_fn=None):
        def specific_eval(split, b_size=self.batch_size, mask_fn=None):
            shapes, dtypes, gen = NeighborPairTraintest.generator_fn(
                self.traintest_file, split,
                batch_size=b_size,
                replace_nan=self.replace_nan,
                mask_fn=mask_fn)
            print(shapes[0][0])
            print(b_size)
            y_loss, y_acc = self.siamese.evaluate_generator(
                gen(), steps=shapes[0][0] // 100,
                max_queue_size=1, verbose=0)

            self.__log.debug("Accuracy %s %s: %f" % (eval_set, split, y_acc))
            return {'accuracy': y_acc}

        input_shape = (self.tr_shapes[0][1],)

        self.build_model(input_shape, load=True)

        results = dict()
        for split in splits:
            results[split] = specific_eval(split, mask_fn=mask_fn)

        return results

    def predict(self, input_mat):
        if self.siamese is None:
            self.build_model((input_mat.shape[1],), load=True)
            self.transformer = self.siamese.layers[2]
        no_nans = np.nan_to_num(input_mat)
        return self.transformer.predict(no_nans)

    def _plot_history(self, history, destination=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4), dpi=600)

        plt.subplot(1, 2, 1)
        plt.title('Train loss evolution')
        plt.plot(history.history["loss"],
                 label="Train loss", lw=1, color="red")
        plt.plot(history.history["val_loss"],
                 label="Val loss", lw=1, color="green")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title('Train accuracy evolution')
        plt.plot(history.history["accuracy"],
                 label="Train accuracy", lw=1, color="red")
        plt.plot(history.history["val_accuracy"],
                 label="Val accuracy", lw=1, color="green")
        plt.ylim(0, 1)
        plt.legend()

        if destination is not None:
            plt.savefig(destination)

    def _plot_eval(self, df,destination=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        g = sns.catplot(x="eval_set", y="accuracy", hue="split", data=df,
                        height=6, kind="bar", palette="muted")
        g.set_ylabels("Accuracy", fontsize=15)
        plt.ylim(0,1)

        if destination is not None:
            plt.savefig(destination)
        
