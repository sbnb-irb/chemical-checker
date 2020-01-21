from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import os
import h5py
from time import time
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Lambda
from keras.optimizers import RMSprop


from keras import backend as K


from chemicalchecker.util import logged
from chemicalchecker.util.splitter import PairTraintest


@logged
class Siamese(object):
    """Siamese class"""

    def __init__(self, model_dir, traintest_file, evaluate, **kwargs):
        """Initialize the AutoEncoder class

        Args:
            model_dir(str): Directorty where models will be stored.
            batch_size(int): The batch size for the NN (default=128)
            epochs(int): The number of epochs (default: 200)
        """

        self.epochs = int(kwargs.get("epochs", 5))
        self.batch_size = int(kwargs.get("batch_size", 100))
        self.learning_rate = float(kwargs.get("learning_rate", 0.001))
        self.replace_nan = float(kwargs.get("replace_nan", 0.0))
        self.augment_fn = kwargs.get("augment_fn", None)
        self.augment_kwargs = kwargs.get("augment_kwargs", None)
        self.augment_scale = int(kwargs.get("augment_scale", 1))
        self.suffix = str(kwargs.get("suffix", 'eval'))
        self.name = 'siamese_%s' % self.suffix
        self.time = 0

        self.traintest_file = os.path.abspath(traintest_file)
        if not os.path.exists(traintest_file):
            raise Exception('Data path not exists!')
        self.model_dir = os.path.abspath(model_dir)
        if not os.path.exists(model_dir):
            self.__log.warning(
                "Specified models directory does not exist: %s",
                self.model_dir)
            os.mkdir(self.model_dir)

        tr_shapes, tr_dtypes, tr_gen = PairTraintest.generator_fn(
            self.traintest_file,
            'train_train',
            batch_size=int(self.batch_size / self.augment_scale),
            replace_nan=self.replace_nan,
            augment_fn=self.augment_fn,
            augment_kwargs=self.augment_kwargs,
            augment_scale=self.augment_scale)
        self.tr_shapes = tr_shapes
        self.tr_gen = tr_gen

        if evaluate:
            val_shapes, val_dtypes, val_gen = PairTraintest.generator_fn(
                self.traintest_file,
                'train_test',
                batch_size=self.batch_size,
                replace_nan=self.replace_nan)
        else:
            val_shapes, val_dtypes, val_gen = PairTraintest.generator_fn(
                self.traintest_file,
                'train_train',
                batch_size=self.batch_size,
                replace_nan=self.replace_nan)
        self.val_shapes = val_shapes
        self.val_gen = val_gen

        self.siamese_model_file = os.path.join(self.model_dir, "siamese.h5")
        self.siamese = None
        self.transformer = None

    def build_model(self, input_shape, load=False):
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
            x = Dense(512, activation='relu')(x)  # 512
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

    def fit(self):

        input_shape = (self.tr_shapes[0][1],)
        self.build_model(input_shape)

        t0 = time()
        history = self.siamese.fit_generator(
            generator=self.tr_gen(),
            steps_per_epoch=np.ceil(self.tr_shapes[0][0] / self.batch_size),
            epochs=self.epochs,
            validation_data=self.val_gen(),
            validation_steps=np.ceil(self.val_shapes[0][0] / self.batch_size))

        self.history = history

        self.siamese.save(self.siamese_model_file)
        self.time = time() - t0
    """
    def save_performances(self, path, suffix):
        trte, tete = self.evaluate()
        perf_file = os.path.join(path, "siamese_%s.pkl" % suffix)

        df = pd.DataFrame(columns=['algo', 'split', 'time', 'epochs',
            'batch_size', 'learning_rate', 'replace_nan', 'augment_fn',
            'augment_scale', 'augment_kwargs'])
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
        row.update('split': 'train_test', 'accuracy': trte)
        df.iloc[len(df)] = pd.Series(row)
        row.update('split': 'test_test', 'accuracy': tete)
        df.iloc[len(df)] = pd.Series(row)
        df.to_pickle(perf_file)

        plot_file = os.path.join(path, "siamese_%s.png" % suffix)
        self._plot_history(self.history, plot_file)"""

    def evaluate(self):

        def specific_eval(split):
            shapes, dtypes, gen = PairTraintest.generator_fn(self.traintest_file, split, 
                    batch_size=100, replace_nan=self.replace_nan, augment_scale=1)

            y_loss, y_acc = self.siamese.evaluate_generator(gen(), steps=shapes[0][1]//100 ,max_queue_size=1, verbose=1)

            self.__log.debug("Accuracy %s: %f" % (split, y_acc))
            return y_acc

        input_shape = (self.tr_shapes[0][1],)

        self.build_model(input_shape, load=True)

        acc_tr_te = specific_eval('train_test')

        acc_te_te = specific_eval('test_test')

        return acc_tr_te, acc_te_te

    def predict(self, traintest_file, dest_file, chunk_size=1000, input_dataset='V'):
        """Take data .h5 and produce an encoded data.

        Args:
            traintest_file(string): a path to input .h5 file.
            dest_file(string): a path to output .h5 file.
            chunk_size(int): numbe rof inputs at each prediction.
            dataset(string): The name of the dataset in the .h5 file to encode(default: 'V')
        """

        self.siamese = load_model(self.siamese_model_file)
        self.transformer = self.siamese.layers[2]

        with h5py.File(dest_file, "w") as results, h5py.File(traintest_file, 'r') as hf:
            input_size = hf[input_dataset].shape[0]
            if "keys" in hf.keys():
                results.create_dataset('keys', data=hf["keys"][
                                       :], maxshape=hf["keys"].shape)
            results.create_dataset(
                'V', (input_size, self.transformer.output_shape[1]),
                dtype=np.float32,
                maxshape=(input_size, self.transformer.output_shape[1]))

            for i in range(0, input_size, chunk_size):
                chunk = slice(i, i + chunk_size)
                results['V'][chunk] = self.transformer.predict(
                    hf[input_dataset][chunk])

    def _plot_history(self, history, destination=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4), dpi=600)

        plt.subplot(2, 2, 1)
        plt.title('Train loss evolution')
        plt.plot(history.history["loss"],
                 label="Train loss", lw=1, color="red")

        plt.subplot(2, 2, 2)
        plt.title('Train accuracy evolution')
        plt.plot(history.history["accuracy"],
                 label="Train accuracy", lw=1, color="red")
        plt.ylim(0, 1)

        plt.subplot(2, 2, 3)
        plt.title('Val loss evolution')
        plt.plot(history.history["val_loss"],
                 label="Val loss", lw=1, color="green")

        plt.subplot(2, 2, 4)
        plt.title('Val accuracy evolution')
        plt.plot(history.history["val_accuracy"],
                 label="Val accuracy", lw=1, color="green")
        plt.ylim(0, 1)

        plt.legend(loc='best')
        if destination is not None:
            plt.savefig(destination)

    @staticmethod
    def predict_online(h5_file, split,
                       mask_fn=None, batch_size=10000, limit=None,
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
        self.siamese = load_model(self.siamese_model_file)
        predict_fn = self.siamese.layers[2]

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
        if limit is None:
            limit = y_shape[0]
        if mask_fn is None:
            def mask_fn(x, y):
                return x, y
        for x_data, y_data in fn():
            x_m, y_m = mask_fn(x_data, y_data)
            if x_m.shape[0] == 0:
                continue
            y_m_pred = predict_fn(x_m)
            y_true[last_idx:last_idx + len(y_m)] = y_m
            y_pred[last_idx:last_idx + len(y_m)] = y_m_pred
            last_idx += len(y_m)
            if last_idx >= limit:
                break
        # we might not reach the limit
        if last_idx < limit:
            limit = last_idx
        return y_pred[:limit], y_true[:limit]