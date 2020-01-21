from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import os
import h5py
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

    def fit(self):
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

        input_shape = (self.tr_shapes[0][1],)

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
        history = model.fit_generator(
            generator=self.tr_gen(),
            steps_per_epoch=np.ceil(self.tr_shapes[0][0] / self.batch_size),
            epochs=self.epochs,
            validation_data=self.val_gen(),
            validation_steps=np.ceil(self.val_shapes[0][0] / self.batch_size))

        model.save(self.siamese_model_file)

        self._plot_history(history, os.path.join(
            self.model_dir, "siamese_validation_plot.png"))

    def evaluate(self):
        def compute_accuracy(generator, y_true):
            y_pred = self.siamese.predict_generator(generator)
            pred = y_pred.ravel() < 0.5
            y_true = p_obj.get_all_y()
            print("y_pred", y_pred.shape)
            return np.mean(pred == y_true)

        def get_y(data_path, split):
            p_obj = PairTraintest(data_path, split=split)
            p_obj.open()
            y_true = p_obj.get_all_y().flatten()
            p_obj.close()
            return y_true

        def specific_eval(split):
            shapes, dtypes, gen = PairTraintest.generator_fn(self.traintest_file, split, 
                    batch_size=100, replace_nan=self.replace_nan, augment_scale=1)

            y_true = get_y(data_path, split)

            acc_tr_te = compute_accuracy(gen(), y_true)

            self.__log.debug("Accuracy %s: %f" % (split, acc_tr_te))



        self.siamese = load_model(self.siamese_model_file, compile=False)

        specific_eval('test_test')

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
