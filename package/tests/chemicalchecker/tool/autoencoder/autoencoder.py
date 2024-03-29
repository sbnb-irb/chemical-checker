import os
import h5py
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Masking, Dropout, Activation
from tensorflow.keras.models import load_model
from chemicalchecker.util import logged
from chemicalchecker.util.splitter import Traintest
from chemicalchecker.core.signature_data import DataSignature

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class NanMaskingLayer(tf.keras.layers.Layer):

    def __init__(self, mask_value=0.0, **kwargs):
        super(NanMaskingLayer, self).__init__(**kwargs)
        self.mask_value = mask_value

    def call(self, input):
        nan_idxs = tf.is_nan(input)
        replace = tf.ones_like(input) * self.mask_value
        return tf.where(nan_idxs, replace, input)


@logged
class AutoEncoder:
    """AutoEncoder class"""

    def __init__(self, models_path, **kwargs):
        """Initialize the AutoEncoder class

        Args:
            models_path(str): Directorty where models will be stored.
            learning_rate(float): The learning rate (default=0.001)
            batch_size(int): The batch size for the NN (default=128)
            optimizer(string): The optimizer (default='adadelta').
            encoding_dim(int): The encoding dimension (default=512).
            loss(string): The loss function (default='mean_squared_error').
            activation(tf): The activation function (default=rf.nn.relu).
            activation_last(tf): The activation function of last layer(default=rf.nn.tanh).
            epochs(int): The number of epochs (default: 200)
            dropout_rate(float): The dropout rate (default: 0.2).
            shuffle(bool): Shuffle data (default=True).
            mask_value(float): The mask value in case we use masks.
                If default then there are no values to mask (default:None)
            cpu(int): The number of cores to use (default: 32)
        """
        # Models path
        self.models_path = os.path.abspath(models_path)
        if not os.path.exists(models_path):
            self.__log.warning(
                "Specified models directory does not exist: %s",
                self.models_path)
            os.mkdir(self.models_path)

        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.batch_size = int(kwargs.get("batch_size", 128))
        self.optimizer = kwargs.get("optimizer", 'adadelta')
        self.encoding_dim = kwargs.get("encoding_dim", 512)
        self.loss = kwargs.get("loss", 'mean_squared_error')
        self.activation = kwargs.get("activation", tf.nn.tanh)
        self.activation_last = kwargs.get("activation_last", tf.nn.tanh)
        self.epochs = int(kwargs.get("epochs", 200))
        self.dropout_rate = float(kwargs.get("dropout_rate", 0.2))
        self.shuffle = kwargs.get("shuffle", True)
        self.gpu = kwargs.get("gpu", False)
        self.mask_value = kwargs.get("mask_value", None)
        self.cpu = kwargs.get("cpu", 32)
        self.shuffles = 10
        self.random_seed = 42

        self.autoencoder_model_file = os.path.join(
            self.models_path, "autoencoder.h5")

        self.encoder = None
        self.decoder = None

    def fit(self, data_path):
        """Take data .h5 and learn an encoder.

        Args:
            data_path(string): a path to .h5 file.
        """
        self.data_path = data_path

        if self.gpu:
            print ("Running on GPUs")
            config = tf.ConfigProto(device_count={'GPU': 1})

        else:

            config = tf.ConfigProto(intra_op_parallelism_threads=self.cpu,
                                    inter_op_parallelism_threads=self.cpu,
                                    allow_soft_placement=True,
                                    device_count={'CPU': self.cpu})
        session = tf.Session(config=config)
        K.set_session(session)

        if not os.path.isfile(self.data_path) or self.data_path[-3:] != '.h5':
            raise Exception("Input data needs to be a H5 file")

        self.traintest_file = os.path.join(self.models_path, 'traintest.h5')

        with h5py.File(data_path, 'r') as hf:
            if 'x' not in hf.keys():
                raise Exception(
                    "Input data file needs to have a dataset called 'x'")

        if not os.path.isfile(self.traintest_file):
            Traintest.split_h5_blocks(self.data_path, self.traintest_file, split_names=[
                'train', 'test'], split_fractions=[.8, .2], datasets=['x'])

            with h5py.File(self.traintest_file, 'r+') as hf:
                x_ds = 'x'
                y_ds = 'y'
                hf["y_train"] = h5py.SoftLink('/x_train')
                hf["y_test"] = h5py.SoftLink('/x_test')

        with h5py.File(self.traintest_file, 'r') as hf:
            x_ds = 'x'
            y_ds = 'y'

            if 'x_train' in hf.keys():
                x_ds = 'x_train'
                y_ds = 'y_train'
            self.input_dimension = hf[x_ds].shape[1]

            self.train_size = hf[x_ds].shape[0]
            self.test_size = hf["x_test"].shape[0]
            self.total_size = 0
            for split in [i for i in hf.keys() if i.startswith('x')]:
                self.total_size += hf[split].shape[0]

        input_dim = Input(shape=(self.input_dimension, ))
        first_layer = input_dim

        if self.mask_value is not None:
            self.loss = self.filtered_loss_function
            input_dim = NanMaskingLayer()(input_dim)

        if self.input_dimension <= self.encoding_dim:
            raise Exception("Input dimension smaller than desired reduction")

        self.num_middle_layers = 1

        # Calculate the reduction rate and then divide by two.
        # The number gives us the number of middle layers including the latent
        # space
        reduc_rate = np.floor(self.input_dimension / self.encoding_dim)

        self.num_middle_layers = np.ceil(reduc_rate / 2)

        last_layer = input_dim

        layer_sizes = np.linspace(
            self.input_dimension, self.encoding_dim, self.num_middle_layers + 1)[1:]

        self.__log.debug("Num of layers: %d" % len(layer_sizes))

        for layer_size in layer_sizes:
            last_layer = Dense(
                int(layer_size), activation=self.activation)(last_layer)

        self.encoder = last_layer

        for layer_size in np.flip(layer_sizes, 0):
            if layer_size == self.encoding_dim:
                continue
            last_layer = Dense(
                int(layer_size), activation=self.activation)(last_layer)

        self.decoder = Dense(self.input_dimension,
                             activation=self.activation_last)(last_layer)

        self.autoencoder_model = Model(
            inputs=first_layer, outputs=self.decoder)
        self.autoencoder_model.compile(
            optimizer=self.optimizer, loss=self.loss)

        self.autoencoder_model.summary()

        history = self.autoencoder_model.fit(self.generator_fn("train"), epochs=self.epochs,
                                             shuffle=self.shuffle, steps_per_epoch=self.train_size / self.batch_size,
                                             validation_data=self.generator_fn(
                                                 "test"),
                                             validation_steps=self.test_size / self.batch_size)

        self.autoencoder_model.save(self.autoencoder_model_file)

        self._plot_history(history, os.path.join(
            self.models_path, "ae_validation_plot.png"))

    def encode(self, data_path, dest_file, chunk_size=1000, input_dataset='V'):
        """Take data .h5 and produce an encoded data.

        Args:
            data_path(string): a path to .h5 file.
            dataset(string): The name of the dataset in the .h5 file to encode(default: 'V')
        """
        self.autoencoder_model = load_model(self.autoencoder_model_file, custom_objects={
                                            "NanMaskingLayer": NanMaskingLayer,
                                            "filtered_loss_function": self.filtered_loss_function})

        self.autoencoder_model.summary()

        self.__log.debug("Number of layers in model: %d" %
                         len(self.autoencoder_model.layers))

        index = int(len(self.autoencoder_model.layers) / 2)

        self.encoder = Model(self.autoencoder_model.input,
                             self.autoencoder_model.layers[index].output)

        self.encoder.summary()

        encoded_data = DataSignature(dest_file)

        with h5py.File(dest_file, "w") as results, h5py.File(data_path, 'r') as hf:
            input_size = hf[input_dataset].shape[0]
            if "keys" in hf.keys():
                results.create_dataset('keys', data=hf["keys"][
                                       :], maxshape=hf["keys"].shape)
            results.create_dataset('V', (input_size, self.autoencoder_model.layers[
                                   index].output_shape[1]), dtype=np.float32,
                                   maxshape=(input_size, self.autoencoder_model.layers[index].output_shape[1]))

            for i in range(0, input_size, chunk_size):
                chunk = slice(i, i + chunk_size)

                results['V'][chunk] = self.encoder.predict(
                    hf[input_dataset][chunk])

        return encoded_data

    def generator_fn(self, split):
        """Generate an input function for the Estimator.

        Args:
            split(str): the split to use within the traintest file.
        """
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

        iterator = dataset.repeat().make_one_shot_iterator()
        return iterator

    def masked_mse(self, mask_value):
        def f(y_true, y_pred):
            mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
            masked_squared_error = K.square(mask_true * (y_true - y_pred))
            masked_mse = K.sum(masked_squared_error, axis=-
                               1) / K.maximum(K.sum(mask_true, axis=-1), 1)
            return masked_mse
        return f

    def filtered_loss_function(self, y_true, y_pred):
        nans = tf.is_nan(y_true)
        masked_y_true = tf.where(nans, y_pred, y_true)
        filtered = tf.losses.mean_squared_error(masked_y_true, y_pred)
        return filtered

    def _plot_history(self, history, destination=None):
        plt.figure(figsize=(4, 4), dpi=600)
        plt.plot(history.history["loss"], label="Train loss", lw=1)
        plt.plot(history.history["val_loss"], label="Test loss", lw=1)
        plt.legend(loc='best')
        if destination is not None:
            plt.savefig(destination)
