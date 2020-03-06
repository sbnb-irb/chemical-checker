import os
import h5py
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
import json
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from chemicalchecker.util import logged
from chemicalchecker.util.splitter import AE_SiameseTraintest
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

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'mask_value': self.mask_value
        })
        return config


class CustomRegularization(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(CustomRegularization, self).__init__(**kwargs)

    def latent_loss(self, x1, x2, z1, z2, batch_size):
        out = abs(tf.losses.cosine_distance(x1, x2, axis=0, reduction=tf.losses.Reduction.MEAN) -
                  tf.losses.cosine_distance(z1, z2, axis=0, reduction=tf.losses.Reduction.MEAN)) / batch_size
        # print out
        return out

    def call(self, x, mask=None, batch_size=128):
        x1 = x[0]
        x2 = x[1]
        z1 = x[2]
        z2 = x[3]

        # If mask is not None, we replace the missing data with the data in the pair if the pair does not contain
        # missing data in those components. Otherwise, we put zeros. Thereby, only the components that do not
        # contain missing data in both elements of the pair are used to calculate the loss.
        if mask is not None:
            mask_value = 0.0
            nan_idxs = tf.is_nan(x1)
            replace = tf.ones_like(x1) * mask_value
            x1_masked = tf.where(nan_idxs, x2, x1)
            nan_idxs = tf.is_nan(x1_masked)
            x1_masked = tf.where(nan_idxs, replace, x1_masked)

            nan_idxs = tf.is_nan(x2)
            replace = tf.ones_like(x2) * mask_value
            x2_masked = tf.where(nan_idxs, x1, x2)
            nan_idxs = tf.is_nan(x2_masked)
            x2_masked = tf.where(nan_idxs, replace, x2_masked)

        else:
            x1_masked = x1
            x2_masked = x2

        loss = self.latent_loss(x1_masked, x2_masked, z1, z2, batch_size)
        self.add_loss(loss, x)
        # you can output whatever you need, just update output_shape adequately
        # But this is probably useful
        return loss

    def get_output_shape_for(self, input_shape):
        return input_shape


@logged
class AutoEncoderSiamese:
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
        self.loss = kwargs.get("loss", self.filtered_loss_function)
        self.activation = kwargs.get("activation", tf.nn.tanh)
        self.activation_last = kwargs.get("activation_last", tf.nn.tanh)
        self.epochs = int(kwargs.get("epochs", 200))
        self.dropout_rate = float(kwargs.get("dropout_rate", 0.2))
        self.shuffle = kwargs.get("shuffle", True)
        self.gpu = kwargs.get("gpu", False)
        self.mask_value = kwargs.get("mask_value", None)
        self.cpu = kwargs.get("cpu", 32)
        self.input_dataset = kwargs.get("input_dataset", 'x')

        self.autoencoder_model_file = os.path.join(
            self.models_path, "autoencoder.h5")

        self.encoder_model_file = os.path.join(
            self.models_path, "encoder.h5")
        self.encoder_model_file_json = os.path.join(
            self.models_path, "encoder.json")

        self.encoder = None
        self.decoder = None

    def fit(self, data_path):
        """Take data .h5 and learn an encoder.

        Args:
            data_path(string): a path to .h5 file.
        """
        self.data_path = data_path

        if not os.path.isfile(self.data_path) or self.data_path[-3:] != '.h5':
            raise Exception("Input data needs to be a H5 file")

        self.traintest_file = os.path.join(self.models_path, 'traintest.h5')

        with h5py.File(data_path, 'r') as hf:
            if self.input_dataset not in hf.keys():
                raise Exception(
                    "Input data file needs to have a dataset called " + self.input_dataset)

            if self.mask_value is not None:
                self.mask_value = np.nanmean(hf[self.input_dataset][:], axis=0)
                self.__log.debug(self.mask_value)

        if not os.path.isfile(self.traintest_file):
            AE_SiameseTraintest.split_h5_blocks(self.data_path, self.traintest_file, split_names=[
                'train', 'test'], split_fractions=[.8, .2], input_datasets=[self.input_dataset])

        (x_shape, y_shape), dtypes, generator_fn_train = AE_SiameseTraintest.generator_fn(
            self.traintest_file, 'train', self.batch_size, shuffle=self.shuffle)

        steps_per_epoch = int(np.ceil(x_shape[0] / self.batch_size))
        self.input_dimension = x_shape[1]

        (x_shape, y_shape), dtypes, generator_fn_test = AE_SiameseTraintest.generator_fn(
            self.traintest_file, 'test', self.batch_size, shuffle=False)

        steps_per_epoch_val = int(np.ceil(x_shape[0] / self.batch_size))

        if self.gpu:
            self.__log.debug("Running on GPUs")
            config = tf.ConfigProto(device_count={'GPU': 1})

        else:

            config = tf.ConfigProto(intra_op_parallelism_threads=self.cpu,
                                    inter_op_parallelism_threads=self.cpu,
                                    allow_soft_placement=True,
                                    device_count={'CPU': self.cpu})
        session = tf.Session(config=config)
        K.set_session(session)

        input_dim_left = Input(shape=(self.input_dimension, ))
        input_dim_right = Input(shape=(self.input_dimension, ))

        if self.input_dimension <= self.encoding_dim:
            raise Exception("Input dimension smaller than desired reduction")

        self.num_middle_layers = 1

        # Calculate the reduction rate and then divide by two.
        # The number gives us the number of middle layers including the latent
        # space
        reduc_rate = np.floor(self.input_dimension / self.encoding_dim)

        self.num_middle_layers = int(np.ceil(reduc_rate / 2))

        if self.num_middle_layers == 1:
            self.num_middle_layers += 1

        layer_sizes = np.linspace(
            self.input_dimension, self.encoding_dim, self.num_middle_layers + 1)[1:]

        self.__log.debug("Num of layers: %d" % len(layer_sizes))

        encoder_net = Sequential()

        if self.mask_value is not None:
            self.__log.debug("mask value is not none")
            self.loss = self.filtered_loss_function
            encoder_net.add(NanMaskingLayer(mask_value=self.mask_value))

        for i, layer_size in enumerate(layer_sizes):
            encoder_net.add(
                Dense(int(layer_size), activation=self.activation))

        self.encoded_left = encoder_net(input_dim_left)
        self.encoded_right = encoder_net(input_dim_right)

        last_layer_left = self.encoded_left
        last_layer_right = self.encoded_right

        for layer_size in np.flip(layer_sizes, 0):
            if layer_size == self.encoding_dim:
                continue
            last_layer_left = Dense(
                int(layer_size), activation=self.activation)(last_layer_left)
            last_layer_right = Dense(
                int(layer_size), activation=self.activation)(last_layer_right)

        self.decoder_left = Dense(self.input_dimension,
                                  activation=self.activation_last)(last_layer_left)

        self.decoder_right = Dense(self.input_dimension,
                                   activation=self.activation_last)(last_layer_right)

        cr = CustomRegularization()(
            [input_dim_left, input_dim_right, self.encoded_left, self.encoded_right], mask=self.mask_value, batch_size=self.batch_size)

        self.autoencoder_model = Model(
            inputs=[input_dim_left, input_dim_right], outputs=[self.decoder_left, self.decoder_right, cr])
        self.autoencoder_model.compile(
            optimizer=self.optimizer, loss=[self.loss, self.loss, None])

        self.autoencoder_model.summary()

        history = self.autoencoder_model.fit_generator(generator_fn_train(), epochs=self.epochs,
                                                       steps_per_epoch=steps_per_epoch,
                                                       validation_data=generator_fn_test(),
                                                       validation_steps=steps_per_epoch_val)

        self.autoencoder_model.save(self.autoencoder_model_file)
        encoder_net.save(self.encoder_model_file)

        json_data = {}
        json_data["layer_sizes"] = layer_sizes.tolist()
        json_data["input_dimension"] = self.input_dimension
        if self.mask_value is not None:
            json_data["mask_value"] = self.mask_value.tolist()
        else:
            json_data["mask_value"] = None

        with open(self.encoder_model_file_json, 'w') as outfile:
            json.dump(json_data, outfile)

        self._plot_history(history, os.path.join(
            self.models_path, "ae_validation_plot.png"))

    def encode(self, data_path, dest_file, chunk_size=1000, input_dataset='V'):
        """Take data .h5 and produce an encoded data.

        Args:
            data_path(string): a path to .h5 file.
            dataset(string): The name of the dataset in the .h5 file to encode(default: 'V')
        """

        # Work around for issue:
        # https://github.com/tensorflow/tensorflow/issues/28668
        with open(self.encoder_model_file_json) as json_file:
            json_data = json.load(json_file)

        input_dim = Input(shape=(json_data["input_dimension"], ))
        encoder_net_loaded = Sequential()
        if json_data["mask_value"] is not None:
            self.__log.debug("mask value is not none")
            encoder_net_loaded.add(NanMaskingLayer(
                np.asarray(json_data["mask_value"])))

        for i, layer_size in enumerate(json_data["layer_sizes"]):

            encoder_net_loaded.add(
                Dense(int(layer_size), activation=self.activation))

        encoder_net_loaded(input_dim)
        encoder_net_loaded.load_weights(self.encoder_model_file)
        encoder_net_loaded.summary()

        encoded_data = DataSignature(dest_file)

        with h5py.File(dest_file, "w") as results, h5py.File(data_path, 'r') as hf:
            input_size = hf[input_dataset].shape[0]
            if "keys" in hf.keys():
                results.create_dataset('keys', data=hf["keys"][
                                       :], maxshape=hf["keys"].shape)
            results.create_dataset('V', (input_size, self.encoding_dim), dtype=np.float32,
                                   maxshape=(input_size, self.encoding_dim))

            for i in range(0, input_size, chunk_size):
                chunk = slice(i, i + chunk_size)

                results['V'][chunk] = encoder_net_loaded.predict(
                    hf[input_dataset][chunk])

        return encoded_data

    def filtered_loss_function(self, y_true, y_pred):
        nans = tf.is_nan(y_true)
        masked_y_true = tf.where(nans, y_pred, y_true)
        filtered = abs(tf.losses.cosine_distance(
            masked_y_true, y_pred, axis=0, reduction=tf.losses.Reduction.MEAN))
        return filtered

    def _plot_history(self, history, destination=None):
        plt.figure(figsize=(4, 4), dpi=600)
        plt.plot(np.log10(history.history["loss"]),
                 label="log10 Train loss", lw=1)
        plt.plot(np.log10(history.history[
                 "val_loss"]), label="log10 Test loss", lw=1)
        plt.legend(loc='best')
        if destination is not None:
            plt.savefig(destination)
