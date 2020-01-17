from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
import os

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop

import tensorflow as tf

from keras import backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from chemicalchecker.util import logged

from chemicalchecker.util.splitter import PairTraintest
from chemicalchecker.core.sign4 import subsample


@logged
class Siamese(object):
    """Siamese class"""

    def __init__(self, models_path, **kwargs):
        """Initialize the AutoEncoder class

        Args:
            models_path(str): Directorty where models will be stored.
            batch_size(int): The batch size for the NN (default=128)
            epochs(int): The number of epochs (default: 200)
        """

        self.epochs = int(kwargs.get("epochs", 15))
        self.batch_size = int(kwargs.get("batch_size", 128))
        self.learning_rate = int(kwargs.get("learning_rate", 0.001))
        self.replace_nan = int(kwargs.get("replace_nan", 0))

        self.models_path = os.path.abspath(models_path)
        if not os.path.exists(models_path):
            self.__log.warning(
                "Specified models directory does not exist: %s",
                self.models_path)
            os.mkdir(self.models_path)

        self.siamese_model_file = os.path.join(self.models_path, "siamese.h5")
        self.siamese = None
        self.transformer = None


    def create_base_network(self, input_shape):
        '''Create network architecture'''
        input = Input(shape=input_shape)
        x = Dense(3200, activation='relu')(input) # 1024
        x = Dropout(0.1)(x)
        x = Dense(1024, activation='relu')(x) # 1024
        x = Dropout(0.1)(x)
        x = Dense(512, activation='relu')(x) # 512
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        return Model(input, x)


      
    def euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))


    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)


    def contrastive_loss(self, y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


    def accuracy(self, y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


    def build_siamese(self, input_shape):

        base_network = self.create_base_network(input_shape)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = Lambda(self.euclidean_distance,
                  output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])

        siamese = Model([input_a, input_b], distance)

        self.__log.debug("Num of layers: %d" % len(siamese.layers))
        return siamese



    def fit(self, data_path, use_geterator=True):

        shapes, dtypes, gen = PairTraintest.generator_fn(data_path, 'train_train', batch_size=self.batch_size, replace_nan=self.replace_nan, augmentation_fn=subsample, augmentation_kwargs=dict(one_dataset=[False]*5 + [True] + [False]*19))

        self.input_shape = (shapes[0][1],)
        
        self.siamese = self.build_siamese(self.input_shape)
        self.siamese.summary()

        rms = RMSprop(learning_rate=self.learning_rate)
        self.siamese.compile(loss=self.contrastive_loss, optimizer=rms, metrics=[self.accuracy])

        history = self.siamese.fit_generator(generator=gen(), epochs=self.epochs, steps_per_epoch=np.ceil(shapes[0][0] / self.batch_size))

        self.siamese.save(self.siamese_model_file)

        self._plot_history(history, os.path.join(self.models_path, "siamese_validation_plot.png"))

    """
    def compute_accuracy(generator):
        y_pred = self.siamese.predict_generator(generator)
        pred = y_pred.ravel() < 0.5
        return np.mean(pred == y_true)


    def evaluate(self, data_path):
        
        generator_tr_te = 0
        generator_te_te = 0

        acc_tr_te = compute_accuracy(generator_tr_te)
        self.__log.debug("Accuracy Tr_Te: %f" % acc_tr_te)

        acc_te_te = compute_accuracy(generator_te_te)
        self.__log.debug("Accuracy Tr_Te: %f" % acc_tr_te)
    """

    def predict(self, data_path, dest_file, chunk_size=1000, input_dataset='V'):
        """Take data .h5 and produce an encoded data.

        Args:
            data_path(string): a path to input .h5 file.
            dest_file(string): a path to output .h5 file.
            chunk_size(int): numbe rof inputs at each prediction.
            dataset(string): The name of the dataset in the .h5 file to encode(default: 'V')
        """

        self.siamese = load_model(self.autoencoder_model_file)
        self.transformer = self.siamese.layers[2]
        
        with h5py.File(dest_file, "w") as results, h5py.File(data_path, 'r') as hf:
            input_size = hf[input_dataset].shape[0]
            if "keys" in hf.keys():
                results.create_dataset('keys', data=hf["keys"][
                                       :], maxshape=hf["keys"].shape)
            results.create_dataset('V', (input_size, self.transformer.output_shape[1]), dtype=np.float32,
                                   maxshape=(input_size, self.transformer.output_shape[1]))

            for i in range(0, input_size, chunk_size):
                chunk = slice(i, i + chunk_size)

                results['V'][chunk] = self.transformer.predict(hf[input_dataset][chunk])


    def _plot_history(self, history, destination=None):
        plt.figure(figsize=(8, 4), dpi=600)
        plt.subplot(1, 2, 1)
        plt.title('Loss evolution')
        plt.plot(history.history["loss"], label="Train loss", lw=1)
        plt.subplot(1, 2, 2)
        plt.title('Accuracy evolution')
        plt.plot(history.history["accuracy"], label="Test loss", lw=1)
        plt.ylim(0,1)
        plt.legend(loc='best')
        if destination is not None:
            plt.savefig(destination)
